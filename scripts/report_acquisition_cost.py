"""Verify and report the development-only, warm-runtime cost study."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aura.cache.store import file_content_hash,config_fingerprint
from aura.provenance import source_fingerprint
from experiments.scaling import load_scaling_artifact
from scripts.evaluate_selection import metrics,assert_disjoint
from scripts.report_independent_study import paired_comparisons
from scripts.study_acquisition_cost import embedding_drift
from scripts.analyze_confirmation_uncertainty import speaker_bootstrap


def frontier(table):
    result=table.copy()
    result["dominated"]=False
    for i,row in result.iterrows():
        better=(result.mean_accuracy >= row.mean_accuracy) & (result.total_seconds <= row.total_seconds)
        strict=(result.mean_accuracy > row.mean_accuracy) | (result.total_seconds < row.total_seconds)
        result.loc[i,"dominated"]=bool((better & strict).any())
    full=result[result.method == "full_pool"].iloc[0]
    result["gap_to_full_pp"]=(full.mean_accuracy-result.mean_accuracy)*100
    result["within_one_point_full"]=result.gap_to_full_pp <= 1.0 + 1e-10
    result["observed_cost_ratio_to_full"]=result.total_seconds/full.total_seconds
    return result


def full_gap_intervals(predictions):
    table=predictions.copy()
    table["correct"]=table.prediction.astype(str).eq(table.label.astype(str)).astype(float)
    full=table[table.method == "full_pool"][["id","speaker_id","correct"]]
    if full.id.duplicated().any():
        raise ValueError("Full reference has duplicate IDs")
    rows=[]
    for (method,budget),group in table[table.method != "full_pool"].groupby(["method","budget"]):
        averaged=group.groupby("id").agg(correct=("correct","mean"),speaker_id=("speaker_id","first"))
        joined=full.merge(averaged,on="id",suffixes=("_full","_subset"),validate="one_to_one")
        if len(joined) != len(full) or not joined.speaker_id_full.eq(joined.speaker_id_subset).all():
            raise ValueError("Full/subset cohorts differ")
        delta=(joined.correct_full-joined.correct_subset).to_numpy()
        low,high=speaker_bootstrap(delta,joined.speaker_id_full.to_numpy())
        rows.append(dict(method=method,budget=int(budget),gap_pp=float(delta.mean()*100),low_pp=low,high_pp=high))
    return pd.DataFrame(rows)


def verify(out):
    protocol=json.loads((out/"protocol.json").read_text())
    if source_fingerprint() != protocol["source_fingerprint"]:
        raise ValueError("Core pipeline changed")
    for path,digest in protocol["sources"].items():
        if file_content_hash(path) != digest:
            raise ValueError(f"Changed source: {path}")
    if file_content_hash(out/"partition.json") != protocol["partition_hash"]:
        raise ValueError("Partition changed")
    parent=Path("data/audiomnist_v5")
    if file_content_hash(parent/"metadata/train_manifest.json") != protocol["parent_train_manifest_hash"]:
        raise ValueError("Parent candidate artifact changed")
    meta,original,_=load_scaling_artifact(parent,"train")
    partition=json.loads((out/"partition.json").read_text())
    indexed=meta.set_index("id",drop=False)
    train=indexed.loc[[r["sample_id"] for r in partition["train"]]].reset_index(drop=True)
    dev=indexed.loc[[r["sample_id"] for r in partition["holdout"]]].reset_index(drop=True)
    assert_disjoint(train,dev)
    if {r["audio_hash"] for r in partition["train"]} & {r["audio_hash"] for r in partition["holdout"]}:
        raise ValueError("Content overlap")
    print("Verifying development-source audio hashes",flush=True)
    for record in partition["train"]+partition["holdout"]:
        if file_content_hash(record["audio_path"]) != record["audio_hash"]:
            raise ValueError("Raw data changed")
    feature_info=json.loads((out/"source_mfcc.json").read_text())
    if feature_info["hash"] != file_content_hash(out/"source_mfcc.npy"):
        raise ValueError("MFCC artifact changed")
    np.testing.assert_allclose(np.load(out/"source_mfcc.npy"),
        np.concatenate([np.stack(train.mfcc_mean),np.stack(train.mfcc_std)],axis=1),atol=1e-5,rtol=1e-5)
    neural_info=json.loads((out/"full_neural.json").read_text())
    neural=np.load(out/"full_neural.npy")
    if neural_info["array_hash"] != file_content_hash(out/"full_neural.npy") or neural_info["ids"] != train.id.tolist():
        raise ValueError("Neural timing artifact changed")
    positions=dict(zip(meta.id,range(len(meta))))
    canonical=original[[positions[sid] for sid in train.id]]
    drift=embedding_drift(neural,canonical)
    if drift["minimum_reference_cosine"] < .99:
        raise ValueError("Fresh full-pool angular audit failed")
    for key,value in drift.items():
        if not np.isclose(value,neural_info[key]):
            raise ValueError("Fresh drift statistics changed")
    for path,digest in json.loads((out/"preencoding_locked.json").read_text()).items():
        if file_content_hash(path) != digest:
            raise ValueError("Pre-encoding selection changed")
    lock=json.loads((out/"models_locked.json").read_text())
    expected={(method,budget,seed) for method in protocol["methods"] for budget in
              (protocol["random_budgets"] if method == "random" else protocol["budgets"]) for seed in protocol["seeds"]}
    expected.add(("full_pool",len(train),0))
    seen=set()
    train_positions=dict(zip(train.id,range(len(train))))
    for path,digest in lock.items():
        if file_content_hash(path) != digest:
            raise ValueError("Locked model manifest changed")
        saved=json.loads(Path(path).read_text())
        job=saved["job"]
        if saved["identity"] != config_fingerprint(dict(job=job,protocol=file_content_hash(out/"protocol.json"))):
            raise ValueError("Model identity changed")
        seen.add((job["method"],job["budget"],job["seed"]))
        for artifact,hash_value in saved["artifacts"].items():
            if file_content_hash(artifact) != hash_value:
                raise ValueError("Model artifact changed")
        ids=json.loads((Path(path).parent/"selection.json").read_text())
        if len(ids) != job["budget"] or len(set(ids)) != len(ids) or not set(ids) <= set(train.id):
            raise ValueError("Invalid selection")
        if job["method"] != "full_pool":
            counts=train.set_index("id").loc[ids].label.value_counts()
            if len(counts) != train.label.nunique() or counts.max()-counts.min() > 1:
                raise ValueError("Class budget changed")
        if job["method"] in {"random","mfcc_representative"}:
            pre=json.loads((out/"preencoding"/f"{Path(path).parent.name}.json").read_text())
            if pre["ids"] != ids:
                raise ValueError("Model selection differs from pre-encoding declaration")
            if job["seed"] == 0:
                array=out/f"replay_{job['method']}_B{job['budget']}.npy"
                replay=json.loads(array.with_suffix(".json").read_text())
                if replay["ids"] != ids or replay["array_hash"] != file_content_hash(array):
                    raise ValueError("Timed selected pass changed")
                audit=embedding_drift(np.load(array),canonical[[train_positions[sid] for sid in ids]])
                if audit["minimum_reference_cosine"] < .99:
                    raise ValueError("Selected replay angular audit failed")
                for key,value in audit.items():
                    if not np.isclose(value,replay[key]):
                        raise ValueError("Selected replay drift statistics changed")
    if expected != seen:
        raise ValueError("Missing or extra model jobs")
    runs=pd.read_csv(out/"runs.csv")
    predictions=pd.read_csv(out/"predictions.csv.gz",dtype={"label":str,"prediction":str})
    if len(runs) != len(seen) or len(predictions) != len(dev)*len(seen):
        raise ValueError("Evaluation incomplete")
    for row in runs.itertuples():
        group=predictions[(predictions.method == row.method)&(predictions.budget == row.budget)&(predictions.seed == row.seed)]
        if group.id.tolist() != dev.id.tolist() or group.label.tolist() != dev.label.astype(str).tolist():
            raise ValueError("Prediction cohort changed")
        scores=metrics(group.label,group.prediction,sorted(train.label.unique()))
        if not np.isclose(scores["accuracy"],row.accuracy,atol=1e-12) or not np.isclose(scores["macro_f1"],row.macro_f1,atol=1e-12):
            raise ValueError("Reported scores differ")
    costs=pd.read_csv(out/"costs.csv")
    if costs.duplicated(["method","budget"]).any() or set(zip(costs.method,costs.budget)) != {(m,b) for m,b,s in expected if s == 0}:
        raise ValueError("Missing or duplicate cost jobs")
    components=["descriptor_seconds","encoding_seconds","encoder_setup_seconds","selection_seconds","fit_seconds"]
    if not np.isfinite(costs[components].to_numpy()).all() or costs[components].lt(0).any().any():
        raise ValueError("Invalid cost components")
    np.testing.assert_allclose(costs[components].sum(axis=1),costs.total_seconds)
    context=json.loads((out/"timing_context.json").read_text())
    for cost in costs.itertuples():
        folder=out/"jobs"/f"{cost.method}_B{cost.budget}_seed0"
        job_info=json.loads((folder/"manifest.json").read_text())
        encoding=(neural_info if cost.method in {"wav2vec_representative","full_pool"} else
            json.loads((out/f"replay_{cost.method}_B{cost.budget}.json").read_text()))
        descriptor=feature_info["seconds"]+context["descriptor_normalization_seconds"] if cost.method == "mfcc_representative" else 0
        np.testing.assert_allclose([cost.descriptor_seconds,cost.encoding_seconds,cost.encoder_setup_seconds,cost.selection_seconds,cost.fit_seconds],
            [descriptor,encoding["seconds"],context["encoder_setup_seconds"],job_info["selection_seconds"],job_info["fit_seconds"]])
    sensitivity=pd.read_csv(out/"numerical_sensitivity.csv")
    if set(zip(sensitivity.method,sensitivity.budget)) != {(m,b) for m,b,s in expected if s == 0}:
        raise ValueError("Numerical sensitivity audit incomplete")
    for row in sensitivity.itertuples():
        canonical=runs[(runs.method == row.method)&(runs.budget == row.budget)&(runs.seed == 0)].accuracy.iloc[0]
        if not np.isclose(row.canonical_seed0_accuracy,canonical):
            raise ValueError("Sensitivity baseline differs from locked model")
    values=sensitivity[["canonical_seed0_accuracy","fresh_seed0_accuracy","prediction_disagreement","selected_jaccard"]]
    if not np.isfinite(values.to_numpy()).all() or values.lt(0).any().any() or values.gt(1).any().any():
        raise ValueError("Invalid numerical sensitivity statistics")
    return runs,predictions,costs,dict(all_checks_passed=True,models=len(seen),training_records=len(train),
        development_records=len(dev),training_speakers=int(train.speaker_id.nunique()),
        development_speakers=int(dev.speaker_id.nunique()),raw_hashes_checked=len(train)+len(dev),
        old_holdout_audio_representations_or_predictions_used=False,protocol_hash=file_content_hash(out/"protocol.json"))


def main():
    out=Path(sys.argv[1] if len(sys.argv)>1 else "results/acquisition_cost_v6")
    runs,predictions,costs,verification=verify(out)
    summary=runs.groupby(["method","budget"]).agg(mean_accuracy=("accuracy","mean"),seed_std=("accuracy","std"),
        mean_macro_f1=("macro_f1","mean")).reset_index()
    table=frontier(summary.merge(costs,on=["method","budget"],validate="one_to_one"))
    table.to_csv(out/"accuracy_cost.csv",index=False)
    paired=paired_comparisons(predictions.rename(columns={"method":"strategy"}),["random","wav2vec_representative"])
    paired.to_csv(out/"speaker_bootstrap.csv",index=False)
    gap_intervals=full_gap_intervals(predictions)
    gap_intervals.to_csv(out/"full_gap_bootstrap.csv",index=False)
    colors={"random":"#2b8c55","mfcc_representative":"#2468bd","wav2vec_representative":"#d8543d"}
    full=table[table.method == "full_pool"].iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for method,color in colors.items():
        group=table[table.method == method].sort_values("budget")
        label=method.replace("_"," ")
        axes[0].plot(group.budget,group.mean_accuracy*100,"o-",color=color,label=label)
        axes[1].plot(group.total_seconds,group.mean_accuracy*100,"o-",color=color,label=label)
        for row in group.itertuples():
            axes[1].annotate(str(row.budget),(row.total_seconds,row.mean_accuracy*100),fontsize=7,xytext=(3,3),textcoords="offset points")
    axes[0].axhline(full.mean_accuracy*100,color="black",ls="--",label="full pool")
    axes[1].scatter([full.total_seconds],[full.mean_accuracy*100],marker="*",s=140,color="black",label="full pool")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_ylabel("Development accuracy (%)")
        ax.grid(alpha=.2)
    axes[0].set_xlabel("Selected recordings (log scale)")
    axes[1].set_xlabel("Measured component total, seconds (log scale)")
    axes[0].legend(fontsize=8)
    fig.suptitle("AudioMNIST training-pool development study; no fresh confirmation")
    fig.tight_layout()
    fig.savefig(out/"accuracy_cost.png",dpi=180)
    fig.savefig(out/"accuracy_cost.svg")
    lines=["# Development accuracy and acquisition cost","",
        "This study uses only the former AudioMNIST candidate training pool: 35 candidate speakers and 10 development speakers. Earlier holdout audio, representations and predictions are not used. Parent split declarations are read for provenance. These are development results, not fresh independent confirmation.","",
        "All heads use C=1. MFCC selection uses 13 source MFCC means and 13 standard deviations, candidate-only standardization, unit normalization, and the same class-wise KMeans exemplars. Five seeds estimate accuracy; cost replays use seed 0 only. Random subsets at 3,200 and 6,400 provide additional cost controls.","",
        "| Method | Recordings | Mean accuracy | Total seconds | Gap to full (points) | Dominated on tested mean/cost grid |",
        "|---|---:|---:|---:|---:|---|"]
    for row in table.sort_values(["method","budget"]).itertuples():
        lines.append(f"| {row.method} | {row.budget} | {row.mean_accuracy*100:.2f}% | {row.total_seconds:.2f} | {row.gap_to_full_pp:.2f} | {row.dominated} |")
    eligible=table[table.within_one_point_full & (table.method != "full_pool")].sort_values("total_seconds")
    lines += ["",f"Full-pool reference: {full.mean_accuracy*100:.2f}% at {full.total_seconds:.2f} measured component seconds."]
    if len(eligible):
        best=eligible.iloc[0]
        lines += [f"Cheapest tested subset within one percentage point of the full mean: {best.method}, budget {best.budget}, {best.mean_accuracy*100:.2f}%, {best.total_seconds:.2f} seconds ({best.observed_cost_ratio_to_full:.3f} times the full-pool component total). This development choice needs fresh confirmation."]
        interval=gap_intervals[(gap_intervals.method == best.method)&(gap_intervals.budget == best.budget)].iloc[0]
        lines += [f"Its full-minus-subset accuracy gap is {interval.gap_pp:.3f} points; exploratory paired speaker-bootstrap 95% interval [{interval.low_pp:.3f}, {interval.high_pp:.3f}]. Check the interval against the one-point margin; the mean criterion alone does not establish noninferiority."]
    else:
        lines += ["No tested subset is within one percentage point of the full-pool mean. This experiment therefore does not demonstrate matched-full-accuracy savings."]
    lines += ["","## What the timing includes","",
        "One sequential process with two CPU threads and CUDA encoding. Fresh raw decoding, resampling/preprocessing, feature computation and neural inference bypass persistent feature/embedding caches. MFCC and random selections are locked before any candidate neural encoding in this run. Fresh neural outputs are checked against the frozen representations.",
        "A recipe total adds model setup/warm-up, required descriptor extraction/normalization, required raw preprocessing plus encoding, selection, and head fitting. Full-pool and wav2vec selection share one measured all-candidate primitive; random and MFCC have separate selected-only raw encoding replays. Accuracy uses canonical frozen training/development embeddings for consistency. These are measured component totals, not independent complete pipeline replications.",
        "Python/library initialization, inventory/provenance checks, construction/comparison of benchmark-only canonical reference matrices, checkpoint serialization, common evaluation preparation/prediction, and previously completed downloads are excluded. OS file caches and loaded library kernels are warm. Model weights come from the local cache; setup and a synthetic neural warm-up are explicitly included in each recipe total. Costs are not energy, hardware rental prices, or cold-machine startup measurements.",
        "Timing has one replay per subset configuration and no timing uncertainty interval. Component order, warm caches and thermal state can affect the observations. A globally optimal cost frontier is not established by this finite grid.","",
        "The reported frontier allows different retained-recording budgets. A method dominated on that grid can still be relevant when there is an additional hard limit on retained recordings. Fixed-budget accuracy and cross-budget cost are different decision criteria.","",
        "## Numerical audit","",
        "Before development scoring, a strict per-coordinate fresh/reference check failed. A candidate-only control reproduced differences between the same clip encoded alone and in an equal-length batch, while repeating the same single-clip call was identical. [PyTorch documents that batch shapes can change floating-point results](https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html). This is not the earlier padding defect: the control uses identical lengths and no padding.",
        "The protocol was amended before any development predictions. The stopped pass and original runner are preserved in acquisition_cost_v6_strict_check. Current accuracy/selection uses canonical frozen embeddings. Fresh uncached timed vectors are checked for shape, ID alignment, finiteness, unit norm and a predeclared minimum cosine of 0.99 to the corresponding canonical vector; coordinate and cosine drift are recorded, not treated as numerical identity.",
        f"Full-pass maximum coordinate difference: {json.loads((out/'full_neural.json').read_text())['max_reference_error']:.6f}; minimum cosine: {json.loads((out/'full_neural.json').read_text())['minimum_reference_cosine']:.6f}. See full_neural.json and replay JSONs for all measured differences.",
        "After all timing, seed-zero shadow fits use fresh training vectors and, for wav2vec representatives, fresh-vector reselection. They are evaluated on the same canonical development embeddings. numerical_sensitivity.csv reports accuracy changes, prediction disagreement and subset overlap. This is a sensitivity diagnostic, not independent full-pipeline evaluation. Precision/backend settings should enter future artifact contracts when strict numerical reproduction is required.","",
        "## Limits and next decision","",
        "The same known candidate labels support class-balanced selection in every method. No annotation-acquisition savings are established. A cost improvement at lower accuracy must be stated together with that accuracy loss. Cross-budget comparisons use means across five fixed seeds while the timed replay uses seed 0.",
        "Speaker-bootstrap differences condition on the fixed models, resample only the 10 development speakers, and have no multiplicity correction. The within-one-point check compares observed means and is not a statistical noninferiority test. All development scores are now observed; method changes and model choices require new confirmation data before a general performance claim.",
        "See accuracy_cost.csv, costs.csv, speaker_bootstrap.csv, timing_context.json, protocol.json and verification.json for the measured components and provenance."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    verification["analysis_sources"]={str(p.resolve()):file_content_hash(p) for p in
        [Path(__file__),Path("scripts/report_independent_study.py"),Path("scripts/analyze_confirmation_uncertainty.py")]}
    (out/"verification.json").write_text(json.dumps(verification,indent=2))
    (out/"experiment.json").write_text(json.dumps(dict(status="complete",verification=verification,
        artifacts={str(p.resolve()):file_content_hash(p) for p in out.iterdir() if p.is_file() and p.name != "experiment.json"}),indent=2))
    print(table[["method","budget","mean_accuracy","total_seconds","gap_to_full_pp","dominated"]].to_string(index=False))


if __name__ == "__main__":
    main()
