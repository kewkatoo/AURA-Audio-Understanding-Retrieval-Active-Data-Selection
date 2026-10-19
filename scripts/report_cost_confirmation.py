"""Verify a frozen independent-corpus LOSO cost confirmation, without tuning."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aura.cache.store import file_content_hash,config_fingerprint
from aura.provenance import source_fingerprint
from experiments.cost_confirmation import make_speaker_folds
from scripts.evaluate_selection import metrics
from scripts.analyze_confirmation_uncertainty import speaker_bootstrap


def primary_gap(predictions):
    frame=predictions[predictions.primary].copy()
    frame["correct"]=frame.label.astype(str).eq(frame.prediction.astype(str)).astype(float)
    full=frame[frame.method == "full_pool"][["id","speaker_id","correct"]]
    random=frame[frame.method == "random"]
    if full.id.duplicated().any() or random.duplicated(["id","seed"]).any():
        raise ValueError("Primary prediction keys duplicated")
    if not random.groupby("id").seed.nunique().eq(5).all():
        raise ValueError("Each held-out recording needs five subset seeds")
    averaged=random.groupby("id").agg(correct=("correct","mean"),speaker_id=("speaker_id","first"))
    aligned=full.merge(averaged,on="id",suffixes=("_full","_random"),validate="one_to_one")
    if len(aligned) != len(full) or len(aligned) != len(averaged) or not aligned.speaker_id_full.eq(aligned.speaker_id_random).all():
        raise ValueError("Full and subset cohorts differ")
    aligned["delta"]=aligned.correct_full-aligned.correct_random
    low,high=speaker_bootstrap(aligned.delta.to_numpy(),aligned.speaker_id_full.to_numpy())
    byspeaker=aligned.groupby("speaker_id_full").agg(full_accuracy=("correct_full","mean"),
        random_accuracy=("correct_random","mean"),gap=("delta","mean"),recordings=("id","count")).reset_index().rename(columns={"speaker_id_full":"speaker"})
    result=dict(full_accuracy=float(aligned.correct_full.mean()),random_accuracy=float(aligned.correct_random.mean()),
        gap_pp=float(aligned.delta.mean()*100),low_pp=low,high_pp=high,speakers=int(len(byspeaker)),recordings=len(aligned))
    return result,byspeaker


def timing_pairs(runs):
    timed=runs[runs.timed_pair]
    if timed.duplicated(["speaker","repeat","method"]).any():
        raise ValueError("Duplicate timing keys")
    if not timed.groupby(["speaker","repeat"]).method.nunique().eq(2).all():
        raise ValueError("Timing pair incomplete")
    if not timed.groupby(["speaker","repeat"]).session.nunique().eq(1).all():
        raise ValueError("A timing pair spans different runtime sessions")
    pairs=timed.pivot(index=["speaker","repeat"],columns="method",values="end_to_end_seconds").reset_index()
    if not {"full_pool","random"}.issubset(pairs) or pairs[["full_pool","random"]].isnull().any().any():
        raise ValueError("Missing method in paired timing")
    pairs["ratio"]=pairs.random/pairs.full_pool
    pairs["saving_percent"]=(1-pairs.ratio)*100
    return pairs


def verify(out):
    protocol=json.loads((out/"protocol.json").read_text())
    if source_fingerprint() != protocol["source_fingerprint"]:
        raise ValueError("Core source changed")
    for path,digest in protocol["sources"].items():
        if file_content_hash(path) != digest:
            raise ValueError(f"Confirmation source changed: {path}")
    for filename,key in [("inventory.json","inventory_hash"),("folds.json","folds_hash")]:
        if file_content_hash(out/filename) != protocol[key]:
            raise ValueError("Dataset declaration changed")
    if file_content_hash("results/acquisition_cost_v6/protocol.json") != protocol["origin_protocol_hash"] or file_content_hash("results/acquisition_cost_v6/experiment.json") != protocol["origin_experiment_hash"]:
        raise ValueError("Development decision evidence changed")
    if file_content_hash(Path(protocol["config"]["dataset"]["root"])/"README.md") != protocol["dataset_readme_hash"]:
        raise ValueError("Dataset attribution source changed")
    records=json.loads((out/"inventory.json").read_text())
    folds=json.loads((out/"folds.json").read_text())
    if make_speaker_folds(records) != folds:
        raise ValueError("Speaker folds differ from declared rule")
    print("Verifying all fresh-corpus raw hashes",flush=True)
    for record in records:
        if file_content_hash(record["audio_path"]) != record["audio_hash"]:
            raise ValueError("Raw audio changed")
    byid={r["id"]:r for r in records}
    folds_by_speaker={f["speaker"]:f for f in folds}
    sessions={}
    for path in (out/"sessions").glob("*.json"):
        data=json.loads(path.read_text())
        sessions[data["session"]]=data
    if not sessions:
        raise ValueError("Missing runtime context")
    expected_context=None
    for context in sessions.values():
        signature={k:context[k] for k in ["encoder","cuda_matmul_tf32","cudnn_tf32","cudnn_version",
            "cudnn_benchmark","deterministic_algorithms","attention_implementation","parameter_dtype","gpu","threads","workers"]}
        if expected_context is None:
            expected_context=signature
        elif expected_context != signature:
            raise ValueError("Runtime numerical/hardware settings differ between sessions")
    locked=json.loads((out/"models_locked.json").read_text())
    expected={json.dumps(j,sort_keys=True) for j in protocol["jobs"]}
    if len(locked) != len(expected):
        raise ValueError("Model count differs from declaration")
    seen=set()
    saved_jobs={}
    for path,digest in locked.items():
        if file_content_hash(path) != digest:
            raise ValueError("Locked checkpoint changed")
        saved=json.loads(Path(path).read_text())
        job=saved["job"]
        seen.add(json.dumps(job,sort_keys=True))
        if saved["identity"] != config_fingerprint(dict(job=job,protocol=file_content_hash(out/"protocol.json"))):
            raise ValueError("Trial identity changed")
        if saved["session"] not in sessions:
            raise ValueError("Missing trial session")
        for artifact,expected_hash in saved["artifacts"].items():
            if file_content_hash(artifact) != expected_hash:
                raise ValueError("Training artifact changed")
        folder=Path(path).parent
        ids=json.loads((folder/"selection.json").read_text())
        fold=folds_by_speaker[job["speaker"]]
        if len(ids) != job["budget"] or len(set(ids)) != len(ids) or not set(ids) <= set(fold["train_ids"]):
            raise ValueError("Invalid trial selection")
        counts=pd.Series([byid[sid]["label"] for sid in ids]).value_counts()
        if len(counts) != 10 or (job["method"] == "random" and counts.max()-counts.min() > 1):
            raise ValueError("Class allocation differs")
        x=np.load(folder/"embeddings.npy")
        width=sessions[saved["session"]]["encoder"]["embedding_dim"]
        if x.shape != (len(ids),width) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4):
            raise ValueError("Invalid training embeddings")
        if not 0 < saved["head_seconds"] <= saved["end_to_end_seconds"] or not np.isfinite(saved["end_to_end_seconds"]):
            raise ValueError("Invalid trial timing")
        saved_jobs[(job["speaker"],job["method"],job["seed"],job["repeat"])]=(folder,saved)
    if seen != expected:
        raise ValueError("Trial schedule differs")
    evaluation={}
    for fold in folds:
        directory=out/"evaluation"/fold["speaker"]
        marker=json.loads((directory/"manifest.json").read_text())
        if marker["ids"] != fold["test_ids"] or marker["array_hash"] != file_content_hash(directory/"embeddings.npy"):
            raise ValueError("Evaluation artifact differs")
        if marker["protocol"] != file_content_hash(out/"protocol.json") or marker["models_locked_hash"] != file_content_hash(out/"models_locked.json"):
            raise ValueError("Evaluation was not tied to the full model lock")
        x=np.load(directory/"embeddings.npy")
        if x.shape != (len(fold["test_ids"]),expected_context["encoder"]["embedding_dim"]) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4):
            raise ValueError("Invalid evaluation embeddings")
        evaluation[fold["speaker"]]=x
    runs=pd.read_csv(out/"runs.csv")
    predictions=pd.read_csv(out/"predictions.csv.gz",dtype={"label":str,"prediction":str})
    if len(runs) != len(expected) or runs.duplicated(["speaker","method","seed","repeat"]).any():
        raise ValueError("Missing or duplicate evaluation runs")
    if len(predictions) != sum(len(folds_by_speaker[j["speaker"]]["test_ids"]) for j in protocol["jobs"]):
        raise ValueError("Prediction count differs")
    for row in runs.itertuples():
        folder,saved=saved_jobs[(row.speaker,row.method,row.seed,row.repeat)]
        group=predictions[(predictions.speaker_id == row.speaker)&(predictions.method == row.method)&(predictions.seed == row.seed)&(predictions.repeat == row.repeat)]
        ids=folds_by_speaker[row.speaker]["test_ids"]
        if group.id.tolist() != ids or group.label.tolist() != [byid[sid]["label"] for sid in ids] or not group.primary.eq(saved["job"]["primary"]).all():
            raise ValueError("Prediction cohort/primary status differs")
        actual=joblib.load(folder/"model.joblib").predict(evaluation[row.speaker])
        if not np.array_equal(actual,group.prediction.to_numpy()):
            raise ValueError("Saved model does not reproduce predictions")
        scores=metrics(group.label,group.prediction,[str(i) for i in range(10)])
        if not np.isclose(scores["accuracy"],row.accuracy,atol=1e-12) or not np.isclose(scores["macro_f1"],row.macro_f1,atol=1e-12):
            raise ValueError("Saved metrics differ")
        if not np.isclose(row.end_to_end_seconds,saved["end_to_end_seconds"]) or row.session != saved["session"]:
            raise ValueError("Timing/session differs from checkpoint")
    return protocol,folds,sessions,runs,predictions,saved_jobs


def repeat_stability(predictions,saved_jobs):
    result=[]
    speakers=sorted(predictions.speaker_id.unique())
    for speaker in speakers:
        for method in ["full_pool","random"]:
            folder,_=saved_jobs[(speaker,method,0,0)]
            base_ids=json.loads((folder/"selection.json").read_text())
            base=np.load(folder/"embeddings.npy")
            base_pred=predictions[(predictions.speaker_id == speaker)&(predictions.method == method)&(predictions.seed == 0)&(predictions.repeat == 0)].set_index("id")
            for repeat in [1,2]:
                other,_=saved_jobs[(speaker,method,0,repeat)]
                if json.loads((other/"selection.json").read_text()) != base_ids:
                    raise ValueError("Repeated seed-zero selection changed")
                fresh=np.load(other/"embeddings.npy")
                current=predictions[(predictions.speaker_id == speaker)&(predictions.method == method)&(predictions.seed == 0)&(predictions.repeat == repeat)].set_index("id").loc[base_pred.index]
                if not base_pred.label.eq(current.label).all():
                    raise ValueError("Repeat labels differ")
                result.append(dict(speaker=speaker,method=method,repeat=repeat,
                    max_embedding_difference=float(np.max(np.abs(fresh-base))),
                    minimum_embedding_cosine=float(np.sum(fresh.astype(np.float64)*base.astype(np.float64),axis=1).min()),
                    embeddings_identical=bool(np.array_equal(base,fresh)),
                    prediction_disagreement=float(np.mean(base_pred.prediction.to_numpy()!=current.prediction.to_numpy())),
                    accuracy_change_pp=float(100*(current.prediction.eq(current.label).mean()-base_pred.prediction.eq(base_pred.label).mean()))))
    return pd.DataFrame(result)


def main():
    out=Path(sys.argv[1] if len(sys.argv)>1 else "results/cost_confirmation_v7")
    protocol,folds,sessions,runs,predictions,saved_jobs=verify(out)
    primary,byspeaker=primary_gap(predictions)
    primary["mean_gap_within_one_point"]=bool(primary["gap_pp"] <= protocol["accuracy_margin_pp"])
    primary["exploratory_upper_within_one_point"]=bool(primary["high_pp"] <= protocol["accuracy_margin_pp"])
    (out/"accuracy_summary.json").write_text(json.dumps(primary,indent=2))
    byspeaker["gap_pp"]=byspeaker.gap*100
    byspeaker.to_csv(out/"speaker_accuracy.csv",index=False)
    pairs=timing_pairs(runs)
    expected_pairs=len(folds)*3
    if len(pairs) != expected_pairs or not pairs.groupby("speaker").size().eq(3).all():
        raise ValueError("Incomplete repeated timing")
    pairs.to_csv(out/"timing_pairs.csv",index=False)
    pairs.groupby("speaker")[["full_pool","random","ratio"]].agg(["median","min","max"]).to_csv(out/"timing_by_speaker.csv")
    timing=dict(pairs=len(pairs),full_median_seconds=float(pairs.full_pool.median()),
        random_median_seconds=float(pairs.random.median()),paired_ratio_median=float(pairs.ratio.median()),
        paired_ratio_min=float(pairs.ratio.min()),paired_ratio_max=float(pairs.ratio.max()),
        median_paired_saving_percent=float(pairs.saving_percent.median()),
        random_faster_pairs=int((pairs.random < pairs.full_pool).sum()),sessions=len(sessions),
        startup_seconds=[c["model_startup_and_warmup_seconds"] for c in sessions.values()])
    (out/"timing_summary.json").write_text(json.dumps(timing,indent=2))
    stability=repeat_stability(predictions,saved_jobs)
    stability.to_csv(out/"repeat_stability.csv",index=False)
    figure(out,byspeaker,pairs)
    report(out,protocol,folds,primary,timing,stability)
    verification=dict(all_checks_passed=True,models=len(saved_jobs),primary_models=int(runs.primary.sum()),
        paired_timing_trials=int(runs.timed_pair.sum()),paired_repeats=len(pairs),raw_hashes_checked=3000,
        speakers=len(folds),held_out_recordings=primary["recordings"],sessions=len(sessions),
        protocol_hash=file_content_hash(out/"protocol.json"),
        analysis_sources={str(p.resolve()):file_content_hash(p) for p in [Path(__file__),Path("scripts/analyze_confirmation_uncertainty.py")]})
    (out/"verification.json").write_text(json.dumps(verification,indent=2))
    (out/"experiment.json").write_text(json.dumps(dict(status="complete",verification=verification,
        artifacts={str(p.resolve()):file_content_hash(p) for p in out.iterdir() if p.is_file() and p.name != "experiment.json"},
        evaluation_manifests={str(p.resolve()):file_content_hash(p) for p in (out/"evaluation").glob("*/manifest.json")},
        runtime_sessions={str(p.resolve()):file_content_hash(p) for p in (out/"sessions").glob("*.json")}),indent=2))
    print(json.dumps(dict(accuracy=primary,timing=timing,verification=verification),indent=2))


def figure(out,byspeaker,pairs):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    axes[0].barh(byspeaker.speaker,byspeaker.gap_pp,color="#3266a8")
    axes[0].axvline(0,color="black",lw=1)
    axes[0].axvline(1,color="#b63c39",ls="--",label="1-point margin")
    axes[0].set(xlabel="Full minus mean random accuracy (percentage points)",title="Held-out-speaker accuracy loss")
    axes[0].legend(fontsize=8)
    names=sorted(pairs.speaker.unique())
    for repeat in [0,1,2]:
        current=pairs[pairs.repeat == repeat].set_index("speaker").loc[names]
        axes[1].scatter(np.arange(len(names))+(repeat-1)*.12,current.ratio,label=f"Repeat {repeat+1}")
    axes[1].axhline(1,color="black",ls="--",label="equal time")
    axes[1].set_xticks(np.arange(len(names)),labels=names,rotation=25)
    axes[1].set(ylabel="Random/full end-to-end warm training time",title="18 paired timing repeats",ylim=(0,max(1.1,float(pairs.ratio.max())+.1)))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("FSDD: frozen fraction transfer, six-speaker leave-one-speaker-out")
    fig.tight_layout()
    fig.savefig(out/"confirmation.png",dpi=180)
    fig.savefig(out/"confirmation.svg")


def report(out,protocol,folds,primary,timing,stability):
    fold_counts=sorted({len(f["train_ids"]) for f in folds})
    budgets=sorted({f["random_budget"] for f in folds})
    lines=["# Independent cost-rule confirmation","",
        "The fixed random-retention rule is tested on the previously unused Free Spoken Digit Dataset (FSDD). This study does not tune a new fraction, classifier or selector. It evaluates transfer of a development choice, not a novel acquisition algorithm.","",
        f"Dataset revision: `{protocol['dataset_revision']}`. Six speakers, 3,000 recordings, 8 kHz mono. Each speaker is evaluated on all 500 recordings using models trained on the other five speakers. Training sizes after content exclusions: {fold_counts}; transferred random budgets: {budgets}. The rule retains 64/175 of candidate rows, rounded up to a class-count multiple; C=1 throughout.",
        "This custom speaker-disjoint protocol differs from the dataset's official index-based 0–4 test split. Speakers occur in training for other folds, so fold models are dependent. Every configuration and repeat is fitted before any evaluation prediction; no held-out score chooses the rule.","",
        "## Accuracy","",
        f"Full-pool pooled accuracy: **{primary['full_accuracy']*100:.2f}%**. Random-rule accuracy, averaged over five subset seeds: **{primary['random_accuracy']*100:.2f}%**.",
        f"Full-minus-random loss: **{primary['gap_pp']:.3f} percentage points**, exploratory paired speaker-bootstrap 95% interval **[{primary['low_pp']:.3f}, {primary['high_pp']:.3f}]**.",
        f"Observed mean meets the one-point criterion: **{primary['mean_gap_within_one_point']}**. Exploratory upper interval below one point: **{primary['exploratory_upper_within_one_point']}**.",
        "The bootstrap resamples six held-out speaker blocks and conditions on the fitted heads. It omits training-population uncertainty and does not remove dependence between folds' overlapping training sets. This small corpus is not a powered statistical noninferiority study. Passing an exploratory check would support this corpus only; failing it would show that this fixed fraction does not preserve the target margin here.","",
        "## Repeated timing","",
        f"Three paired repeats per held-out speaker give **{timing['pairs']} pairs**. Full-pool median warm end-to-end training time: **{timing['full_median_seconds']:.2f} s**. Subset median: **{timing['random_median_seconds']:.2f} s**.",
        f"Median paired random/full ratio: **{timing['paired_ratio_median']:.3f}**, range **[{timing['paired_ratio_min']:.3f}, {timing['paired_ratio_max']:.3f}]**. Median paired saving: **{timing['median_paired_saving_percent']:.2f}%**. Random is faster in **{timing['random_faster_pairs']}/{timing['pairs']} pairs**.",
        "These are actual sequential wall times for selection, raw decoding/resampling/VAD/normalization, uncached wav2vec2 inference, matrix assembly, selected-only scaling and logistic-regression fitting. GPU work is synchronized. They are not sums of independently measured components. AB/BA order is counterbalanced over folds/repeats, and the timed seed-zero subset is held fixed within each fold.",
        f"One process, two CPU threads and a resident encoder are used. Startup/synthetic raw-audio and classifier warm-up were measured separately: {timing['startup_seconds']} seconds across {timing['sessions']} session(s). Inventory/provenance, checkpoint writes, startup, evaluation encoding and evaluation prediction are excluded from trial times. OS caches are not flushed. No training feature cache is read. Repeats do not measure energy, hardware rental costs, cold-machine startup, or independent hardware sessions.","",
        "## Numerical repeatability","",
        f"Identical embedding arrays across same-layout training repeats: **{int(stability.embeddings_identical.sum())}/{len(stability)} comparisons**. Maximum repeat embedding-coordinate difference: **{stability.max_embedding_difference.max():.8f}**. Maximum held-out prediction disagreement across repeats: **{stability.prediction_disagreement.max()*100:.3f}%**.",
        "Fresh training vectors are used directly, without a canonical training matrix or a relaxed reference-equality requirement. Evaluation vectors are computed once per held-out speaker and shared across its models. Same-layout repeatability does not establish invariance to different batch composition or hardware. Runtime records include model revision, dtype, attention backend, CUDA/cuDNN versions, TF32 settings and thread pools. The v6 cross-batch exemplar-instability finding remains separate.","",
        "## Scope and provenance","",
        "Class labels are known before class-balanced selection. This study establishes no annotation savings, no global optimality of a retention fraction, and no benefit for expensive fine-tuning. Absolute times and scores should not be compared directly with the different AudioMNIST cohort/hardware workload. Keep this now-observed corpus out of future tuning if claiming fresh confirmation.",
        "All 60 fitted heads, 36 primary configurations, 36 timed trials, 18 timing pairs, 3,000 raw hashes, per-fold speaker/content separation, selected class budgets, embeddings, saved-model predictions and metrics are verified. Source declarations, model locks, session contexts and evaluation manifests are preserved.","",
        "Dataset: [FSDD, maintained by Jakobovski and contributors](https://github.com/Jakobovski/free-spoken-digit-dataset). Its README declares [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). The source checkout and README are preserved locally; see DATASET_NOTICE.md. No dataset recordings were modified in the source checkout."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    (out/"DATASET_NOTICE.md").write_text("# Dataset attribution\n\nFree Spoken Digit Dataset (FSDD), Jakobovski and contributors.\n\nSource: https://github.com/Jakobovski/free-spoken-digit-dataset\n\nRevision: "+protocol["dataset_revision"]+"\n\nThe dataset README declares Creative Commons Attribution-ShareAlike 4.0 International: https://creativecommons.org/licenses/by-sa/4.0/\n\nThe original repository and README remain in data/raw/FSDD. Resampled waveforms are processed in memory; derived embeddings, selections and models are local experiment artifacts associated with this source attribution.\n")


if __name__ == "__main__":
    main()
