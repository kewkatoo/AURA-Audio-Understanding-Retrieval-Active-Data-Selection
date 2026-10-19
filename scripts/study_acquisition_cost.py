#!/usr/bin/env python
"""Development-only accuracy/cost study; never opens the observed v5 holdout.

Times real uncached computation with one process/two CPU threads. Reported
recipe costs add measured components; they are not independent end-to-end
repetitions. Evaluation representations are shared and excluded from training
cost. OS/model caches are warm, persistent feature/embedding caches bypassed.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import torch
from aura.cache.store import file_content_hash, config_fingerprint
from aura.config import AuraConfig
from aura.encoders.wav2vec2 import Wav2Vec2Encoder
from aura.ingest.audio import process_audio
from aura.provenance import source_fingerprint
from aura.selection.random import RandomSelector
from experiments.acoustic import source_mfcc, normalized_descriptors
from experiments.representative import RepresentativeSelector
from experiments.scaling import partition_corpus, load_scaling_artifact
from scripts.evaluate_selection import fit_probe, metrics, assert_disjoint
from scripts.analyze_confirmation_uncertainty import speaker_bootstrap

METHODS = ["random", "mfcc_representative", "wav2vec_representative"]
BUDGETS = [100, 400, 1600]
SEEDS = list(range(5))
RANDOM_BUDGETS = [100, 400, 1600, 3200, 6400]


def immutable(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Existing declaration differs: {path}")
    else:
        path.write_text(json.dumps(data, indent=2))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def embedding_drift(matrix, reference):
    if matrix.shape != reference.shape or not np.isfinite(matrix).all():
        raise ValueError("Invalid fresh embedding shape/values")
    if not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-4):
        raise ValueError("Fresh embeddings are not unit vectors")
    cosine=np.sum(matrix.astype(np.float64)*reference.astype(np.float64),axis=1)
    differences=np.max(np.abs(matrix-reference),axis=1)
    return dict(max_reference_error=float(differences.max()),minimum_reference_cosine=float(cosine.min()),
        median_reference_cosine=float(np.median(cosine)),mean_reference_cosine=float(cosine.mean()),
        rows_above_1e_3_max_abs=int((differences>1e-3).sum()))


def encoded_pass(metadata, encoder, config, reference, output):
    """Real decode/preprocess/encoding pass, with no persistent cache reads."""
    marker = output.with_suffix(".json")
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["ids"] != metadata.id.tolist() or saved["array_hash"] != file_content_hash(output):
            raise ValueError("Changed timed encoding artifact")
        if saved["minimum_reference_cosine"] < .99:
            raise ValueError("Fresh embedding angular audit failed")
        return np.load(output), saved
    sync()
    started = time.perf_counter()
    arrays = []
    for start in range(0, len(metadata), 500):
        frame = metadata.iloc[start:start+500]
        waveforms = [process_audio(path, target_sample_rate=config.audio.sample_rate,
            do_mono=config.audio.mono, do_normalize=config.audio.normalize,
            do_vad=config.audio.vad, do_trim=config.audio.vad,
            vad_frame_ms=config.audio.vad_frame_ms, vad_hop_ms=config.audio.vad_hop_ms,
            vad_energy_percentile=config.audio.vad_energy_percentile,
            vad_noise_margin_db=config.audio.vad_noise_margin_db,
            vad_min_rms=config.audio.vad_min_rms).waveform for path in frame.audio_path]
        order = sorted(range(len(waveforms)), key=lambda i: len(waveforms[i]))
        chunk = np.empty((len(frame), encoder.info.embedding_dim), dtype=np.float32)
        for offset in range(0, len(order), config.encoder.batch_size):
            batch = order[offset:offset+config.encoder.batch_size]
            chunk[batch] = encoder.encode_batch([waveforms[i] for i in batch], config.audio.sample_rate)
        arrays.append(chunk)
        print(f"{output.stem}: {min(start+500,len(metadata))}/{len(metadata)}", flush=True)
    sync()
    seconds = time.perf_counter() - started
    matrix = np.concatenate(arrays)
    drift=embedding_drift(matrix,reference)
    np.save(output, matrix)
    saved = dict(ids=metadata.id.tolist(), seconds=seconds, records=len(metadata),
                 **drift, array_hash=file_content_hash(output))
    marker.write_text(json.dumps(saved, indent=2))
    if drift["minimum_reference_cosine"] < .99:
        raise ValueError("Fresh embedding angular audit failed; diagnostic array saved")
    return matrix, saved


def preselect(train, acoustic, out, signature):
    """No neural representation is accepted at this acquisition boundary."""
    paths = []
    lookup = dict(zip(train.id, acoustic))
    for method in ["random", "mfcc_representative"]:
        for budget in RANDOM_BUDGETS if method == "random" else BUDGETS:
            for seed in SEEDS:
                path = out/"preencoding"/f"{method}_B{budget}_seed{seed}.json"
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved["signature"] != signature:
                        raise ValueError("Changed pre-encoding selection protocol")
                else:
                    started = time.perf_counter()
                    selector = RandomSelector() if method == "random" else RepresentativeSelector()
                    ids = selector.select(train, {} if method == "random" else lookup, budget, seed).selected_ids
                    saved = dict(ids=ids, selection_seconds=time.perf_counter()-started, signature=signature)
                    path.write_text(json.dumps(saved,indent=2))
                paths.append(path)
    immutable(out/"preencoding_locked.json",{str(p.resolve()):file_content_hash(p) for p in paths})


def fit_jobs(train, neural, acoustic, dev, dev_neural, out, signature):
    lookup = dict(zip(train.id, neural))
    positions = dict(zip(train.id, range(len(train))))
    jobs = [dict(method=method, budget=budget, seed=seed, C=1.0)
            for method in METHODS for budget in (RANDOM_BUDGETS if method == "random" else BUDGETS) for seed in SEEDS]
    jobs.append(dict(method="full_pool", budget=len(train), seed=0, C=1.0))
    folders = []
    for number, job in enumerate(jobs, 1):
        folder = out / "jobs" / f"{job['method']}_B{job['budget']}_seed{job['seed']}"
        folder.mkdir(parents=True, exist_ok=True)
        identity = config_fingerprint(dict(job=job, protocol=signature))
        marker = folder / "manifest.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["identity"] != identity or any(file_content_hash(p) != h for p,h in saved["artifacts"].items()):
                raise ValueError("Changed model checkpoint")
        else:
            started = time.perf_counter()
            if job["method"] == "full_pool":
                ids = train.id.tolist()
                selection_seconds = time.perf_counter() - started
            elif job["method"] in {"random", "mfcc_representative"}:
                saved_selection = json.loads((out/"preencoding"/f"{folder.name}.json").read_text())
                ids = saved_selection["ids"]
                selection_seconds = saved_selection["selection_seconds"]
            else:
                ids = RepresentativeSelector().select(train, lookup, job["budget"], job["seed"]).selected_ids
                selection_seconds = time.perf_counter() - started
            rows = [positions[sid] for sid in ids]
            model, fit_seconds = fit_probe(neural[rows], train.label.to_numpy()[rows], job["C"])
            (folder / "selection.json").write_text(json.dumps(ids))
            joblib.dump(model, folder / "model.joblib")
            saved = dict(identity=identity, job=job, selection_seconds=selection_seconds, fit_seconds=fit_seconds,
                artifacts={str(p.resolve()): file_content_hash(p) for p in [folder/"selection.json", folder/"model.joblib"]})
            marker.write_text(json.dumps(saved, indent=2))
        folders.append(folder)
        print(f"Models ready: {number}/{len(jobs)} ({folder.name})", flush=True)
    immutable(out / "models_locked.json", {str((p/"manifest.json").resolve()): file_content_hash(p/"manifest.json") for p in folders})
    # No development predictions are generated until every configuration is fixed.
    runs, predictions = [], []
    for folder in folders:
        saved = json.loads((folder/"manifest.json").read_text())
        job = saved["job"]
        pred = joblib.load(folder/"model.joblib").predict(dev_neural)
        runs.append(dict(**job, **metrics(dev.label, pred, sorted(train.label.unique())),
                         selection_seconds=saved["selection_seconds"], fit_seconds=saved["fit_seconds"]))
        predictions.append(pd.DataFrame(dict(id=dev.id, speaker_id=dev.speaker_id, label=dev.label,
            prediction=pred, method=job["method"], budget=job["budget"], seed=job["seed"])))
    pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    return folders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/acquisition_cost_v6")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "experiment.json").exists():
        raise ValueError("Study is complete; preserve it and choose a new output directory")
    config = AuraConfig.from_yaml("configs/experiment.v2.yaml")
    config = replace(config,dataset=replace(config.dataset,name="AudioMNIST_development",
        root="data/raw/AudioMNIST",classes=[str(i) for i in range(10)],
        sample_limit=replace(config.dataset.sample_limit,train=None,validation=None,test=None)))
    parent = Path("data/audiomnist_v5")
    source, old_neural, manifest = load_scaling_artifact(parent, "train")
    parent_plan = json.loads((parent/"partition.json").read_text())
    partition = partition_corpus(parent_plan["train"], set(), set(), held_out_speakers=10, seed=20261019)
    immutable(out/"partition.json", partition)
    sources = [Path(__file__), Path("experiments/acoustic.py"), Path("experiments/representative.py"),
               Path("scripts/evaluate_selection.py"), Path("experiments/scaling.py")]
    protocol = dict(source_fingerprint=source_fingerprint(),
        sources={str(p.resolve()):file_content_hash(p) for p in sources},
        parent_train_manifest_hash=file_content_hash(parent/"metadata/train_manifest.json"),
        partition_hash=file_content_hash(out/"partition.json"), config=asdict(config),
        methods=METHODS, budgets=BUDGETS, random_budgets=RANDOM_BUDGETS, seeds=SEEDS, classifier_C=1.0,
        scope="development only; 35/10 speakers from previous candidate training pool; no old holdout opened",
        descriptor="13 source MFCC means plus 13 stds; candidate-only StandardScaler then L2 normalization",
        timing="one sequential process, 2 CPU threads; persistent caches bypassed, OS/model warm; component accounting",
        timing_seeds=[0], common_evaluation_cost="excluded from every recipe",
        amendment="before any development scoring: strict coordinate equality replaced by a recorded cosine audit; accuracy uses canonical frozen embeddings, with separate fresh-replay sensitivity checks",
        fresh_audit="finite unit vectors, matching IDs/shapes, minimum cosine to canonical >=0.99; this is not numerical identity",
        primary="accuracy/time frontier; matched-full-accuracy threshold = within 1 percentage point",
        packages={name:version(name) for name in ["numpy","scipy","librosa","torch","transformers","scikit-learn"]})
    immutable(out/"protocol.json", protocol)
    signature = file_content_hash(out/"protocol.json")
    source = source.set_index("id", drop=False)
    train = source.loc[[r["sample_id"] for r in partition["train"]]].reset_index(drop=True)
    dev = source.loc[[r["sample_id"] for r in partition["holdout"]]].reset_index(drop=True)
    dev["split"] = "validation"
    assert_disjoint(train, dev)
    row = dict(zip(source.id, range(len(source))))
    reference = old_neural[[row[sid] for sid in train.id]]
    dev_neural = old_neural[[row[sid] for sid in dev.id]]
    print(f"Development study: candidates={len(train)}, development={len(dev)}", flush=True)
    # Warm library initialization with synthetic audio, not measured candidate data.
    import librosa
    librosa.feature.mfcc(y=np.zeros(16000, dtype=np.float32), sr=16000, n_mfcc=13, n_fft=400, hop_length=160)
    mfcc_path = out/"source_mfcc.npy"
    marker = out/"source_mfcc.json"
    if marker.exists():
        mfcc_info = json.loads(marker.read_text())
        if mfcc_info["hash"] != file_content_hash(mfcc_path):
            raise ValueError("Changed acoustic descriptor artifact")
        features = np.load(mfcc_path)
    else:
        started = time.perf_counter()
        descriptors = []
        for i,path in enumerate(train.audio_path,1):
            descriptors.append(source_mfcc(path))
            if i % 1000 == 0:
                print(f"Cheap features: {i}/{len(train)}", flush=True)
        features = np.stack(descriptors)
        seconds = time.perf_counter() - started
        existing = np.concatenate([np.stack(train.mfcc_mean), np.stack(train.mfcc_std)], axis=1)
        if not np.allclose(features, existing, atol=1e-5, rtol=1e-5):
            raise ValueError("Cheap feature implementation differs from frozen descriptors")
        np.save(mfcc_path,features)
        mfcc_info = dict(seconds=seconds, records=len(train), hash=file_content_hash(mfcc_path))
        marker.write_text(json.dumps(mfcc_info,indent=2))
    started = time.perf_counter()
    scaler, acoustic = normalized_descriptors(features)
    normalization_seconds = time.perf_counter() - started
    joblib.dump(scaler, out/"descriptor_scaler.joblib")
    np.save(out/"acoustic_unit.npy", acoustic)
    preselect(train, acoustic, out, signature)
    print("Cheap/random selections locked before neural encoding",flush=True)
    started = time.perf_counter()
    encoder = Wav2Vec2Encoder(checkpoint=config.encoder.checkpoint, pooling=config.encoder.pooling,
        normalize=config.encoder.normalize, device=config.encoder.device, batch_size=config.encoder.batch_size)
    encoder.encode_batch([np.zeros(16000,dtype=np.float32)],16000)
    sync()
    setup_seconds = time.perf_counter() - started
    if asdict(encoder.info) != manifest["encoder"]:
        raise ValueError("Encoder differs from frozen reference")
    fresh_neural, neural_info = encoded_pass(train, encoder, config, reference, out/"full_neural.npy")
    neural=reference
    folders = fit_jobs(train,neural,acoustic,dev,dev_neural,out,signature)
    costs = []
    for folder in folders:
        saved = json.loads((folder/"manifest.json").read_text())
        job = saved["job"]
        if job["seed"] != 0:
            continue
        descriptor_seconds = (mfcc_info["seconds"] + normalization_seconds) if job["method"] == "mfcc_representative" else 0
        if job["method"] in {"full_pool","wav2vec_representative"}:
            encoding_seconds = neural_info["seconds"]
        else:
            ids = json.loads((folder/"selection.json").read_text())
            selected = train.set_index("id",drop=False).loc[ids].reset_index(drop=True)
            ref = neural[[dict(zip(train.id,range(len(train))))[sid] for sid in ids]]
            _, replay = encoded_pass(selected,encoder,config,ref,out/f"replay_{job['method']}_B{job['budget']}.npy")
            encoding_seconds = replay["seconds"]
        total = setup_seconds + descriptor_seconds + encoding_seconds + saved["selection_seconds"] + saved["fit_seconds"]
        costs.append(dict(**job, descriptor_seconds=descriptor_seconds, encoding_seconds=encoding_seconds,
                          encoder_setup_seconds=setup_seconds, selection_seconds=saved["selection_seconds"],
                          fit_seconds=saved["fit_seconds"], total_seconds=total))
    pd.DataFrame(costs).to_csv(out/"costs.csv",index=False)
    (out/"timing_context.json").write_text(json.dumps(dict(encoder_setup_seconds=setup_seconds,
        descriptor_normalization_seconds=normalization_seconds, encoder=asdict(encoder.info),
        threads=2, workers=1, persistent_cache_reads=False, evaluation_costs_excluded=True),indent=2))
    # Untimed sensitivity checks run only after all recipe timing has finished.
    sensitivity=[]
    train_rows=dict(zip(train.id,range(len(train))))
    fresh_lookup=dict(zip(train.id,fresh_neural))
    for folder in folders:
        saved=json.loads((folder/"manifest.json").read_text())
        job=saved["job"]
        if job["seed"] != 0:
            continue
        canonical_ids=json.loads((folder/"selection.json").read_text())
        canonical_prediction=joblib.load(folder/"model.joblib").predict(dev_neural)
        if job["method"] == "wav2vec_representative":
            ids=RepresentativeSelector().select(train,fresh_lookup,job["budget"],0).selected_ids
            fresh_x=fresh_neural[[train_rows[sid] for sid in ids]]
        elif job["method"] == "full_pool":
            ids=canonical_ids
            fresh_x=fresh_neural
        else:
            ids=canonical_ids
            fresh_x=np.load(out/f"replay_{job['method']}_B{job['budget']}.npy")
        fresh_model,_=fit_probe(fresh_x,train.set_index("id").loc[ids].label.to_numpy(),1.0)
        fresh_prediction=fresh_model.predict(dev_neural)
        sensitivity.append(dict(**job,canonical_seed0_accuracy=float(np.mean(canonical_prediction==dev.label.to_numpy())),
            fresh_seed0_accuracy=float(np.mean(fresh_prediction==dev.label.to_numpy())),
            prediction_disagreement=float(np.mean(canonical_prediction!=fresh_prediction)),
            selected_jaccard=len(set(ids)&set(canonical_ids))/len(set(ids)|set(canonical_ids))))
    pd.DataFrame(sensitivity).to_csv(out/"numerical_sensitivity.csv",index=False)
    context=json.loads((out/"timing_context.json").read_text())
    context.update(cuda_matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
        numerical_sensitivity="seed 0; fresh training/selection versus canonical, evaluated on shared canonical development embeddings")
    (out/"timing_context.json").write_text(json.dumps(context,indent=2))
    print("Study finished; run scripts/report_acquisition_cost.py",flush=True)


if __name__ == "__main__":
    main()
