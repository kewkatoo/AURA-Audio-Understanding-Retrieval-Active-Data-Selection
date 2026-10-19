#!/usr/bin/env python
"""Prospective FSDD LOSO confirmation with repeated uncached warm-model timing.

The transferred 64/175 fraction and C=1 are fixed. No tuning or old holdouts.
All configurations and timing repeats finish before evaluation encoding or
prediction. Each speaker appears in other folds' training, as standard LOSO.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="2"
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from threadpoolctl import threadpool_info
from aura.cache.store import file_content_hash,config_fingerprint
from aura.config import AuraConfig
from aura.encoders.wav2vec2 import Wav2Vec2Encoder
from aura.provenance import source_fingerprint
from aura.selection.random import RandomSelector
from experiments.cost_confirmation import make_speaker_folds,paired_order
from experiments.raw_pipeline import encode_recordings
from scripts.evaluate_selection import fit_probe,metrics

REVISION="26eb9aaf76e81b692f806f9140c2d2777410d7a1"


def immutable(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Existing declaration differs: {path}")
    else:
        path.write_text(json.dumps(payload,indent=2))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def inventory(repository):
    revision=subprocess.check_output(["git","-C",str(repository),"rev-parse","HEAD"],text=True).strip()
    if revision != REVISION:
        raise ValueError("Dataset revision differs")
    records=[]
    for path in sorted((repository/"recordings").glob("*.wav")):
        digit,speaker,index=path.stem.split("_")
        info=sf.info(path)
        if info.samplerate != 8000 or info.channels != 1 or info.frames*2 < 400:
            raise ValueError("Unexpected source audio format/length")
        records.append(dict(id="fsdd/"+path.stem,audio_path=str(path.resolve()),label=digit,
            speaker_id=speaker,utterance=int(index),audio_hash=file_content_hash(path),
            frames=info.frames,sample_rate=info.samplerate))
    if len(records) != 3000 or len({r["speaker_id"] for r in records}) != 6:
        raise ValueError("Unexpected FSDD inventory")
    return records


def jobs_for(folds):
    timed=[]
    for i,fold in enumerate(folds):
        for repeat in range(3):
            for method in paired_order(i,repeat):
                timed.append(dict(speaker=fold["speaker"],method=method,seed=0,repeat=repeat,
                    budget=len(fold["train_ids"]) if method == "full_pool" else fold["random_budget"],
                    timed_pair=True,primary=repeat == 0,C=1.0))
    extra=[dict(speaker=f["speaker"],method="random",seed=seed,repeat=0,
        budget=f["random_budget"],timed_pair=False,primary=True,C=1.0) for f in folds for seed in [1,2,3,4]]
    return timed+extra


def train_trial(job,train,encoder,config,out,signature,session):
    folder=out/"jobs"/f"{job['speaker']}_{job['method']}_seed{job['seed']}_rep{job['repeat']}"
    folder.mkdir(parents=True,exist_ok=True)
    marker=folder/"manifest.json"
    identity=config_fingerprint(dict(job=job,protocol=signature))
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved["identity"] != identity or any(file_content_hash(p) != h for p,h in saved["artifacts"].items()):
            raise ValueError("Changed trial checkpoint")
        return folder
    sync()
    started=time.perf_counter()
    if job["method"] == "full_pool":
        selected=train
    else:
        ids=RandomSelector().select(train,{},job["budget"],job["seed"]).selected_ids
        selected=train.set_index("id",drop=False).loc[ids].reset_index(drop=True)
    matrix=encode_recordings(selected,encoder,config.audio)
    model,head_seconds=fit_probe(matrix,selected.label.to_numpy(),job["C"])
    sync()
    elapsed=time.perf_counter()-started
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-4):
        raise ValueError("Invalid trial embeddings")
    (folder/"selection.json").write_text(json.dumps(selected.id.tolist()))
    np.save(folder/"embeddings.npy",matrix)
    joblib.dump(model,folder/"model.joblib")
    marker.write_text(json.dumps(dict(identity=identity,job=job,end_to_end_seconds=elapsed,
        head_seconds=head_seconds,session=session,
        artifacts={str(p.resolve()):file_content_hash(p) for p in
            [folder/"selection.json",folder/"embeddings.npy",folder/"model.joblib"]}),indent=2))
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",default="results/cost_confirmation_v7")
    args=parser.parse_args()
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"experiment.json").exists():
        raise ValueError("Completed confirmation is preserved; use a new output directory")
    config=AuraConfig.from_yaml("configs/experiment.v2.yaml")
    config=replace(config,dataset=replace(config.dataset,name="FSDD_LOSO",root="data/raw/FSDD",
        classes=[str(i) for i in range(10)],sample_limit=replace(config.dataset.sample_limit,train=None,validation=None,test=None)))
    repository=Path(config.dataset.root)
    records=inventory(repository)
    folds=make_speaker_folds(records)
    immutable(out/"inventory.json",records)
    immutable(out/"folds.json",folds)
    sources=[Path(__file__),Path("experiments/raw_pipeline.py"),Path("experiments/cost_confirmation.py"),
        Path("scripts/evaluate_selection.py")]
    protocol=dict(dataset_url="https://github.com/Jakobovski/free-spoken-digit-dataset",dataset_revision=REVISION,
        dataset_license="CC BY-SA 4.0",dataset_readme_hash=file_content_hash(repository/"README.md"),
        inventory_hash=file_content_hash(out/"inventory.json"),folds_hash=file_content_hash(out/"folds.json"),
        source_fingerprint=source_fingerprint(),sources={str(p.resolve()):file_content_hash(p) for p in sources},
        config=asdict(config),jobs=jobs_for(folds),retention_rule="64/175 of training rows, rounded up to a multiple of the class count",
        rule_origin="AudioMNIST v6 development: 6400/17500; no FSDD tuning",
        origin_protocol_hash=file_content_hash("results/acquisition_cost_v6/protocol.json"),
        origin_experiment_hash=file_content_hash("results/acquisition_cost_v6/experiment.json"),
        primary="pooled held-out-speaker full accuracy minus mean of 5 random-subset seed accuracies",
        accuracy_margin_pp=1.0,uncertainty="exploratory 2000-replicate speaker-block bootstrap; 6 speakers, overlapping fold training sets; not a powered noninferiority trial",
        timing="3 paired uncached complete selection/decode/preprocess/encode/head-fit repeats per fold; alternate AB/BA; one process, two CPU threads, warm resident encoder",
        exclusions="shared inventory/provenance, model startup, synthetic raw-audio and classifier warm-up, checkpoint writes, shared evaluation encoding/prediction",
        numerical="fresh vectors used for actual training; fixed batch procedure; repeats assessed after model locking",
        packages={name:version(name) for name in ["numpy","scipy","pandas","soundfile","librosa","torch","torchaudio","transformers","scikit-learn"]})
    immutable(out/"protocol.json",protocol)
    signature=file_content_hash(out/"protocol.json")
    meta=pd.DataFrame(records).set_index("id",drop=False)
    train_frames={f["speaker"]:meta.loc[f["train_ids"]].reset_index(drop=True).assign(split="train") for f in folds}
    session=datetime.now(timezone.utc).isoformat()
    started=time.perf_counter()
    encoder=Wav2Vec2Encoder(checkpoint=config.encoder.checkpoint,pooling=config.encoder.pooling,
        normalize=config.encoder.normalize,device=config.encoder.device,batch_size=config.encoder.batch_size)
    warmup_path=out/"synthetic_warmup.wav"
    sf.write(warmup_path,(.1*np.sin(2*np.pi*440*np.arange(4000)/8000)).astype(np.float32),8000)
    encode_recordings(pd.DataFrame(dict(id=["synthetic"],audio_path=[str(warmup_path.resolve())])),encoder,config.audio)
    fit_probe(np.random.default_rng(0).normal(size=(20,encoder.info.embedding_dim)),np.array(["0","1"]*10),1.0)
    sync()
    setup_seconds=time.perf_counter()-started
    context=dict(session=session,encoder=asdict(encoder.info),model_startup_and_warmup_seconds=setup_seconds,
        cuda_matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
        cudnn_version=torch.backends.cudnn.version(),cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        attention_implementation=getattr(encoder.model.config,"_attn_implementation",None),
        parameter_dtype=str(next(encoder.model.parameters()).dtype),threadpools=threadpool_info(),
        torch_cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        python=sys.version,threads=2,workers=1)
    session_path=out/"sessions"/(session.replace(":","-")+".json")
    session_path.parent.mkdir(exist_ok=True)
    session_path.write_text(json.dumps(context,indent=2))
    folders=[]
    for i,job in enumerate(protocol["jobs"],1):
        folder=train_trial(job,train_frames[job["speaker"]],encoder,config,out,signature,session)
        folders.append(folder)
        timing=json.loads((folder/"manifest.json").read_text())["end_to_end_seconds"]
        print(f"Models {i}/{len(protocol['jobs'])}: {folder.name}, {timing:.2f}s",flush=True)
    immutable(out/"models_locked.json",{str((f/"manifest.json").resolve()):file_content_hash(f/"manifest.json") for f in folders})
    # Evaluation begins only after the entire predeclared fit/timing schedule.
    evaluation={}
    for fold in folds:
        directory=out/"evaluation"/fold["speaker"]
        directory.mkdir(parents=True,exist_ok=True)
        test=meta.loc[fold["test_ids"]].reset_index(drop=True)
        marker=directory/"manifest.json"
        if marker.exists():
            saved=json.loads(marker.read_text())
            if saved["ids"] != test.id.tolist() or saved["protocol"] != signature or saved["array_hash"] != file_content_hash(directory/"embeddings.npy"):
                raise ValueError("Changed evaluation artifact")
            matrix=np.load(directory/"embeddings.npy")
        else:
            matrix=encode_recordings(test,encoder,config.audio)
            np.save(directory/"embeddings.npy",matrix)
            marker.write_text(json.dumps(dict(ids=test.id.tolist(),protocol=signature,
                models_locked_hash=file_content_hash(out/"models_locked.json"),array_hash=file_content_hash(directory/"embeddings.npy")),indent=2))
        evaluation[fold["speaker"]]=(test,matrix)
        print(f"Encoded held-out speaker {fold['speaker']}: {len(test)}",flush=True)
    runs,predictions=[] , []
    for folder in folders:
        saved=json.loads((folder/"manifest.json").read_text())
        job=saved["job"]
        test,matrix=evaluation[job["speaker"]]
        predicted=joblib.load(folder/"model.joblib").predict(matrix)
        runs.append(dict(**job,**metrics(test.label,predicted,config.dataset.classes),
            end_to_end_seconds=saved["end_to_end_seconds"],head_seconds=saved["head_seconds"],session=saved["session"]))
        predictions.append(pd.DataFrame(dict(id=test.id,speaker_id=test.speaker_id,label=test.label,prediction=predicted,
            method=job["method"],seed=job["seed"],repeat=job["repeat"],primary=job["primary"])))
    pd.DataFrame(runs).to_csv(out/"runs.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/"predictions.csv.gz",index=False,compression="gzip")
    print("Finished all declared trials and held-out evaluation; run scripts/report_cost_confirmation.py",flush=True)


if __name__ == "__main__":
    main()
