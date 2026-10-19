# Reproduce and verify the AURA research handoff

Prepared 2026-10-19. Run PowerShell commands from the project root. The saved studies and local environment have been verified; a fresh installation and a second hardware/session replication have not been performed. A dependency capture is not a portable, wheel-hash-locked installer.

## 1. Preserve the evidence before changing anything

Keep the complete workspace source, `results/`, `handoff/`, the original dataset checkouts and prepared `data/` artifacts, and the Hugging Face cache used for the checkpoint. Git alone is insufficient: `.gitignore` excludes `data/` and `*.npy`, including saved training/evaluation embeddings. The research source also contains uncommitted changes; the current Git HEAD alone does not identify this implementation.

`handoff/research_snapshot.json` hashes project Python sources, configurations, documentation, dependency captures and every existing file under `results/` at handoff time. It includes historical interrupted runs as archived evidence, not successful experiments. Paths in this new snapshot are relative and can be verified after copying the preserved tree. Raw datasets, prepared `data/`, the Python environment and external model cache are excluded from this checksum inventory; stage-specific verifiers validate the raw/prepared inputs they need.

```powershell
python scripts/verify_research_handoff.py
.venv/Scripts/python.exe scripts/verify_research_handoff.py --environment
.venv/Scripts/python.exe -m pip check
```

The first command uses only the standard library and does not load models, execute training or rewrite results. The second additionally compares Python and all captured package versions. Its success does not establish an identical GPU driver, operating system, model cache or numerical execution environment. The adjacent snapshot checksum detects accidental changes; it is not a digital signature or an independently trusted authenticity record.

Do not run old reporters simply to check integrity. Reporters can rewrite plots, manifests and report bytes. In particular, the v7 protocol pins the exact v6 protocol and final experiment manifest hashes. The read-only handoff checker is the default archive check.

## 2. Recorded environment and numerical contract

The observed environment is Windows, Python 3.11.0, PyTorch/torchaudio 2.6.0+cu126, CUDA runtime 12.6, cuDNN 90501, NVIDIA GeForce RTX 4050 Laptop GPU, an AMD Ryzen 7 7435HS CPU and approximately 24 GB RAM. Exact installed distribution versions are in `requirements.research.lock.txt`; Python/platform and package capture are in `handoff/environment.json`. The v7 session JSON under `results/cost_confirmation_v7/sessions/` is the authoritative record of inference settings and loaded thread-pool libraries during that run.

The recorded model is `facebook/wav2vec2-base`, resolved revision `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8`, float32, SDPA attention, equal-length batches without padding, batch size 16, chunk size 500, mean pooling and L2 normalization. CUDA matmul TF32 is false; cuDNN TF32 is true; cuDNN benchmarking and deterministic algorithms are false. Timed v7 trials use one process, two CPU threads and a resident warmed model. The source pipeline fingerprint is `da6d02ce267eb32b0f5daa30adbb1ffaf2e6a867d1b1161ef0cbf3becef26026`.

```powershell
$env:HF_HUB_OFFLINE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
.venv/Scripts/python.exe -m pytest -q
```

Use the preserved cache and verify its resolved checkpoint before a rerun:

```powershell
.venv/Scripts/python.exe -c "from aura.encoders.wav2vec2 import Wav2Vec2Encoder; e=Wav2Vec2Encoder(batch_size=16); print(e.info); assert e.info.revision == '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8'; assert e.info.device == 'cuda'"
```

The current encoder resolves a model name through its cache and has no explicit revision argument. Do not assume a newly downloaded `main` has the recorded revision. Revision pinning as a public configuration field is future work; modifying the frozen encoder now invalidates historical source fingerprints. CPU fallback is supported by the implementation, but CPU measurements do not reproduce the GPU timing result. Same-layout v7 repeats were identical; v6 shows that changed batch composition can alter exemplar selections. Source/seed equality alone is not a numerical equivalence guarantee.

## 3. Candidate clean environment

In a separate copy, with Python 3.11 available:

```powershell
py -3.11 -m venv .venv-repro
.venv-repro/Scripts/python.exe -m pip install -r requirements.research.lock.txt
.venv-repro/Scripts/python.exe -m pip check
.venv-repro/Scripts/python.exe scripts/verify_research_handoff.py --environment
.venv-repro/Scripts/python.exe -m pytest -q
```

The lock capture includes the official CUDA 12.6 PyTorch wheel index and all observed package versions. Python must match 3.11.0 for the strict environment check; another 3.11 patch will be reported as different. Installation still requires compatible available wheels, network access and a suitable GPU driver. No wheels are bundled and no clean-environment solve/install has been validated. Preserve the original cache separately; installing dependencies does not download the model or datasets. A successful dependency check is a prerequisite, not evidence of reproduced scientific results.

## 4. Dataset and split provenance

| Dataset | Pinned source and attribution | Protocol |
|---|---|---|
| Speech Commands v0.02 | Existing source inventory and saved raw hashes; original attribution/license files must accompany any source copy | Ten configured commands; v4 reserves 250 fresh speakers, excludes earlier candidate recordings |
| AudioMNIST | `https://github.com/soerenab/AudioMNIST`, revision `630d7dab4c040882834de6fa21baf9a60372accd`; repository MIT license and citation preserved | v5: 45 candidate/15 held-out speakers; v6: 35/10 development split from those 45 candidates |
| FSDD | `https://github.com/Jakobovski/free-spoken-digit-dataset`, revision `26eb9aaf76e81b692f806f9140c2d2777410d7a1`; Jakobovski and contributors, README declares CC BY-SA 4.0 | v7: six speaker-disjoint folds; custom protocol differs from official index split |

For a separate clean data tree, use the pinned revisions rather than the repositories' current default branches:

```powershell
git init data/raw/AudioMNIST
git -C data/raw/AudioMNIST remote add origin https://github.com/soerenab/AudioMNIST.git
git -C data/raw/AudioMNIST fetch --depth 1 origin 630d7dab4c040882834de6fa21baf9a60372accd
git -C data/raw/AudioMNIST checkout --detach FETCH_HEAD
git init data/raw/FSDD
git -C data/raw/FSDD remote add origin https://github.com/Jakobovski/free-spoken-digit-dataset.git
git -C data/raw/FSDD fetch --depth 1 origin 26eb9aaf76e81b692f806f9140c2d2777410d7a1
git -C data/raw/FSDD checkout --detach FETCH_HEAD
```

These commands are for absent dataset checkouts, not an instruction to alter preserved ones. FSDD's detailed attribution is in `results/cost_confirmation_v7/DATASET_NOTICE.md`. AudioMNIST citation is in `PROJECT.md` and its source repository. Do not redistribute a dataset or derived artifact bundle without preserving its applicable attribution and license materials; the checksum snapshot itself contains no source audio.

## 5. Read-only scientific verification of the saved studies

After the matching dependencies and original raw/prepared data are present, in the original workspace layout:

```powershell
.venv/Scripts/python.exe scripts/verify_saved_independent_study.py --control
.venv/Scripts/python.exe scripts/verify_saved_independent_study.py
.venv/Scripts/python.exe -c "from pathlib import Path; from scripts.report_cost_confirmation import verify; verify(Path('results/cost_confirmation_v7')); print('FSDD model/prediction verification passed')"
```

These commands validate the duplicate control, independent AudioMNIST study and v7 checkpoint/prediction provenance without fitting models. The wrapper suppresses the legacy verifier's report writes and requires the regenerated contents to match the saved evidence. The v5 checks recompute metrics from saved predictions and verify checkpoint hashes; v7 additionally executes each saved classifier to reproduce predictions from saved evaluation embeddings. The last command calls the verifier function, not the reporter entry point. Full source/result byte integrity, including v6, is checked by the handoff checker. Byte integrity and semantic/model verification are different guarantees.

Legacy stage manifests contain absolute Windows paths. Their semantic verifiers and checkpoint-resume logic have not been made relocatable; moving the project can make those commands fail even when the relative handoff checksum passes. Preserve original locations for historical semantic verification, or implement a separately tested read-only path mapping without editing frozen manifests. The portable checksum checker does not solve legacy path portability.

## 6. Fresh v7 replication without overwriting the archive

V7 needs the FSDD checkout, recorded model cache and preserved v6 decision artifacts. It does not need to reopen the earlier AudioMNIST holdout. Use an unused output directory:

```powershell
.venv/Scripts/python.exe scripts/study_cost_confirmation.py --output-dir results/cost_confirmation_v7_replication
.venv/Scripts/python.exe scripts/report_cost_confirmation.py results/cost_confirmation_v7_replication
```

Run these sequentially with no tests, downloads or other heavy jobs competing for resources. The runner fits the predeclared 60 heads, checkpoints training arrays and models, locks all heads, then evaluates. The reporter checks all 3,000 raw hashes and saved predictions, and writes accuracy/timing/stability tables and figures. A completed output directory is rejected by the runner. Interrupted jobs can resume if declarations and hashes match, but multiple hardware/runtime sessions must remain visible; a pair crossing sessions is rejected. A replication is a repeat of a now-observed benchmark, not a fresh generalization test.

V2–v6 reconstruction commands and dependency order are documented in `PROJECT.md`. Their partition declarations and locked choices are part of the preserved evidence. Do not regenerate an upstream decision artifact in place: downstream studies pin its exact hash. Start a separate full tree for reconstruction and distinguish a newly generated study from the immutable original evidence. In particular, the amended v6 canonical-accuracy/component-timing design must not be silently replaced by a strict numerical-identity claim.

## 7. Acceptance and remaining limits

The local handoff acceptance checks are: exact source/result snapshot passes, installed packages match capture, `pip check` succeeds, regression tests pass, saved independent-study metrics/checkpoint hashes verify, and saved v7 classifiers reproduce their predictions. Record new results separately rather than replacing archived files.

The baseline's supported claims are in `RESEARCH_FINDINGS.md`. Fixed-budget data efficiency has positive evidence. The v7 fixed fraction misses the one-point accuracy goal despite faster warm training. Universal savings, annotation efficiency, algorithmic novelty, cross-device numerical identity, energy savings and production readiness are not established. A clean-machine run and a second hardware/session replication remain unvalidated; the handoff documents how to attempt them, not that they have happened.
