# Candidate-pool numerical control

This control ran before any v6 development predictions. It used four recordings from the former AudioMNIST training candidate pool, the frozen preprocessing and checkpoint, CUDA, and the same encoder implementation.

For each recording, compare a single-clip call with a call containing sixteen copies of that exact processed waveform. All batch members have identical length; no padding occurs. A third call repeats the single-clip invocation.

| Recording | Maximum coordinate difference, B1 vs B16 | Cosine, B1 vs B16 | Maximum coordinate difference, repeated B1 |
|---|---:|---:|---:|
| audiomnist/0_01_0 | 0.000207599 | 0.999998868 | 0 |
| audiomnist/0_01_1 | 0.000018293 | 0.999999940 | 0 |
| audiomnist/2_12_0 | 0.000037941 | 0.999999940 | 0 |
| audiomnist/4_23_0 | 0.000177413 | 0.999999344 | 0 |

Observed flags: CUDA matmul TF32 disabled; cuDNN TF32 enabled. No precision flags were changed for the cost experiment. This control does not isolate the cause to a specific kernel or TF32 operation.

[PyTorch 2.6 numerical-accuracy documentation](https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html) explains that mathematically equivalent batched/sliced computations need not produce bitwise-identical floating-point outputs. The earlier padding defect is separate: this control has neither padding nor unequal waveform lengths.

A complete first timed pass stopped on its strict per-coordinate comparison (maximum difference 0.01416147) before any models or development scores were generated. Its protocol and original runner are preserved in `results/acquisition_cost_v6_strict_check/`.

The amended protocol records full-pool/selected-pass drift and guards shapes, IDs, finiteness, unit norms, and minimum corresponding-vector cosine >= 0.99. This threshold is an integrity audit, not a claim of numerical identity. Fixed canonical embeddings supply the main accuracy comparison; fresh-vector shadow fits and reselection provide a separate seed-zero sensitivity diagnostic after all timing is complete.
