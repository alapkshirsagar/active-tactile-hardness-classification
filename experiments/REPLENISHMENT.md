# ConvLSTM active-sampling replenishment

Launch recipe: `sky/replenish-convlstm.yaml`. This is prepared but has not been
submitted. It uses the existing tested Docker image plus file mounts for the
new runner and instrumentation, so no image rebuild is needed.

## Scope and provenance

The W&B audit found 122 converged original trajectories out of 150 attempts in
`patchouli_lab/Final_exp_hardness_ConvLSTM`. The target is **10 successful runs
per comparison class and strategy**, i.e. 50 per strategy and 150 total successes.
The additional-success quotas are:

| Strategy | hardness1 | hardness2 | hardness3 | hardness4 | hardness5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Variance | 1 | 3 | 2 | 1 | 2 | 9 |
| Entropy | 2 | 3 | 3 | 1 | 3 | 12 |
| Random | 1 | 2 | 2 | 0 | 2 | 7 |

`convlstm_replenishment.json` freezes these audited counts, original failed IDs,
settings and seed ranges. Smoke runs do not count. New runs use project
`patchouli_lab/tactile-hardness-replenishment` and are linked to the original
campaign in their config. Keep original and new failed attempts in the analysis;
150 successes requires at least 178 attempts across both cohorts.

The Linux container uses TF 2.10.1/Keras 2.10.0. An original run's saved packages
instead reported TF/Keras 2.9.0. New runs are explicitly marked as a reconstructed
environment cohort; do not claim an exact historical environment replay.
Historical train/query sample-selection behavior is preserved, including the
unresolved possibility of overlapping source videos; this is not a corrected
holdout protocol. Data checksums and source hashes are recorded for this cohort,
but they do not prove identity with the original dataset ID 42.

## Launch

From the training repository root, with RunPod access configured and the existing
`high_scarlet_vole` volume registered, load Docker Hub and W&B credentials into
`SKYPILOT_DOCKER_PASSWORD` and `WANDB_API_KEY`, then run:

```sh
sky launch -c tactile-replenish sky/replenish-convlstm.yaml \
  --env SKYPILOT_DOCKER_PASSWORD \
  --secret WANDB_API_KEY \
  --down -i 5
```

Default GPU: one RTX2000-Ada in RO/EU-RO-1 for the cost/throughput trial.
The earlier single-process smoke test used an RTX4090; concurrent training on
the 16 GB RTX2000-Ada has not yet been validated. Override with `--gpus` if
required. There are no explicit CPU/RAM constraints.
Default concurrency is **two** independent child processes per GPU. Each uses
two TensorFlow intra-op/OpenMP threads, one inter-op thread, and one BLAS thread.
These limits leave room for data loading on the small GPU instances. Per-run batch
size and training settings remain unchanged. Check concurrent throughput and peak
host/GPU memory before increasing concurrency further. Override with
`--env REPLENISH_WORKERS=1` for a sequential baseline. Thread settings can be
adjusted using environment variables; record changes for performance comparisons.

For the first two-attempt trial, use a fresh cluster name and cap scheduling:

```sh
sky launch -c tactile-2000-trial sky/replenish-convlstm.yaml \
  --env SKYPILOT_DOCKER_PASSWORD --secret WANDB_API_KEY \
  --env REPLENISH_MAX_ATTEMPTS=2 --down -i 5
```

Both attempts use the real campaign settings. Verified successes count toward
the quotas. At the trial cap the controller reports "Reached attempt cap before
quotas were filled" and the SkyPilot job is marked FAILED; that is the expected
trial stop, provided neither attempt has an infrastructure error. Outputs and
GPU telemetry remain on the network volume. To continue after reviewing memory
and throughput, launch without the cap override; completed attempts are resumed
without repetition and the normal total cap is 84.

The existing image tag is reused. `sky launch` syncs patched training files and
scripts; `sky exec` does not update file mounts. Use a fresh cluster name or clean
up previous launch requests when changing GPU types. No retry flag is included
by default: capacity errors should be inspected rather than retried indefinitely.
The network volume restricts compute to EU-RO-1.

## Counting and stopping

Each attempt runs in its own directory and process, with independent Python,
NumPy and TensorFlow seeds set before training. Seed ranges are fixed in the
plan and a new seed is used after non-convergence; original seeds are unavailable.
The Own-data sampler used `random.seed(None)` internally. Only when explicit
run seeding is enabled, it now derives those reseeds deterministically. This
preserves random sampling rather than resetting to OS entropy. Bitwise GPU
reproducibility is not guaranteed.

A success requires:
- Process exit 0 and the historical initial-loss convergence gate passing.
- Five completed sampling updates and 350 logged training epochs.
- Six accuracy/MAE points, five entropy/variance points, all finite.
- W&B run finished, with the same history counts verified through its API.

Convergence failures are logged and replaced with another attempt in the same
group. Infrastructure, authentication, upload verification, incomplete histories,
or unexpected process failures stop the controller. The controller does not
select runs based on their final accuracy. Groups already at quota are skipped,
and a group has at most one active attempt, preventing quota overshoot even
with two workers. No model checkpoints are retained.

The campaign stops at **84 new attempts** if quotas remain unfilled. Increase
that cap only after reviewing the outcomes. It locks its persistent directory
against a second controller, freezes the plan and provenance, and resumes
completed attempts without repeating them. Interrupted attempts without complete
process/W&B evidence require reconciliation before restarting; they are not
silently counted or discarded.

## Outputs and validation

All outputs persist under:
`/workspace/runs/convlstm-replenishment-20261022-v1/`.

- `progress.json`: original successes, new successes and remaining quotas.
- `plan.json`, `provenance.json`: settings, seeds, data/source/lock hashes.
- `controller.log`, `gpu.csv`: controller events and GPU telemetry.
- `attempt-*/`: command, seed, W&B ID, stdout, W&B logs, exit status and result.

```sh
sky logs tactile-replenish
```

SkyPilot autodown is requested after five idle minutes; setup/provisioning errors
may require manual teardown. Do not delete the independent network volume.

CPU-only checks:

```sh
python3 -m unittest discover -s tests -v
python3 scripts/replenish_convlstm.py --dry-run
```

The tests cover audited quotas, group isolation, incomplete-history rejection,
W&B-finish persistence, seeded sample repeatability, retry after non-convergence,
and resuming a completed quota without launching duplicates. The full new
launcher still requires an online training run; local tests do not establish
historical equivalence or concurrent speedup.

## Concurrent I/O checks

Two processes may read the preprocessed `.npy` inputs simultaneously; each
`np.load` returns process-local arrays. Replenishment sets
`TACTILE_READONLY_DATA=1`: if either cached array is absent, the Own-data loader
raises an error instead of regenerating files through the shared dataset link.
The campaign must not be used while another program edits these input arrays.

Each attempt has independent working, log, result, temporary, CUDA, Keras,
Matplotlib and W&B cache/config/data paths. Python bytecode writes are disabled
in children. Checkpoint writes are disabled. The SDK compatibility patch is
applied once before the child processes start. The controller is the sole writer
to quota/progress state and holds a campaign lock.

The CPU-only concurrency test starts two real OS processes against the real
cached-data loader. Both repeatedly read a common pair of read-only arrays and
mutate their private copies; input hashes and modification times remain unchanged.
Missing and partial cache pairs are tested to fail without writes. A separate
two-worker scheduler test checks distinct paths/run IDs/seeds and no quota
overshoot. All 11 checks pass. These tests do not replace measuring host RAM,
VRAM and throughput with two real training processes on the worker.
