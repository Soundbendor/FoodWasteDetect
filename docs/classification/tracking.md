# W&B monitoring and progress

Read this document when changing W&B initialization, artifacts, final summaries,
or progress reporting. [`tracking.py`](../../src/classification/tracking.py) owns run
lifetime through `tracked_run()`, used by both the CLI stages and the dimension sweep,
and `Tracker` publishes through the run it opens. Experiment and description loops own
their progress bars.

## Run lifetime

`tracked_run(config, stage, execute)` enters W&B's native `wandb.init(...)` context with the initial
experiment settings and stage as `job_type`.
Configuration publication, `Experiment` construction, and stage execution all
occur inside it; it returns the result of `execute(config, tracker)` and the run ID. Native context exit handles success, exceptions, and interrupts,
including cleanup failures. There is no application wrapper to preserve an earlier
exception when W&B cleanup itself raises, and no manual `finish()` call.

Authentication, destination, and mode use native `WANDB_*` settings. Automatic Git,
code, system metadata, and dependency-list capture remain disabled through W&B
settings. Do not add separate provenance capture to the whole-image runner.

`experiments.matroyshka.run_sweep()` opens one native context per dimension, with no
outer run. Every run in a sweep shares the W&B `group` `matroyshka_sweep_<UTC>`, the
name of its local summary folder, and carries its own `d<dimension>` tag. CLI stages
pass no group or tags. Resume is unsupported. Direct Python `Experiment(config)` calls are
untracked by default; their progress bars still work.

## Published data

| Publisher | Output and completion boundary |
| --- | --- |
| `Tracker.configuration(config: ExperimentConfig)` | `configuration-<run.id>` artifact of type `configuration`, containing `resolved_config.yaml` with settings captured before stage execution. |
| `Tracker.reference_counts(counts)` | `reference_source_counts` in the run summary. A completed index build publishes counts after its local summary write; evaluation supplies matching summary counts or null. |
| `Tracker.evaluation(report, report_path)` | Every aggregate metric under `evaluation/<key>`, a per-class W&B table, and an `evaluation-<run.id>` artifact of type `evaluation` containing `report.json`. Called only after a complete evaluation and successful local report write. |

The per-class table columns are `class`, `support`, `correct`, `recall`, `precision`,
and `f1`. Metric definitions belong in [retrieval](retrieval.md); local artifact
formats and reference-summary matching belong in [experiments](experiments.md).

## Progress ownership

Context-managed tqdm bars remain enabled without a TTY and use `mininterval=5`.
They close on errors or interruption and do not count unfinished work as complete.

| Loop | Total and advancement |
| --- | --- |
| Descriptions | Known class-count × descriptions-per-class total; advance after each returned request. |
| Text references | Known pending-record total; advance after a successful append. |
| Image references and validation embeddings | Running image count without an extra total scan or percentage/ETA; advance after a successful append. |
| Evaluation | Full validation count; advance after all predictions in a batch succeed. |

Resumed insertion bars count only newly processed work. Empty insertion loops can
show zero work. Disabled or cached description generation creates no description
loop. Persistence and counts advance at completed batches.

W&B's native console capture records tqdm's stderr output; disabling capture removes
that visibility, and offline runs retain output locally.

## Verification

[`tests/test_tracking.py`](../../tests/test_tracking.py) covers exact resolved
configuration, native context exit, publication ordering, failure boundaries,
progress counts, and isolated native offline runs. Description-loop checks also
use [`tests/test_descriptor_lifecycle.py`](../../tests/test_descriptor_lifecycle.py).
