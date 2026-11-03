# W&B monitoring and progress

Read this document when changing W&B initialization, artifacts, final summaries,
or progress reporting. [`cli.py`](../../src/classification/cli.py) owns run lifetime;
[`tracking.py`](../../src/classification/tracking.py) only publishes through a supplied
run. Experiment and description loops own their progress bars.

## Run lifetime

After parsing and shared validation, `_tracked_run()` enters W&B's native
`wandb.init(...)` context with the resolved dictionary and stage as `job_type`.
Configuration publication, `Experiment` construction, and stage execution all
occur inside it. Native context exit handles success, exceptions, and interrupts,
including cleanup failures. There is no application wrapper to preserve an earlier
exception when W&B cleanup itself raises, and no manual `finish()` call.

Authentication, destination, and mode use native `WANDB_*` settings. Automatic Git,
code, system metadata, and dependency-list capture remain disabled through W&B
settings. Do not add separate provenance capture to the whole-image runner.

The dimension sweep has one native context per dimension, with no outer run or
explicit grouping/resume settings. Direct Python `Experiment(config)` calls are
untracked by default; their progress bars still work.

## Published data

| Publisher | Output and completion boundary |
| --- | --- |
| `Tracker.configuration(config)` | `configuration-<run.id>` artifact of type `configuration`, containing only `resolved_config.yaml`. SDK-managed `artifact.new_file(..., encoding="utf-8")` writes the effective dictionary. Source YAML, invocation metadata, and application-managed temporary files are excluded. |
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
loop. Progress callbacks, reused offsets, `Tracker.progress`, and intermediate W&B
progress metrics are absent. Persistence and counts advance at completed batches.

W&B's native console capture records tqdm's stderr output; disabling capture removes
that visibility, and offline runs retain output locally. Online Logs-tab rendering
has not been tested.

## Verification

[`tests/test_tracking.py`](../../tests/test_tracking.py) covers exact resolved
configuration, native context exit, publication ordering, failure boundaries,
progress counts, and isolated native offline runs. Description-loop checks also
use [`tests/test_descriptor_lifecycle.py`](../../tests/test_descriptor_lifecycle.py).
See [verification history](../experiments.md) for dated evidence and limitations.
