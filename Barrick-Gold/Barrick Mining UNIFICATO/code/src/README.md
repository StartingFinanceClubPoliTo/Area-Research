# Source inventory

Application code lives in `barrick_unified/`. Repository-root entry points orchestrate these modules; licensed raw observations are not source assets.

| Module | Responsibility |
| --- | --- |
| `cli.py` | `status`, `test` and named `run` dispatch. |
| `project.py` | Project discovery, handoff audit and entry points. |
| `market_data.py` | Canonical schemas and validation. |
| `lse_adapter.py` | Historical provider-data adaptation. |
| `research_snapshot.py` | Dated snapshot assembly and provenance. |
| `empirical_figures.py` | Historical market/econometric figures. |
| `team8_option_audit.py` | Surface and calibration audit. |
| `team4_operating_figures.py` | Operating actuals/forecast figures. |
| `valuation.py` | Common operating, DCF, WACC and equity-proxy contract. |
| `valuation_reporting.py` | Single-configuration reporting. |
| `multimodel_valuation.py` | Four-engine comparison with identical non-gold inputs. |
| `multimodel_reporting.py` | Tables, figures and byte-hashed manifests. |
| `__init__.py` | Package boundary. |

`refactored/` separates `application/`, `domain/`, `gold/`, `operations/`, `simulation/`, `valuation/` and `reporting/`. Its gold adapters are checked against the legacy engine using deterministic array-equality tests. Historical and September snapshots remain separate; production guards reject conflicting cached engines.

Editable-install `*.egg-info/` is local build metadata, not authored source. See the [root README](../README.md) for setup and outputs.
