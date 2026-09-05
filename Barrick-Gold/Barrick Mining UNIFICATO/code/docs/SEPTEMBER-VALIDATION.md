# September validation boundaries

The authoritative run remains `20260904T130000Z-team8-refresh-v4`, using the 2 September GLD/CC64 snapshot. An editorial rebuild does not acquire data or recalibrate models.

- Heston leads current in-sample fit; Full Bates-Hawkes leads equal-date dense OOS among the models, not against persistence.
- Model comparisons use 4,667 observations; persistence comparisons use 4,026 within the origin interpolation support.
- The Treasury input is a continuously compounded par-yield proxy, not a bootstrapped zero curve.
- GLD/Q-to-gold/corporate transfer remains conditional. Persistent costs are prudential relative to otherwise identical efficiency-improvement cases, not a mathematical lower bound.

## Offline checks

The complete published companion suite passed **71 tests** on 5 September 2026 (`python -m pytest -q -p no:cacheprovider`, `PYTHONPATH=src`, bytecode generation disabled). No test was excluded in the public checkout. Three pre-existing Python cache directories were removed before this clean-handoff check.

The refresh checks contract/rankings, positive finite deterministic distinct paths, Hawkes stationarity and end-to-end reporting with recomputed input/code/artifact hashes. Numerical parity of the refactor is tested independently of labels.

On 5 September the applicable combined suite passed 83 tests. One local-only experimental file, `test_cli_data_provider.py`, was excluded because it requests an unimplemented `data` selector outside the published CLI. This is not a claim that unrestricted local pytest passes. That file is preserved, not silently deleted or marked as passing.

Three frozen references affected by Windows newline conversion were restored from verified archives using their originally declared byte hashes, after confirming equality apart from CRLF/LF. Historical configuration hashes were not rewritten. Git attributes preserve original bytes under `parity/sources/` and `data/manifests/`.

New manifests retain raw SHA256; a narrowly scoped exact LF match for externally supplied UTF-8 Python references is explicitly recorded. Whitespace, encoding and content changes still fail closed. Immutable run manifests are not rewritten to describe later code.

The legacy thesis generator is not the September editorial source of truth. Its local guard blocks workspaces containing the September refresh plan. Compile the maintained Overleaf source directly; historical reconstruction belongs in a separate archived workspace.
