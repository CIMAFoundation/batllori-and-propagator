# Repository Guidelines

## Project Structure & Module Organization
Core simulation code sits in the repo root: `model.py` drives the timestep loop and plotting, `propagator_module.py` wraps the external `propagator` library, and `data.py` handles raster ingestion plus colour maps. Input rasters (`data/clc_*.tif`) and any derived subsets stay under `data/`; generated figures belong in `images/`. Keep experimental notebooks or scripts in a dedicated `notebooks/` (create it as needed) so the root stays focused on runnable modules.

## Build, Test, and Development Commands
Create an isolated Python 3.13 environment, install dependencies, then execute the simulation script:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # includes propagator from Git
python model.py                  # runs the full fire-drought iteration and plots outputs
```
When iterating quickly, prefer `uv pip sync` if you already use `uv` elsewhere; it respects `pyproject.toml`.

## Coding Style & Naming Conventions
Follow standard PEP 8: four-space indentation, `snake_case` for functions and variables, and `UPPER_CASE` for constants (see `G_INDEX`, `TSF`). Leverage NumPy vector operations before writing nested Python loops unless clarity suffers. Document non-obvious domain formulas with short comments, and keep plotting code separate from data processing blocks for readability.

## Testing Guidelines
No automated suite is present yet. Add unit tests under a new `tests/` package using `pytest`, targeting pure functions in `data.py` and any helpers you introduce. Use lightweight raster fixtures (e.g., 10x10 arrays) rather than full GeoTIFFs to keep tests fast. Run tests with `pytest -q`; aim for coverage on new logic and regression checks for fire probability calculations.

## Commit & Pull Request Guidelines
Commits should stay short, present tense, and imperative (`add data loading`, `update propagator dependency` are good references). Each PR should: summarise the change, link any related issues, note expected impacts on runtime, and attach updated plots if visuals change. Confirm in the PR description that `python model.py` and the relevant `pytest` targets complete successfully before requesting review.

## Data & Dependency Notes
Raster assets are large; avoid re-committing them and prefer `.gitignore` for local experiments. The `propagator` dependency is pulled from GitHub—pin to commits when upgrading and call out the hash in PRs so reviewers can recreate environments deterministically.
