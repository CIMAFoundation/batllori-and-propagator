# Fire–and–Drought Model: Technical Walkthrough

This report documents how the coupled vegetation–fire simulator in this repository operates, starting from the orchestration in `main.py` and expanding to the supporting modules. Emphasis is placed on the scientific rationale behind each processing block so new contributors can reason about sensitivities, inputs, and outputs.

## 1. Entry Point (`main.py`)

### 1.1 Configuration and Setup
`main(config: SimulationConfig)` (`main.py:112`) is the only executable entry point. A deterministic NumPy generator is created from `SimulationConfig.seed`, allowing controlled stochastic sampling of winds, moisture, ignition points, and event durations. File-system side effects are constrained to `output/`, which is created up front.

### 1.2 Raster Ingestion
`load_rasters()` (`main.py:148`) pulls three rasters with rasterio:

- `dem.tif`: a digital elevation model used by the fire propagator for slope-aware spread.
- `clc_2018.tif`: categorical CORINE land-cover classes.
- `mask.tif`: study-area stencil; only `True` pixels participate in vegetation updates or ignition sampling.

The vegetation raster is masked (`np.where(mask, raw_veg, 0)`) to prevent boundary noise from contaminating downstream proportions.

### 1.3 Translating to Batllori State Space
The Batllori model requires per-cell vegetation mixtures across six classes. `veg_propagator_to_batllori()` (`main.py:57`) maps CORINE codes to six-dimensional proportion vectors, explicitly handling non-vegetated (`-3333`) and nodata (`-9999`) sentinels. Each cell is then perturbed with Gaussian noise (`apply_initial_noise`, `INITIAL_NOISE_STD = 0.05`) to avoid perfectly uniform states that could stall transitions in the deterministic parts of the Batllori equations.

### 1.4 Vegetation Model Initialization
`Batllori6CL` (`batllori_6cl.py`) encapsulates succession and post-fire recovery dynamics. After instantiation, `warm_up_model(..., steps=5)` lets the system relax without fire forcing so that the time-since-fire (`tsf`) counters accumulate and the logistic feedbacks (fuel build-up, recruitment rates) stabilize before coupling with the fire model.

### 1.5 Baseline Accounting
At the start of the main loop the code caches:

- `batllori_veg`: the current proportion tensor.
- `initial_proportions`: total area per Batllori class inside the mask (`compute_initial_proportions`).
- `proportions_history`: a `6 × timesteps` array, initialized to `NaN`, to store class trajectories normalized to the initial area.

## 2. Coupled Time Loop

For each simulated timestep (`main.py:128`):

1. **State Exchange:** The latest Batllori proportions are pulled and converted back into categorical fuel codes via `veg_batllori_to_propagator`. This supplies the fire simulator with per-cell fuel types matching its internal spread tables.
2. **Stochastic Ignitions:** `generate_fire_events()` draws event durations (`sample_event_durations` in `extract_probabilities.py`) and ignition coordinates (`extract_ignition_points`). Durations mimic the empirical distribution of maximum daily fire lengths, and ignitions are sampled proportional to a susceptibility raster (`susc_monti_pisani.tif`) raised to the 4th power to emphasize known hotspots.
3. **Fire Spread:** Each ignition is passed to `simulate_single_fire`. This function:
   - Constructs a fresh `Propagator` simulator seeded with the current DEM and categorical vegetation.
   - Samples wind speed (bounded below at 0), wind direction (uniform 0–360°), and fuel moisture (bounded at 0) from normal distributions defined in `SimulationConfig`.
   - Builds `BoundaryConditions` with the ignition point, meteorology, and moisture (`create_boundary_conditions` in `propagator_module.py`).
   - Runs `start_simulation` until either the requested duration (converted to seconds) elapses or the fire exits the domain, in which case `PropagatorOutOfBoundsError` is caught and logged.
   - Thresholds the resulting fire probability raster at `FIRE_SCAR_THRESHOLD = 0.3`, returning both the binary scar and the mean fireline intensity (`get_fire_scar`).
4. **Fire Aggregation:** `run_fire_events` stacks all scars and intensities for the timestep and takes per-cell maxima. This “any ignition burns the cell” rule is conservative and ensures overlapping fires do not dilute severity.
5. **Vegetation Response:** `batllori_model.step(fire_scars)` applies the scars as boolean masks, triggering either the unburned succession equations or the post-fire redistribution in `_step_kernel` (Numba-accelerated). Key scientific mechanisms implemented in `_step_kernel` include:
   - **Fuel-limited recruitment:** Transition rates from shrubs to young conifers (`K_u_sy`) and from young to mature cohorts depend on both local and landscape-mean mature cover via the `omega_cell`/`omega_l` blend.
   - **Succession constants:** `k_sy_sm`, `k_ry_rm`, and `k_au` encode background growth from grasses to shrubs and from young to mature cohorts.
   - **Fire mortality & redistribution:** Inside a fire mask, proportional removal of shrub/forest cohorts uses empirically derived weights (`w_*`). Time-since-fire (`tsf`) modulates survival probability through `power_val` and `frac_val`, embodying higher mortality shortly after previous fires.
   - **Resetting clocks:** Cells that burn have `tsf` reset to zero, influencing the next stochastic fire response.
6. **Diagnostics:**
   - `update_proportions_history` records the relative abundance (current / initial) for each Batllori class.
   - `save_vegetation_and_fire_map` overlays categorical vegetation with fire scars.
   - `save_proportions_over_time` plots temporal trajectories for all classes.
   - `save_batllori_heatmaps` writes per-class spatial heatmaps, masking out non-study areas with `np.nan`.

All Matplotlib figures are closed (`plt.close("all")`) at the end of `main` to avoid resource leaks when the loop is large.

## 3. Supporting Modules

### 3.1 `Batllori6CL`
Located in `batllori_6cl.py`, this class is parameterized by `ModelParams`, which mirrors literature values for growth rates (`k_*`), recruitment sensitivities (`rho_*`), and fire redistribution weights (`w_*`). The computational core `_step_kernel` is JIT-compiled with Numba to keep the per-cell updates tractable even for large grids. Invalid data sentinels (`-9999`, `-3333`, `-6666`) propagate through most operations so the simulation respects nodata regions and non-vegetated surfaces.

### 3.2 `propagator_module.py`
This module isolates the dependency on the external `propagator` package, simplifying mocking during tests. `get_simulator` configures the simulator to use the legacy fuel system and disables spotting. `start_simulation` handles the iterative stepping with graceful termination when the fire exits the computational domain or the duration limit is reached.

### 3.3 `extract_probabilities.py`
This script loads historical extreme-event data (`data/extreme_events.csv`) to produce two stochastic drivers:

- `sample_event_durations`: draws an annual count from the empirical distribution of daily maxima and samples durations with probability weighting that favors long-lasting events (scaled by `(duration / MAX_DURATION)^2`).
- `extract_ignition_points`: samples ignition coordinates within the study mask, using susceptibility weights raised to the 4th power to represent a non-linear increase in ignition likelihood for high-risk cells.

The module keeps a dedicated `np.random.Generator` seeded for reproducibility.

### 3.4 `data.py`
While not used directly by `main.py`, `data.py` provides utilities for exploring historical land-cover rasters (windowed slices, color maps). It is useful when validating the correctness of `veg_propagator_to_batllori` mappings or constructing lightweight fixtures for future tests.

## 4. Outputs and Diagnostics

Each timestep produces:

- `veg_mapXX_fire_scar.png`: categorical vegetation plus fire-scar contour.
- `batllori_proportions_timestep_XX.png`: six heatmaps (2×3 panel) for class proportions inside the mask.
- `veg_area_over_time.png`: cumulative figure updated every iteration with the latest normalized class areas.

Because plots are overwritten each timestep, the final state captures the full simulation history while lowering disk usage. For deeper analysis, the `proportions_history` array could be serialized (currently in-memory only).

## 5. Scientific Considerations

1. **Coupling Strategy:** Vegetation feedback into fire occurs through fuel typing (`veg_batllori_to_propagator`), while fire feedback into vegetation is a binary scar mask. Eventually, incorporating fire intensity (already returned by `run_fire_events`) could permit severity-dependent mortality.
2. **Stochastic Drivers:** Wind, moisture, event durations, and ignition points are stochastic but reproducible via `SimulationConfig.seed`. Sensitivity to these priors can be explored by varying `mean_*`/`std_*` parameters.
3. **Time Resolution:** Each loop iteration represents an abstract timestep, not necessarily a calendar year. The Batllori kernel increments `tsf` by one per step, so matching observed fire-return intervals requires calibrating event frequency and `config.timesteps`.
4. **Numerical Stability:** The noise injection and warm-up mitigate division-by-zero during normalization and prevent the kinetic terms from locking into symmetric equilibria.

## 6. Running and Extending

To reproduce the simulation:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py          # writes outputs/ diagnostics
```

Recommended extensions include:

- Writing `pytest`-based unit tests around `veg_propagator_to_batllori`, `veg_batllori_to_propagator`, and `sample_event_durations` using small synthetic rasters.
- Persisting `proportions_history` as NetCDF or GeoTIFF stacks for downstream statistical analysis.
- Adding CLI arguments to expose `SimulationConfig` parameters without editing source code.

This architecture keeps domain logic modular: `main.py` orchestrates data flow, `Batllori6CL` governs ecological succession, `propagator_module.py` encapsulates fire spread physics, and `extract_probabilities.py` provides data-driven ignition statistics. Together they implement a reproducible experiment for exploring cumulative fire–drought impacts on Mediterranean vegetation mosaics.
