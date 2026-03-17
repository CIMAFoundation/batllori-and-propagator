from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec

from batllori_6cl import Batllori6CL
from propagator_module import (
    create_boundary_conditions,
    get_fire_scar,
    get_simulator,
    start_simulation,
)

###############################################################################
# SETTINGS AND CONFIGURATION
###############################################################################

# number of years to simulate in a single realization
TIMESTEPS = 100

# >>> Batllori model parameters
BATLLORI_CLASSES = 6
INITIAL_NOISE_STD = 0.05  # initial noise injected into vegetation proportions
WARMUP_STEPS = 5  # number of steps to run the Batllori model before starting the simulation

# >>> Fire event generation parameters

# the number of fire events per year is a normal distribution
MEAN_NUMBER_EVENTS_PER_YEAR = 10
STD_NUMBER_EVENTS_PER_YEAR = 5

# probability of an extreme event among the fire events
PROB_EXTREME_EVENT = 0.05

# the weather conditions for the extreme event
EXTREME_EVENT_WIND_SPEED = 40.0
EXTREME_EVENT_WIND_DIRECTION = 30.0
EXTREME_EVENT_FUEL_MOISTURE = 5.0
EXTREME_TIME_LIMIT = 28800  # seconds (8 hours)

# the weather conditions for the normal events
NORMAL_EVENT_WIND_SPEED = 5.0
NORMAL_EVENT_FUEL_MOISTURE = 15.0
NORMAL_TIME_LIMIT = 3600  # seconds (1 hour)

N_FIRE_REALIZATIONS = 5  # number of stochastic realizations to run for each fire event
FIRE_SCAR_THRESHOLD = 0.3  # probability threshold to consider a cell as burned in the fire scar map

# >>> general settings
# SEED = 42
SEED = None
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/normal")
DEM_PATH = DATA_DIR / "dem.tif"  # Digital Elevation Model raster path [m]
VEG_PATH = DATA_DIR / "clc_2018.tif"  # Land-cover raster path with PROPAGATOR classes

# SUSCEPTIBILITY_PATH = DATA_DIR / "susc_monti_pisani.tif"  # raster path with fire susceptibility values, which is between 0 and 1 (high susceptibility) [OPTIONAL]
SUSCEPTIBILITY_PATH = None  # if no susceptibility provided, ignitions will be sampled uniformly in the masked area

MASK_PATH = DATA_DIR / "mask.tif"  # mask to define the area of interest (1 for valid cells, 0 for excluded cells) [OPTIONAL]
# MASK_PATH = None  # if no mask provided, consider all cells as valid

# >>> plot settings
BATLLORI_LABELS = [
    "Grassland (A)",
    "Shrubs (U)",
    "Conifers - young (Sy)",
    "Conifers - mature (Sm)",
    "Broadleaves - young (Ry)",
    "Broadleaves - mature (Rm)",
]

PROPAGATOR_CLASS_LABELS = {
    0: "Nodata",
    1: "Broadleaves",
    2: "Shrubs",
    3: "Bare/Non-vegetated",
    4: "Grasslands",
    5: "Conifers",
    # 6: "Croplands and agro-forestry areas",  # mapped to "Bare/Non-vegetated" in the simulation
    # 7: "Not fire-prone forest"  # mapped to "Broadleaves" in the simulation
}
PROPAGATOR_CLASS_COLORS = [
    "#d0d0d0",  # nodata / fallback
    "#1b7837",  # broadleaves
    "#b35806",  # shrubs
    "#f7f7f7",  # bare/non-vegetated
    "#a6d96a",  # grasslands
    "#00441b",  # conifers
]
PROPAGATOR_BOUNDS = np.arange(len(PROPAGATOR_CLASS_LABELS) + 1) - 0.5
PROPAGATOR_CMAP = ListedColormap(PROPAGATOR_CLASS_COLORS)
PROPAGATOR_CMAP.set_bad("#f0f0f0")
PROPAGATOR_NORM = BoundaryNorm(PROPAGATOR_BOUNDS, PROPAGATOR_CMAP.N)


###############################################################################
# HELPERS
###############################################################################

def load_rasters(mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load DEM, vegetation, susceptibility and mask rasters."""
    with rio.open(DEM_PATH) as dem_src:
        dem = dem_src.read(1).astype("int16")
    with rio.open(VEG_PATH) as veg_src:
        veg = veg_src.read(1).astype("int8")
    if SUSCEPTIBILITY_PATH is not None:
        with rio.open(SUSCEPTIBILITY_PATH) as susc_src:
            susceptibility = susc_src.read(1).astype("float32")
    else:
        susceptibility = np.ones(dem.shape, dtype="float32")  # if no susceptibility provided, use uniform susceptibility
    if mask is None:
        if MASK_PATH is not None:
            with rio.open(MASK_PATH) as mask_src:
                mask = mask_src.read(1) > 0  # boolean mask
        else:
            # if no mask provided, consider all cells as valid
            mask = np.ones(dem.shape, dtype=bool)
    # add check that all rasters are aligned
    if not (dem.shape == veg.shape == susceptibility.shape == mask.shape):
        raise ValueError("Input rasters have different shapes, please check the input files.")
    return dem, veg, susceptibility, mask


def apply_initial_noise(initial_map: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject small perturbations on vegetation proportions and renormalize proportion vectors."""
    noise = rng.normal(0, INITIAL_NOISE_STD, initial_map.shape)
    perturbed = np.where(initial_map > 0, initial_map + noise, initial_map)
    sums = perturbed.sum(axis=2, keepdims=True)
    return np.where(initial_map > 0, perturbed / sums, initial_map)


def warm_up_model(model: Batllori6CL, steps: int) -> None:
    for _ in range(steps):
        model.step()


def compute_initial_proportions(batllori_veg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    initial_proportions = np.zeros(BATLLORI_CLASSES)
    for batllori_class in range(BATLLORI_CLASSES):
        batllori_slice = batllori_veg[:, :, batllori_class]
        batllori_class_sum = np.where(mask & (batllori_slice >= 0), batllori_slice, 0).sum()
        initial_proportions[batllori_class] = batllori_class_sum
    return initial_proportions


###############################################################################
# BATLLORI-PROPAGATOR VEGETATION MAPPING
###############################################################################


def veg_propagator_to_batllori(land_cover: np.ndarray) -> np.ndarray:
    """Translate land-cover codes into vegetation proportion vectors."""
    grid_size = land_cover.shape[0]
    land_cover = land_cover.copy()
    initial_map = np.zeros((grid_size, grid_size, BATLLORI_CLASSES), dtype=float)

    # mapping rules PROPAGATOR -> Batllori

    # some classes of PROPAGATOR are removed
    land_cover[land_cover == 6] = 3  # "croplands" mappet to "bare/non-vegetated"
    land_cover[land_cover == 7] = 1  # "not fire-prone forest" in "broadleaves"
    vector_map = {
        1: np.array([0, 0, 0, 0, 0.1, 0.9]),  # broadleaves -> Ry, Rm
        2: np.array([0, 1, 0, 0, 0, 0]),  # shrubs -> U
        4: np.array([1, 0, 0, 0, 0, 0]),  # grasslands -> A
        5: np.array([0, 0, 0.2, 0.8, 0, 0]),  # conifers -> Sy, Sm
        0: np.full(BATLLORI_CLASSES, -9999.0),  # nodata
        3: np.full(BATLLORI_CLASSES, -3333.0),  # bare/Non-vegetated
        -3333: np.full(BATLLORI_CLASSES, -3333.0),
        -9999: np.full(BATLLORI_CLASSES, -9999.0),
    }

    for i in range(grid_size):
        for j in range(grid_size):
            code = land_cover[i, j]
            if code not in vector_map:
                raise ValueError(f"Unexpected land-cover value {code} at position ({i}, {j})")
            initial_map[i, j] = vector_map[code]

    return initial_map


def veg_batllori_to_propagator(veg: np.ndarray) -> np.ndarray:
    """Translate vegetation proportion vectors into land-cover codes."""
    grid_size = veg.shape[0]
    land_cover = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for i in range(grid_size):
        for j in range(grid_size):
            proportions = veg[i, j]
            if np.all(proportions == 0):
                land_cover[i, j] = 3  # Non-vegetated areas
            
            # Mapping rules Batllori -> PROPAGATOR
            # conifers if Sy+Sm > 0.3
            # shrubs if U > 0.3
            # broadleaves if Ry+Rm > 0.7
            # grasslands otherwise

            sum_conifers = proportions[2] + proportions[3]
            sum_broadleaves = proportions[4] + proportions[5]
            sum_shrubs = proportions[1]
            if sum_conifers > 0.3:
                land_cover[i, j] = 5  # conifers
            elif sum_shrubs > 0.3:
                land_cover[i, j] = 2  # shrubs
            elif sum_broadleaves > 0.7:
                land_cover[i, j] = 1  # broadleaves
            else:
                land_cover[i, j] = 4  # grasslands
    return land_cover


###############################################################################
# FIRE SIMULATION
###############################################################################

@dataclass(frozen=True)
class FireEvent:
    """Data class to represent a fire event with its parameters."""
    coord: tuple[int, int]  # (row, col) coordinates of the ignition point
    wind_dir: float  # wind direction (from which wind comes) in degrees (0-360, where 0 is from north, 90 is from east, etc.)
    wind_speed: float  # wind speed in km/h
    fuel_moisture: float  # fuel moisture content in percentage (0-100)
    time_limit: int  # maximum simulation time of the event in seconds
    is_extreme: bool = False  # flag to indicate if the event is extreme


def extract_ignition_points(
    n_events: int,
    rng: np.random.Generator,
    mask: np.ndarray,
    susceptibility: np.ndarray,
) -> list[tuple[int, int]]:
    """Sample ignition coordinates in the masked area, eventually with probability coming from susceptiblity."""
    rng = rng or np.random.default_rng()

    ignition_points = []
    for _ in range(n_events):
        # Get valid indices in the masked area
        valid_indices = np.where(mask)
        if len(valid_indices[0]) == 0:
            continue
        
        # Extract susceptibility values for valid cells
        valid_susceptibility = susceptibility[valid_indices]
        
        # Normalize susceptibility to create probability distribution
        susceptibility_sum = valid_susceptibility.sum()
        if susceptibility_sum > 0:
            probabilities = valid_susceptibility / susceptibility_sum
        else:
            probabilities = np.ones_like(valid_susceptibility) / len(valid_susceptibility)
        
        # Sample an index based on susceptibility probabilities
        sampled_idx = rng.choice(len(valid_indices[0]), p=probabilities)
        row = int(valid_indices[0][sampled_idx])
        col = int(valid_indices[1][sampled_idx])
        ignition_points.append((row, col))

    return ignition_points


def generate_fire_events(
    rng: np.random.Generator,
    mask: np.ndarray,
    susceptibility: np.ndarray
) -> list[FireEvent]:
    """Generate a list of fire events for the current timestep."""
    # sample number of events
    n_events = int(rng.normal(MEAN_NUMBER_EVENTS_PER_YEAR, STD_NUMBER_EVENTS_PER_YEAR))
    if n_events < 0:
        n_events = 0
    # extract ignition points
    ignition_coords = extract_ignition_points(
        n_events, rng=rng, mask=mask, susceptibility=susceptibility
    )
    # define which events are extreme
    extreme_events_flags = rng.uniform(0, 1, n_events) < PROB_EXTREME_EVENT
    # assign weather conditions and time limits based on the event type and create FireEvent instances
    events: list[FireEvent] = []
    for is_extreme, coord in zip(extreme_events_flags, ignition_coords):
        if is_extreme:
            wind_speed = EXTREME_EVENT_WIND_SPEED
            wind_direction = EXTREME_EVENT_WIND_DIRECTION
            fuel_moisture = EXTREME_EVENT_FUEL_MOISTURE
            time_limit = EXTREME_TIME_LIMIT  # seconds
        else:
            wind_speed = rng.normal(NORMAL_EVENT_WIND_SPEED, 2.0)
            wind_direction = float(rng.uniform(0, 360))
            fuel_moisture = rng.normal(NORMAL_EVENT_FUEL_MOISTURE, 2.0)
            time_limit = NORMAL_TIME_LIMIT  # seconds
        # add the fire event to the list
        events.append(
            FireEvent(
                coord=coord,
                wind_speed=wind_speed,
                wind_dir=wind_direction,
                fuel_moisture=fuel_moisture,
                time_limit=time_limit,
                is_extreme=is_extreme,
            )
        )

    return events


def run_fire_events(
    events: Iterable[FireEvent],
    dem: np.ndarray,
    veg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the fire simulation for a list of fire events and return the combined fire scar map and intensity map."""
    fire_scars_list = []
    fire_intensities_list = []

    for event in events:
        print(f'Simulating {event}')
        fire_scar, intensity = simulate_single_fire(dem, veg, event)
        fire_scars_list.append(fire_scar)
        fire_intensities_list.append(intensity)

    if not fire_scars_list:
        shape = veg.shape
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.float32)

    fire_scars = np.max(np.stack(fire_scars_list), axis=0)
    fire_intensities = np.max(np.stack(fire_intensities_list), axis=0)
    return fire_scars, fire_intensities


def simulate_single_fire(
    dem: np.ndarray,
    veg: np.ndarray,
    event: FireEvent,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the external propagator for a single ignition event."""
    simulator = get_simulator(dem, veg, realizations=N_FIRE_REALIZATIONS)
    wind_speed = event.wind_speed
    wind_direction = event.wind_dir
    fuel_moisture = event.fuel_moisture
    time_limit = event.time_limit
    boundary_conditions = create_boundary_conditions(
        wind_speed,
        wind_direction,
        fuel_moisture,
        event.coord,
    )
    start_simulation(simulator, boundary_conditions, time_limit)
    return get_fire_scar(simulator, threshold=FIRE_SCAR_THRESHOLD)


###############################################################################
# OUTPUT AND VISUALIZATION
###############################################################################

def update_proportions_history(
    batllori_veg: np.ndarray,
    mask: np.ndarray,
    initial_proportions: np.ndarray,
    history: np.ndarray,
    timestep: int,
) -> None:
    for batllori_class in range(BATLLORI_CLASSES):
        batllori_slice = batllori_veg[:, :, batllori_class]
        batllori_class_sum = np.where(mask & (batllori_slice >= 0), batllori_slice, 0).sum()
        baseline = initial_proportions[batllori_class]
        if baseline > 0:
            ratio = batllori_class_sum / baseline
            history[batllori_class, timestep] = ratio
        else:
            history[batllori_class, timestep] = np.nan


def save_vegetation_and_fire_map(
    batllori_veg: np.ndarray,
    fire_scars: np.ndarray,
    mask: np.ndarray,
    timestep: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    propagator_map = veg_batllori_to_propagator(batllori_veg)
    masked_map = np.where(mask, propagator_map, np.nan)
    im = ax.imshow(masked_map, cmap=PROPAGATOR_CMAP, norm=PROPAGATOR_NORM)
    masked_fire = np.where(mask, fire_scars, np.nan)
    ax.contour(np.ma.masked_invalid(masked_fire), [0.5], colors=["red"])
    cbar = fig.colorbar(
        im,
        ax=ax,
        ticks=list(PROPAGATOR_CLASS_LABELS.keys()),
        shrink=0.8,
        label="Vegetation / fuel class",
    )
    cbar.ax.set_yticklabels(PROPAGATOR_CLASS_LABELS.values())
    fig.savefig(OUTPUT_DIR / f"veg_map{timestep + 1:02d}_fire_scar.png")
    plt.close(fig)


def save_proportions_over_time(
    proportions_history: np.ndarray,
    fire_counts: np.ndarray,
    burned_area: np.ndarray,
    extreme_events: np.ndarray,
) -> None:
    timesteps = np.arange(1, proportions_history.shape[1] + 1)
    # fig, (ax_line, ax_bar) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 1, height_ratios=[2, 2, 1], figure=fig)
    ax_line = fig.add_subplot(gs[0, 0])
    for batllori_class in range(BATLLORI_CLASSES):
        ax_line.plot(
            proportions_history[batllori_class, :],
            label=BATLLORI_LABELS[batllori_class],
        )
    ax_line.set_title("Relative Batllori Class Area Over Time")
    ax_line.set_ylabel("Area proportion (relative to initial state)")
    ax_line.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax_line.set_xlim(0, TIMESTEPS+1)
    ax_line.label_outer()

    width = 0.4
    ax_bar = fig.add_subplot(gs[1, 0])
    bars_counts = ax_bar.bar(
        timesteps - width / 2,
        fire_counts,
        width=width,
        color="tab:orange",
        label="Wildfires",
    )
    ax_bar_area = ax_bar.twinx()
    bars_area = ax_bar_area.bar(
        timesteps + width / 2,
        burned_area,
        width=width,
        color="tab:blue",
        alpha=0.6,
        label="Burned pixels",
    )
    ax_bar.set_ylabel("Wildfires per timestep")
    ax_bar_area.set_ylabel("Burned area (pixels)")
    ax_bar.set_xlabel("Timestep")
    handles = [bars_counts, bars_area]
    labels = [h.get_label() for h in handles]
    ax_bar.legend(handles, labels, loc="upper right")
    ax_bar.set_xlim(0, TIMESTEPS+1)
    ax_bar.label_outer()

    ax_extreme = fig.add_subplot(gs[2, 0])
    ax_extreme.bar(
        timesteps,
        extreme_events,
        width=width,
        color="tab:red",
        label="Extreme events",
    )
    ax_extreme.set_ylabel("Extreme events")
    ax_extreme.set_xlabel("Timestep")
    ax_extreme.set_xlim(0, TIMESTEPS+1)
    ax_extreme.legend(loc="upper right")
    ax_extreme.label_outer()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "veg_area_over_time.png")
    plt.close(fig)


def save_batllori_heatmaps(
    batllori_veg: np.ndarray,
    mask: np.ndarray,
    fire_scars: np.ndarray,
    timestep: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    masked_fire = np.where(mask, fire_scars, np.nan)
    for idx, batllori_class in enumerate(BATLLORI_LABELS):
        axis = axes[idx]
        axis.set_title(f"Class {batllori_class}")
        batllori_slice = batllori_veg[:, :, idx]
        masked_slice = np.where(mask & (batllori_slice >= 0), batllori_slice, np.nan)
        image = axis.imshow(masked_slice, cmap="Greens", vmin=0.0, vmax=1.0)
        axis.contour(np.ma.masked_invalid(masked_fire), [0.5], colors=["red"], linewidths=0.5)
        fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"batllori_proportions_timestep_{timestep + 1:02d}.png")
    plt.close(fig)


###############################################################################
# MAIN SIMULATION LOGIC
###############################################################################

def main() -> None:
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # load rasters
    dem, raw_veg, susceptibility, mask = load_rasters()
    masked_veg = np.where(mask, raw_veg, 0)

    # setting initial Batllori state based on the land-cover map and applying initial noise
    batllori_initial = veg_propagator_to_batllori(masked_veg)
    batllori_initial = apply_initial_noise(batllori_initial, rng)
    batllori_model = Batllori6CL(initial_map=batllori_initial)
    warm_up_model(batllori_model, steps=WARMUP_STEPS)  # warm-up to let the model stabilize before starting the simulation

    # setting up data structures to track outputs
    batllori_veg = batllori_model.get_vegetation_map()
    initial_proportions = compute_initial_proportions(batllori_veg, mask)
    proportions_history = np.full((BATLLORI_CLASSES, TIMESTEPS), np.nan)
    fire_counts = np.zeros(TIMESTEPS, dtype=int)
    burned_area = np.zeros(TIMESTEPS, dtype=int)
    extreme_events = np.zeros(TIMESTEPS, dtype=int)

    # main simulation loop
    for timestep in range(TIMESTEPS):
        # get current vegetation map and translate it into PROPAGATOR land-cover classes
        batllori_veg = batllori_model.get_vegetation_map()
        propagator_veg = veg_batllori_to_propagator(batllori_veg)
        # generate fire events for the current timestep based on the current mask
        fire_events = generate_fire_events(rng, mask, susceptibility)
        n_extreme = sum(event.is_extreme for event in fire_events)  # number of extreme events in the current timestep
        
        print("-------------------------------------------------------")
        print(f"Timestep {timestep + 1}: {len(fire_events)} ignitions.")
        print(f"Number of extreme events: {n_extreme}")

        # run the fire simulation for the current vegetation state and fire events, and get the resulting fire scar map
        fire_scars, _ = run_fire_events(fire_events, dem, propagator_veg)
        # save outputs and update history
        fire_counts[timestep] = len(fire_events)
        burned_area[timestep] = np.where(mask, fire_scars > 0, False).sum()  # count of burned pixels
        extreme_events[timestep] = n_extreme
        # update the Batllori model with the fire scars as disturbances
        batllori_model.step(fire_scars)

        # update output history and save maps
        update_proportions_history(batllori_veg, mask, initial_proportions, proportions_history, timestep)
        save_vegetation_and_fire_map(batllori_veg, fire_scars, mask, timestep)
        save_proportions_over_time(proportions_history, fire_counts, burned_area, extreme_events)
        save_batllori_heatmaps(batllori_veg, mask, fire_scars, timestep)

    plt.close("all")

if __name__ == "__main__":
    main()
