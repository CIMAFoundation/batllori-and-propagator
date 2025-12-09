from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from matplotlib.colors import BoundaryNorm, ListedColormap

from batllori_6cl import Batllori6CL
from propagator_module import (
    create_boundary_conditions,
    get_fire_scar,
    get_simulator,
    start_simulation,
)

TIMESTEPS = 100
OUTPUT_DIR = Path("output/normal")

MEAN_NUMBER_EVENTS_PER_YEAR = 10
STD_NUMBER_EVENTS_PER_YEAR = 5


EXTREME_EVENT_WIND_SPEED = 40.0
EXTREME_EVENT_WIND_DIRECTION = 45.0
EXTREME_EVENT_FUEL_MOISTURE = 3.0
EXTREME_TIME_LIMIT = 86400  # seconds

NORMAL_EVENT_WIND_SPEED = 5.0
NORMAL_EVENT_FUEL_MOISTURE = 15.0
NORMAL_TIME_LIMIT = 3600  # seconds


SEED = 42
DATA_DIR = Path("data")
DEM_PATH = DATA_DIR / "dem.tif"
VEG_PATH = DATA_DIR / "clc_2018.tif"
MASK_PATH = DATA_DIR / "mask.tif"

BATLLORI_CLASSES = 6
FIRE_SCAR_THRESHOLD = 0.3
INITIAL_NOISE_STD = 0.05
WARMUP_STEPS = 5

BATLLORI_LABELS = [
    "Praterie (A)",
    "Vegetazione arbustiva (U)",
    "Conifere - young (Sy)",
    "Conifere - mature (Sm)",
    "Latifoglie - young (Ry)",
    "Latifoglie - mature (Rm)",
]

PROPAGATOR_CLASS_LABELS = {
    0: "Nodata",
    1: "Latifoglie",
    2: "Vegetazione arbustiva",
    3: "Non vegetato",
    4: "Praterie",
    5: "Conifere",
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


@dataclass(frozen=True)
class FireEvent:
    coord: tuple[int, int]
    wind_dir: float
    wind_speed: float
    fuel_moisture: float
    time_limit: int



def veg_propagator_to_batllori(land_cover: np.ndarray) -> np.ndarray:
    """Translate land-cover codes into vegetation proportion vectors."""
    grid_size = land_cover.shape[0]
    land_cover = land_cover.copy()
    initial_map = np.zeros((grid_size, grid_size, BATLLORI_CLASSES), dtype=float)

    land_cover[land_cover == 6] = 3  # "coltivi" in "aree non o poco vegetate"
    land_cover[land_cover == 7] = 1  # "boschi poco soggetti al fuoco" in "latifoglie"
    vector_map = {
        1: np.array([0, 0, 0, 0, 0.1, 0.9]),  # latifoglie -> Ry, Rm
        2: np.array([0, 1, 0, 0, 0, 0]),  # vegetazione arbustiva -> U
        4: np.array([1, 0, 0, 0, 0, 0]),  # praterie -> A
        5: np.array([0, 0, 0.2, 0.8, 0, 0]),  # conifere -> Sy, Sm
        0: np.full(BATLLORI_CLASSES, -9999.0),  # nodata
        3: np.full(BATLLORI_CLASSES, -3333.0),  # aree non vegetate
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
            
            # new rules: conifers if Sy+Sm > 0.3, shrubs if U > 0.3, broadleaves if Ry+Rm > 0.7, grasslands otherwise

            sum_conifers = proportions[2] + proportions[3]
            sum_broadleaves = proportions[4] + proportions[5]
            sum_shrubs = proportions[1]
            if sum_conifers > 0.3:
                land_cover[i, j] = 5  # conifere
            elif sum_shrubs > 0.3:
                land_cover[i, j] = 2  # vegetazione arbustiva
            elif sum_broadleaves > 0.7:
                land_cover[i, j] = 1  # latifoglie
            else:
                land_cover[i, j] = 4  # praterie
    return land_cover




def load_rasters(mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rio.open(DEM_PATH) as dem_src:
        dem = dem_src.read(1).astype("int16")
    with rio.open(VEG_PATH) as veg_src:
        veg = veg_src.read(1).astype("int8")
    if mask is None:
        with rio.open(MASK_PATH) as mask_src:
            mask = mask_src.read(1) > 0
    return dem, veg, mask


def apply_initial_noise(initial_map: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject small perturbations and renormalize proportion vectors."""
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


def extract_ignition_points(
    n_events: int,
    rng: np.random.Generator,
    mask: np.ndarray 
) -> list[tuple[int, int]]:
    """Sample ignition coordinates using the susceptibility map as weights."""
    rng = rng or np.random.default_rng()

    ignition_points = []
    for _ in range(n_events):
        point = rng.choice(np.arange(mask.size))
        point = np.unravel_index(point, mask.shape)
        ignition_points.append((int(point[0]), int(point[1])))

    return ignition_points


def generate_fire_events(
    mask: np.ndarray,
    rng: np.random.Generator
) -> list[FireEvent]:
    """Generate a list of fire events for the current timestep."""
    # events per year: 30 on average (sample with gaussian)
    # extreme events: for those events, 0.01 probability of being an extreme event
    
    n_events = int(rng.normal(MEAN_NUMBER_EVENTS_PER_YEAR, STD_NUMBER_EVENTS_PER_YEAR))
    if n_events < 0:
        n_events = 0
    ignition_coords = extract_ignition_points(
        n_events, rng=rng, mask=mask
    )
    extreme_events_flags = rng.uniform(0, 1, n_events) < 0.01

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

        events.append(
            FireEvent(
                coord=coord,
                wind_speed=wind_speed,
                wind_dir=wind_direction,
                fuel_moisture=fuel_moisture,
                time_limit=time_limit,
            )
        )

    return events


def run_fire_events(
    events: Iterable[FireEvent],
    dem: np.ndarray,
    veg: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
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
    simulator = get_simulator(dem, veg, realizations=5)
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
    # ax.contour(np.ma.masked_invalid(masked_fire), [0.5], colors=["red"])
    masked_fire_ok = np.where(masked_fire >= 0.5, 1.0, np.nan)
    ax.imshow(np.ma.masked_invalid(masked_fire_ok), cmap='Reds')
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
) -> None:
    timesteps = np.arange(1, proportions_history.shape[1] + 1)
    fig, (ax_line, ax_bar) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for batllori_class in range(BATLLORI_CLASSES):
        ax_line.plot(
            proportions_history[batllori_class, :],
            label=BATLLORI_LABELS[batllori_class],
        )
    ax_line.set_title("Relative Batllori Class Area Over Time")
    ax_line.set_ylabel("Area proportion (relative to initial state)")
    ax_line.legend(loc="upper right")

    width = 0.4
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
    ax_bar.legend(handles, labels, loc="upper left")

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



def main() -> None:
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rio.open(MASK_PATH) as src:
        mask = src.read(1) > 0
    dem, raw_veg, mask = load_rasters(mask=mask)
    masked_veg = np.where(mask, raw_veg, 0)
    batllori_initial = veg_propagator_to_batllori(masked_veg)
    batllori_initial = apply_initial_noise(batllori_initial, rng)

    batllori_model = Batllori6CL(initial_map=batllori_initial)
    warm_up_model(batllori_model, steps=WARMUP_STEPS)

    batllori_veg = batllori_model.get_vegetation_map()
    initial_proportions = compute_initial_proportions(batllori_veg, mask)
    proportions_history = np.full((BATLLORI_CLASSES, TIMESTEPS), np.nan)
    fire_counts = np.zeros(TIMESTEPS, dtype=int)
    burned_area = np.zeros(TIMESTEPS, dtype=int)

    for timestep in range(TIMESTEPS):
        batllori_veg = batllori_model.get_vegetation_map()
        propagator_veg = veg_batllori_to_propagator(batllori_veg)

        fire_events = generate_fire_events(
            mask,
            rng,
        )
        print(f"Timestep {timestep + 1}: {len(fire_events)} ignitions.")

        fire_scars, _ = run_fire_events(fire_events, dem, propagator_veg, rng)
        fire_counts[timestep] = len(fire_events)
        burned_area[timestep] = np.where(mask, fire_scars > 0, False).sum()
        batllori_model.step(fire_scars)

        update_proportions_history(
            batllori_veg, mask, initial_proportions, proportions_history, timestep
        )
        save_vegetation_and_fire_map(batllori_veg, fire_scars, mask, timestep)
        save_proportions_over_time(proportions_history, fire_counts, burned_area)
        save_batllori_heatmaps(batllori_veg, mask, fire_scars, timestep)

    plt.close("all")

if __name__ == "__main__":
    main()
