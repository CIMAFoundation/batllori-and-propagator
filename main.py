from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio

from batllori_6cl import Batllori6CL
from extract_probabilities import extract_ignition_points, sample_event_durations
from propagator_module import (
    create_boundary_conditions,
    get_fire_scar,
    get_simulator,
    start_simulation,
)


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
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

@dataclass(frozen=True)
class SimulationConfig:
    timesteps: int = 30
    mean_wind_speed: float = 10.0
    std_wind_speed: float = 5.0
    mean_fuel_moisture: float = 10.0
    std_fuel_moisture: float = 5.0
    seed: int | None = 42


@dataclass(frozen=True)
class FireEvent:
    duration: int
    coord: tuple[int, int]


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
        3: np.full(BATLLORI_CLASSES, -3333.0),  # aree non vegetate
        4: np.array([1, 0, 0, 0, 0, 0]),  # praterie -> A
        5: np.array([0, 0, 0.2, 0.8, 0, 0]),  # conifere -> Sy, Sm
        0: np.full(BATLLORI_CLASSES, -9999.0),  # nodata
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
            idxmax = np.argmax(proportions)
            match idxmax:
                case 0:
                    land_cover[i, j] = 4  # praterie
                case 1:
                    land_cover[i, j] = 2  # vegetazione arbustiva
                case 2 | 3:
                    land_cover[i, j] = 5  # conifere
                case 4 | 5:
                    land_cover[i, j] = 1  # latifoglie
                case _:
                    land_cover[i, j] = 0  # nodata or unexpected

    return land_cover


def main(config: SimulationConfig = SimulationConfig()) -> None:
    rng = np.random.default_rng(config.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dem, raw_veg, mask = load_rasters()
    masked_veg = np.where(mask, raw_veg, 0)
    batllori_initial = veg_propagator_to_batllori(masked_veg)
    batllori_initial = apply_initial_noise(batllori_initial, rng)

    batllori_model = Batllori6CL(initial_map=batllori_initial)
    warm_up_model(batllori_model, steps=WARMUP_STEPS)

    batllori_veg = batllori_model.get_vegetation_map()
    initial_proportions = compute_initial_proportions(batllori_veg, mask)
    proportions_history = np.full((BATLLORI_CLASSES, config.timesteps), np.nan)

    for timestep in range(config.timesteps):
        batllori_veg = batllori_model.get_vegetation_map()
        propagator_veg = veg_batllori_to_propagator(batllori_veg)

        fire_events = generate_fire_events()
        print(f"Timestep {timestep + 1}: {len(fire_events)} ignitions.")

        fire_scars, _ = run_fire_events(fire_events, dem, propagator_veg, rng, config)
        batllori_model.step(fire_scars)

        update_proportions_history(
            batllori_veg, mask, initial_proportions, proportions_history, timestep
        )
        save_vegetation_and_fire_map(batllori_veg, fire_scars, timestep)
        save_proportions_over_time(proportions_history)
        save_batllori_heatmaps(batllori_veg, mask, timestep)

    plt.close("all")


def load_rasters() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rio.open(DEM_PATH) as dem_src:
        dem = dem_src.read(1).astype("int16")
    with rio.open(VEG_PATH) as veg_src:
        veg = veg_src.read(1).astype("int8")
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


def generate_fire_events() -> list[FireEvent]:
    event_durations = sample_event_durations()
    ignition_coords = extract_ignition_points(len(event_durations))
    return [FireEvent(duration, coord) for duration, coord in zip(event_durations, ignition_coords)]


def run_fire_events(
    events: Iterable[FireEvent],
    dem: np.ndarray,
    veg: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    fire_scars_list = []
    fire_intensities_list = []

    for event in events:
        fire_scar, intensity = simulate_single_fire(dem, veg, event, rng, config)
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
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the external propagator for a single ignition event."""
    simulator = get_simulator(dem, veg)
    wind_speed = max(0.0, float(rng.normal(config.mean_wind_speed, config.std_wind_speed)))
    wind_direction = float(rng.uniform(0, 360))
    fuel_moisture = max(
        0.0, float(rng.normal(config.mean_fuel_moisture, config.std_fuel_moisture))
    )
    boundary_conditions = create_boundary_conditions(
        veg,
        wind_speed,
        wind_direction,
        fuel_moisture,
        event.coord,
    )
    start_simulation(simulator, boundary_conditions, time_limit=event.duration * 3600)
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
        history[batllori_class, timestep] = (
            batllori_class_sum / initial_proportions[batllori_class]
        )


def save_vegetation_and_fire_map(
    batllori_veg: np.ndarray,
    fire_scars: np.ndarray,
    timestep: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(veg_batllori_to_propagator(batllori_veg), cmap="Set2")
    ax.contour(fire_scars, [0.5], colors=["red"])
    fig.savefig(OUTPUT_DIR / f"veg_map{timestep + 1:02d}_fire_scar.png")
    plt.close(fig)


def save_proportions_over_time(proportions_history: np.ndarray) -> None:
    fig, ax = plt.subplots()
    for batllori_class in range(BATLLORI_CLASSES):
        ax.plot(
            proportions_history[batllori_class, :],
            label=BATLLORI_LABELS[batllori_class],
        )
    ax.set_title("Relative Batllori Class Area Over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Area proportion (relative to initial state)")
    ax.legend()
    fig.savefig(OUTPUT_DIR / "veg_area_over_time.png")
    plt.close(fig)


def save_batllori_heatmaps(
    batllori_veg: np.ndarray,
    mask: np.ndarray,
    timestep: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for batllori_class in range(BATLLORI_CLASSES):
        axis = axes[batllori_class]
        axis.set_title(f"Class {batllori_class + 1}")
        batllori_slice = batllori_veg[:, :, batllori_class]
        image = axis.imshow(np.where(mask & (batllori_slice >= 0), batllori_slice, np.nan), cmap="Greens")
        fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"batllori_proportions_timestep_{timestep + 1:02d}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
