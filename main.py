
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from propagator_module import get_initial_data, get_simulator, create_boundary_conditions, start_simulation, get_fire_scar

from batllori_6cl import Batllori6CL

rng = np.random.default_rng(42)

@dataclass(frozen=True)
class SimulationConfig:
    timesteps: int
    n_ignitions_rate: float
    mean_wind_speed: float
    std_wind_speed: float
    mean_fuel_moisture: float
    std_fuel_moisture: float
    seed: int | None

config = SimulationConfig(
    timesteps=10,
    n_ignitions_rate=5.0,
    mean_wind_speed=5.0,
    std_wind_speed=2.0,
    mean_fuel_moisture=10.0,
    std_fuel_moisture=5.0,
    seed=42,
)

def veg_propagator_to_batllori(land_cover: np.ndarray) -> np.ndarray:
    """Translate land-cover codes into vegetation proportion vectors."""
    grid_size = land_cover.shape[0]
    land_cover = land_cover.copy()
    initial_map = np.zeros((grid_size, grid_size, 6), dtype=float)

    land_cover[land_cover==6] = 3   # "coltivi" in "aree non o poco vegetate"
    land_cover[land_cover==7] = 1   # "boschi poco soggetti al fuoco " in "latifoglie"
    vector_map = {
        1: np.array([0, 0, 0, 0, 0.2, 0.8]),  # latifoglie -> Ry, Rm
        2: np.array([0, 1, 0, 0, 0, 0]),  # vegetazione arbustiva -> U
        3: np.full(6, -3333.0),  # aree non vegetate
        4: np.array([1, 0, 0, 0, 0, 0]),  # praterie -> A
        5: np.array([0, 0, 0.2, 0.8, 0, 0]),  # conifere -> Sy, Sm
        -3333: np.full(6, -3333.0),
        -9999: np.full(6, -9999.0),
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
            else:
                land_cover[i, j] = np.argmax(proportions)

    return land_cover

def generate_ignition_coords(veg: np.ndarray) -> list[tuple[int, int]]:
    """Sample ignition points from a Poisson process."""
    n_ignitions = int(rng.poisson(config.n_ignitions_rate))
    if n_ignitions == 0:
        return []
    n_rows, n_cols = veg.shape
    xs = rng.integers(0, n_rows, size=n_ignitions)
    ys = rng.integers(0, n_cols, size=n_ignitions)
    return list(zip(xs, ys))


def simulate_single_fire(veg: np.ndarray, coord: tuple[int, int]):
    """Run the external propagator for a single ignition."""
    simulator = get_simulator(dem, veg)
    wind_speed = max(0.0, float(rng.normal(config.mean_wind_speed, config.std_wind_speed)))
    wind_direction = float(rng.uniform(0, 360))
    fuel_moisture = max(0.0, float(rng.normal(config.mean_fuel_moisture, config.std_fuel_moisture)))
    boundary_conditions = create_boundary_conditions(
        veg,
        wind_speed,
        wind_direction,
        fuel_moisture,
        coord,
    )
    start_simulation(simulator, boundary_conditions, time_limit=6 * 60 * 60)
    return get_fire_scar(simulator, threshold=0.3)


# load initial vegetation from propagator utils
dem, initial_veg = get_initial_data(
    "data/dem_monti_pisani.tif",
    "data/clc_2018.tif"
)


batllori_veg = veg_propagator_to_batllori(initial_veg)
batllori_model = Batllori6CL(initial_map=batllori_veg)

for timestep in range(config.timesteps):
    veg = batllori_model.get_vegetation_map()
    propagator_veg = veg_batllori_to_propagator(veg)

    ignition_coords = generate_ignition_coords(propagator_veg)
    print(f"Timestep {timestep + 1}: {len(ignition_coords)} ignitions.")

    fire_scars_list = []
    fire_intensities_list = []
    

    for coord in ignition_coords:
        fire_scar, intensity = simulate_single_fire(propagator_veg, coord)
        fire_scars_list.append(fire_scar)
        fire_intensities_list.append(intensity)

    if not fire_scars_list:
        shape = veg.shape
        fire_scars_list = [np.zeros(shape, dtype=np.uint8)]
        fire_intensities_list = [np.zeros(shape, dtype=np.float32)]

    fire_scars = np.max(np.stack(fire_scars_list), axis=0)
    fire_intensities = np.max(np.stack(fire_intensities_list), axis=0)
    batllori_model.step(fire_scars)


