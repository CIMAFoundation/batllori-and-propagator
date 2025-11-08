
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from propagator_module import get_simulator, create_boundary_conditions, start_simulation, get_fire_scar
from extract_probabilities import sample_event_durations, extract_ignition_points
import rasterio as rio

from batllori_6cl import Batllori6CL

rng = np.random.default_rng(42)

@dataclass(frozen=True)
class SimulationConfig:
    timesteps: int
    mean_wind_speed: float
    std_wind_speed: float
    mean_fuel_moisture: float
    std_fuel_moisture: float
    seed: int | None

config = SimulationConfig(
    timesteps=30,
    mean_wind_speed=10.0,
    std_wind_speed=5.0,
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
        0: np.full(6, -9999.0),  # nodata
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

def generate_ignition_coords(veg: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    """Generate ignition coordinates based on sampled event durations."""
    event_durations = sample_event_durations()
    ignition_coords = extract_ignition_points(len(event_durations))
    return list(zip(event_durations, ignition_coords))


def simulate_single_fire(veg: np.ndarray, coord: tuple[int, int], duration: int):
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
    start_simulation(
        simulator, 
        boundary_conditions, 
        time_limit=duration * 3600
    )
    return get_fire_scar(simulator, threshold=0.3)


with rio.open('data/dem.tif') as dem_src:
    dem = dem_src.read(1).astype("int16")   
with rio.open('data/clc_2018.tif') as veg_src:
    initial_veg = veg_src.read(1).astype("int8")
with rio.open('data/mask.tif') as mask_src:
    mask = mask_src.read(1) > 0
initial_veg = np.where(mask, initial_veg, 0)  # set non-vegetated areas


batllori_veg_no_noise = veg_propagator_to_batllori(initial_veg)
# inject small perturbations in the initial state and keep proportions normalized
noise = rng.normal(0, 0.05, batllori_veg_no_noise.shape)
batllori_veg_with_noise = np.where(batllori_veg_no_noise>0, noise+batllori_veg_no_noise, batllori_veg_no_noise)
sums = batllori_veg_with_noise.sum(axis=2, keepdims=True)
batllori_veg = np.where(batllori_veg_no_noise>0, batllori_veg_with_noise / sums, batllori_veg_no_noise)

batllori_model = Batllori6CL(initial_map=batllori_veg)
# run the simulation for initialization purposes at least 5 timesteps
for _ in range(5):
    batllori_model.step()

batllori_veg = batllori_model.get_vegetation_map()
batllori_proportions_over_time = np.full((6, config.timesteps), np.nan)
initial_proportions = np.zeros(6)
for batllori_class in range(6):
    batllori_slice = batllori_veg[:,:, batllori_class].copy()
    
    batllori_class_sum = np.where(mask & (batllori_slice>=0), batllori_slice, 0).sum()
    initial_proportions[batllori_class] = batllori_class_sum


for timestep in range(config.timesteps):
    batllori_veg = batllori_model.get_vegetation_map()
    propagator_veg = veg_batllori_to_propagator(batllori_veg)
    duration_and_ignition_coords = generate_ignition_coords(propagator_veg)
    print(f"Timestep {timestep + 1}: {len(duration_and_ignition_coords)} ignitions.")

    fire_scars_list = []
    fire_intensities_list = []
    
    for duration, coord in duration_and_ignition_coords:
        # print(f' Simulating fire at {coord} for {duration} hours...')
        fire_scar, intensity = simulate_single_fire(propagator_veg, coord, duration)
        fire_scars_list.append(fire_scar)
        fire_intensities_list.append(intensity)

    if not fire_scars_list:
        shape = batllori_veg.shape
        fire_scars_list = [np.zeros(shape, dtype=np.uint8)]
        fire_intensities_list = [np.zeros(shape, dtype=np.float32)]


    fire_scars = np.max(np.stack(fire_scars_list), axis=0)
    fire_intensities = np.max(np.stack(fire_intensities_list), axis=0)
    batllori_model.step(fire_scars)

    # save images for visualization to output/timestep_{timestep + 1:02d}.png
    # vegetation and fire scars on top
    plt.figure(figsize=(12, 6))
    plt.imshow(veg_batllori_to_propagator(batllori_veg), cmap='Set2')
    plt.contour(fire_scars, [0.5], colors=['red'])
    plt.savefig(f'output/veg_map{timestep + 1:02d}_fire_scar.png')

    plt.figure()
    for batllori_class in range(6):
        batllori_slice = batllori_veg[:,:, batllori_class].copy()
        
        batllori_class_sum = np.where(mask & (batllori_slice>=0), batllori_slice, 0).sum()
        batllori_proportions_over_time[batllori_class, timestep] = batllori_class_sum / initial_proportions[batllori_class]
        plt.plot(batllori_proportions_over_time[batllori_class,:], label=f'proportion_{batllori_class+1}')
    
    plt.legend()
    plt.savefig('output/veg_area_over_time.png')

    # create a 6-plot figure of batllori proportions
    plt.figure(figsize=(12, 8))
    for batllori_class in range(6):
        plt.subplot(2, 3, batllori_class + 1)
        plt.title(f'Class {batllori_class + 1}')
        batllori_slice = batllori_veg[:,:, batllori_class]
        plt.imshow(np.where(mask & (batllori_slice>=0), batllori_slice, np.nan), cmap='Greens')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'output/batllori_proportions_timestep_{timestep + 1:02d}.png')
    plt.close('all')

