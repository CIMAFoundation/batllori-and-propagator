import numpy as np
from propagator.functions import moist_proba_correction_1, p_time_wang
from propagator.loader.geotiff import PropagatorDataFromGeotiffs
from propagator.propagator import (
    Propagator,
    PropagatorBoundaryConditions,
)


v0 = np.loadtxt("data/v0_table.txt")
prob_table = np.loadtxt("data/prob_table.txt")
p_veg = np.loadtxt("data/p_vegetation.txt")


loader = PropagatorDataFromGeotiffs(
    dem_file="data/dem_clip.tif",
    veg_file="data/veg_clip.tif",
)


def get_simulator(veg: np.ndarray):
    dem = loader.get_dem()
    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=10,
        ros_0=v0,
        probability_table=prob_table,
        veg_parameters=p_veg,
        do_spotting=False,
        p_time_fn=p_time_wang,
        p_moist_fn=moist_proba_correction_1,
    )
    return simulator

def get_initial_veg() -> np.ndarray:
    veg = loader.get_veg()
    return veg


def create_boundary_conditions(dem: np.ndarray, probability_of_ignition=0.001) -> list[PropagatorBoundaryConditions]:
    ignition_array = np.zeros(dem.shape, dtype=np.uint8)
    for row in range(ignition_array.shape[0]):
        for col in range(ignition_array.shape[1]):
            if np.random.rand() < probability_of_ignition:
                ignition_array[row, col] = 1 

    boundary_conditions: PropagatorBoundaryConditions = PropagatorBoundaryConditions(
        time=0,
        ignitions=ignition_array,
        wind_speed=np.ones(dem.shape) * 0,
        wind_dir=np.ones(dem.shape) * 0,
        moisture=np.ones(dem.shape) * 0.05,
    )
    
    return boundary_conditions

def start_simulation(
        simulator: Propagator, 
        boundary_conditions: PropagatorBoundaryConditions,
        time_limit: float,
    ):
    
    if boundary_conditions.ignitions.sum() == 0:
        return
    
    simulator.set_boundary_conditions(boundary_conditions)
    while True:
        next_time = simulator.next_time()
        if next_time is None:
            break
        if next_time > time_limit:
            break

        simulator.step()

def get_fire_scar(simulator: Propagator) -> np.ndarray:
    output = simulator.get_output()
    fire_probability = output.fire_probability
    return fire_probability>0