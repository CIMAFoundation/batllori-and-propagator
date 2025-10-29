import numpy as np

from propagator.core import (  # type: ignore
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
    Propagator,
)
from propagator.io import PropagatorDataFromGeotiffs # type: ignore


loader = PropagatorDataFromGeotiffs(
    dem_file="data/dem_clip.tif",
    veg_file="data/veg_clip.tif",
)


def get_simulator(veg: np.ndarray) -> Propagator:
    dem = loader.get_dem()
    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=10,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="raise",
    )
    return simulator

def get_initial_veg() -> np.ndarray:
    veg = loader.get_veg()
    return veg


def create_boundary_conditions(
        dem: np.ndarray,
        wind_speed: float, 
        wind_direction: float,
        fuel_moisture: float,
        ignition_coords: tuple[int, int],
    ) -> BoundaryConditions:
    """Create boundary conditions for the simulation including ignition mask, wind, and moisture.   
    Parameters
    ----------
    dem : np.ndarray
        A 2D numpy array representing the digital elevation model.
    wind_speed : float
        The wind speed to be applied uniformly across the grid. [km/h]
    wind_direction : float
        The wind direction to be applied uniformly across the grid. [degrees, clockwise, north->south is 0°]
    fuel_moisture : float
        The fuel moisture content to be applied uniformly across the grid. [%]
    probability_of_ignition : float
        The probability of ignition for each cell in the grid (optional, default is 0.001).
    Returns
    -------
    BoundaryConditions
        The boundary conditions including ignition mask, wind, and moisture.
    """
    ignition_array = np.zeros(dem.shape, dtype=np.uint8)
    ignition_array[ignition_coords] = 1
    
    boundary_conditions: BoundaryConditions = BoundaryConditions(
        time=0,
        ignition_mask=ignition_array,
        wind_speed=np.ones(dem.shape) * wind_speed,
        wind_dir=np.ones(dem.shape) * wind_direction,
        moisture=np.ones(dem.shape) * fuel_moisture,
    )
    
    return boundary_conditions

def start_simulation(
        simulator: Propagator, 
        boundary_conditions: BoundaryConditions,
        time_limit: int,
    ):
    """
    Start the fire simulation with given boundary conditions up to a time limit (in seconds).

    Parameters
    ----------
    simulator : Propagator
        The fire propagator simulator instance.
    boundary_conditions : BoundaryConditions
        The boundary conditions including ignition mask, wind, and moisture.
    time_limit : int
        The maximum simulation time in seconds.
    """
    
    if boundary_conditions.ignition_mask is None:
        return
    
    if boundary_conditions.ignition_mask.sum() == 0:
        return
    
    simulator.set_boundary_conditions(boundary_conditions)
    next_time = 0
    while next_time < time_limit:
        next_time = simulator.next_time()
        if next_time is None:
            break
        if next_time > time_limit:
            break

        simulator.step()

def get_fire_scar(simulator: Propagator, threshold: float) -> np.ndarray:
    """Retrieve the fire scar raster from the simulator after the simulation.
    
    Parameters
    ----------
    simulator : Propagator
        The fire propagator simulator instance.
    threshold : float
        The threshold for determining burned areas.
    
    Returns
    -------
    np.ndarray
        A 2D numpy array representing the fire scar (1 for burned, 0 for unburned).
    """
    output = simulator.get_output()
    fire_probability = output.fire_probability
    return fire_probability > threshold