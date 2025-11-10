import numpy as np

from propagator.core import (  # type: ignore
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
    Propagator,
    PropagatorOutOfBoundsError
)



def get_simulator(dem: np.ndarray,veg: np.ndarray, realizations: int = 10) -> Propagator:
    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=realizations,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="raise",
    )
    return simulator


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
    boundary_conditions: BoundaryConditions = BoundaryConditions(
        time=0,
        ignitions=[ignition_coords],
        wind_speed=wind_speed,
        wind_dir=wind_direction,
        moisture=fuel_moisture,
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
    
    if boundary_conditions.ignitions is None:
        return
    
    if isinstance(boundary_conditions.ignitions, np.ndarray) and boundary_conditions.ignitions.sum() == 0:
        return
    elif isinstance(boundary_conditions.ignitions, list) and len(boundary_conditions.ignitions) == 0:
        return
    
    simulator.set_boundary_conditions(boundary_conditions)
    
    while simulator.next_time() is not None:
        try:
            simulator.step()
        except PropagatorOutOfBoundsError:
            print("Simulation stopped: fire reached out of bounds area.")
            break
        if simulator.time >= time_limit:
            break

def get_fire_scar(simulator: Propagator, threshold: float) -> tuple[np.ndarray, np.ndarray]:
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
    fire_intensity = output.fli_mean
    return fire_probability > threshold, fire_intensity

