# %%
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

# %%

###############################################################################
# SETTINGS AND CONFIGURATION
###############################################################################

# number of years to simulate in a single realization
TIMESTEPS = 100

# >>> Batllori model parameters
BATLLORI_CLASSES = 6
INITIAL_NOISE_STD = 0.05  # initial noise injected into vegetation proportions
WARMUP_STEPS = 5  # number of steps to warm up the Batllori model
BATLLORI_NODATA = [-9999.0, -3333.0]  # values to consider as nodata

# >>> Fire event generation parameters

# the number of fire events per year is a normal distribution
MEAN_NUMBER_EVENTS_PER_YEAR = 10
STD_NUMBER_EVENTS_PER_YEAR = 5

# no fire events situation
# MEAN_NUMBER_EVENTS_PER_YEAR = 0
# STD_NUMBER_EVENTS_PER_YEAR = 0

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

# number of stochastic realizations to run for each fire event
N_FIRE_REALIZATIONS = 5
# probability threshold to consider a cell as burned in the fire scar map
FIRE_SCAR_THRESHOLD = 0.3
# size of cells in meters
CELL_SIZE = 20

# >>> general settings
# SEED = 42
SEED = None
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# Digital Elevation Model raster path [m]
DEM_PATH = DATA_DIR / "dem.tif"
# Land-cover raster path with PROPAGATOR classes
VEG_PATH = DATA_DIR / "clc_2018.tif"

# raster path with fire susceptibility values,
# which is between 0 and 1 (high susceptibility) [OPTIONAL]
# SUSCEPTIBILITY_PATH = DATA_DIR / "susc_monti_pisani.tif"
# if no susceptibility provided, ignitions will be sampled uniformly
SUSCEPTIBILITY_PATH = None

# mask to define the area of interest (1 for valid cells, 0 for excluded cells)
MASK_PATH = DATA_DIR / "mask.tif"
# if no mask provided, consider all cells as valid
# MASK_PATH = None

# >>> plot settings
BATLLORI_LABELS = [
    "Grassland (A)",
    "Shrubs (U)",
    "Conifers - young (Sy)",
    "Conifers - mature (Sm)",
    "Broadleaves - young (Ry)",
    "Broadleaves - mature (Rm)",
]

BATLLORI_COLORS = [
    "#4fbccf",  # grassland
    "#ffd700",  # shrubs
    "#ff6b6b",  # conifers - young
    "#b60b0b",  # conifers - mature
    "#a6d96a",  # broadleaves - young
    "#00441b",  # broadleaves - mature
]

PROPAGATOR_CLASS_LABELS = {
    0: "Nodata",
    1: "Broadleaves",
    2: "Shrubs",
    3: "Bare/Non-vegetated",
    4: "Grasslands",
    5: "Conifers",
    # these classes are mapped to "Bare/Non-vegetated" in the simulation
    # and not really considered
    # 6: "Croplands and agro-forestry areas",
    # 7: "Not fire-prone forest"
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


# %%
###############################################################################
# HELPERS
###############################################################################

def load_rasters(
    mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load DEM, vegetation, susceptibility and mask rasters."""
    with rio.open(DEM_PATH) as dem_src:
        dem = dem_src.read(1).astype("int16")
    with rio.open(VEG_PATH) as veg_src:
        veg = veg_src.read(1).astype("int8")
    if SUSCEPTIBILITY_PATH is not None:
        with rio.open(SUSCEPTIBILITY_PATH) as susc_src:
            susceptibility = susc_src.read(1).astype("float32")
    else:
        # if no susceptibility provided, use uniform susceptibility
        susceptibility = np.ones(dem.shape, dtype="float32")
    if mask is None:
        if MASK_PATH is not None:
            with rio.open(MASK_PATH) as mask_src:
                mask = mask_src.read(1) > 0  # boolean mask
        else:
            # if no mask provided, consider all cells as valid
            mask = np.ones(dem.shape, dtype=bool)
    # add check that all rasters are aligned
    if not (dem.shape == veg.shape ==
            susceptibility.shape == mask.shape):  # type: ignore
        raise ValueError("Input rasters have different shapes.")
    return dem, veg, susceptibility, mask  # type: ignore


def apply_initial_noise(
    initial_map: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Inject small perturbations on vegetation proportions
    and renormalize proportion vectors
    """
    noise = rng.normal(0, INITIAL_NOISE_STD, initial_map.shape)
    perturbed = np.where(initial_map > 0, initial_map + noise, initial_map)
    sums = perturbed.sum(axis=2, keepdims=True)
    return np.where(initial_map > 0, perturbed / sums, initial_map)


def warm_up_model(model: Batllori6CL, steps: int) -> None:
    for _ in range(steps):
        model.step()


def compute_initial_proportions(
    batllori_veg: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    initial_proportions = np.zeros(BATLLORI_CLASSES)
    for batllori_class in range(BATLLORI_CLASSES):
        batllori_slice = batllori_veg[:, :, batllori_class]
        batllori_class_sum = np.where(
            mask & (batllori_slice >= 0), batllori_slice, 0).sum()
        initial_proportions[batllori_class] = batllori_class_sum
    return initial_proportions


# %%
###############################################################################
# BATLLORI-PROPAGATOR VEGETATION MAPPING
###############################################################################


def veg_propagator_to_batllori(land_cover: np.ndarray) -> np.ndarray:
    """Translate land-cover codes into vegetation proportion vectors."""
    grid_size = land_cover.shape[0]
    land_cover = land_cover.copy()
    initial_map = np.zeros(
        (grid_size, grid_size, BATLLORI_CLASSES), dtype=float)

    # mapping rules PROPAGATOR -> Batllori

    # some classes of PROPAGATOR are removed
    # "croplands" mapped to "bare/non-vegetated"
    # "not fire-prone forest" in "broadleaves"
    land_cover[land_cover == 6] = 3
    land_cover[land_cover == 7] = 1
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
                raise ValueError(f"Unexpected land-cover value {code}"
                                 f"at position ({i}, {j})")
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
    # (row, col) coordinates of the ignition point
    coord: tuple[int, int]
    # wind direction (from which wind comes) in degrees
    # (0-360, where 0 is from north, 90 is from east, etc.)
    wind_dir: float
    # wind speed in km/h
    wind_speed: float
    # fuel moisture content in percentage (0-100)
    fuel_moisture: float
    # maximum simulation time of the event in seconds
    time_limit: int
    # flag to indicate if the event is extreme
    is_extreme: bool = False

    def info(self) -> str:
        return (f"is_extreme={self.is_extreme} \t"
                f"coord={self.coord} \t"
                f"wind_dir={self.wind_dir:.1f}° \t"
                f"wind_speed={self.wind_speed:.1f} km/h \t"
                f"fuel_moisture={self.fuel_moisture:.1f}% \t"
                f"time_limit={self.time_limit}s")


def extract_ignition_points(
    n_events: int,
    rng: np.random.Generator,
    mask: np.ndarray,
    susceptibility: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Sample ignition coordinates in the masked area,
    eventually with probability coming from susceptiblity.
    """
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
            probabilities = (np.ones_like(valid_susceptibility) /
                             len(valid_susceptibility))

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
    n_events = int(rng.normal(
        MEAN_NUMBER_EVENTS_PER_YEAR,
        STD_NUMBER_EVENTS_PER_YEAR)
    )
    if n_events < 0:
        n_events = 0
    # extract ignition points
    ignition_coords = extract_ignition_points(
        n_events, rng=rng, mask=mask, susceptibility=susceptibility
    )
    # define which events are extreme
    extreme_events_flags = rng.uniform(0, 1, n_events) < PROB_EXTREME_EVENT
    # assign weather conditions and time limits based on
    # the event type and create FireEvent instances
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
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the fire simulation for a list of fire events and
    return the combined fire scar map and intensity map.
    """
    fire_scars_list = []
    fire_intensities_list = []
    print("Simulating fire events ...")
    for event in events:
        if verbose:
            print('    ' + event.info())
        fire_scar, intensity = simulate_single_fire(dem, veg, event, verbose)
        fire_scars_list.append(fire_scar)
        fire_intensities_list.append(intensity)
    print("Simulations completed.")
    if not fire_scars_list:
        shape = veg.shape
        return np.zeros(shape, dtype=np.uint8), \
            np.zeros(shape, dtype=np.float32)

    fire_scars = np.max(np.stack(fire_scars_list), axis=0)
    fire_intensities = np.max(np.stack(fire_intensities_list), axis=0)
    return fire_scars, fire_intensities


def simulate_single_fire(
    dem: np.ndarray,
    veg: np.ndarray,
    event: FireEvent,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Run the external propagator for a single ignition event."""
    simulator = get_simulator(
        dem, veg,
        realizations=N_FIRE_REALIZATIONS,
        cellsize=CELL_SIZE
    )
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
    start_simulation(simulator, boundary_conditions, time_limit, verbose)
    return get_fire_scar(simulator, threshold=FIRE_SCAR_THRESHOLD)


# %%
###############################################################################
# OUTPUT AND VISUALIZATION
###############################################################################

class SimulationSummary:

    def __init__(
        self,
        timesteps: int,
        hist_bins=30
    ):
        self.timesteps = timesteps
        self.n_classes = BATLLORI_CLASSES
        self.hist_bins = int(hist_bins)
        self.bin_edges = np.linspace(0.0, 1.0, self.hist_bins + 1)
        self.nodata_values = BATLLORI_NODATA
        # statistics storage
        self.hist_history = np.full(
            (self.timesteps, self.n_classes, self.hist_bins),
            np.nan
        )
        self.mean_history = np.full((self.timesteps, self.n_classes), np.nan)
        # fire information
        self.fire_counts = np.zeros(self.timesteps, dtype=int)
        self.burned_area = np.zeros(self.timesteps, dtype=float)
        self.extreme_events = np.zeros(self.timesteps, dtype=int)
        # checkpoints for vegetation maps and fire scars
        self.checkpoint_veg = {}  # timestep -> full array (Nx, Ny, C)
        self.checkpoint_fire = {}  # timestep -> full array (Nx, Ny)
        # info for plotting
        self.class_labels = BATLLORI_LABELS
        self.class_colors = BATLLORI_COLORS

    def update(
        self,
        time: int,
        veg_arr: np.ndarray,
        fire_count: int,
        n_extreme: int,
        burned_area: float,
        fire_scar: np.ndarray,
        checkpoint: bool = False,
    ) -> None:
        veg_arr = np.asarray(veg_arr, dtype=float)
        # check shape
        if veg_arr.ndim != 3:
            raise ValueError(f"Expected shape (Nx, Ny, C)"
                             f"got {veg_arr.shape}")
        if veg_arr.shape[2] != self.n_classes:
            raise ValueError(
                f"Expected {self.n_classes} classes, got {veg_arr.shape[2]}"
            )
        # remove nodata values from statistics
        for nodata_value in self.nodata_values:
            veg_arr = np.where(veg_arr == nodata_value, np.nan, veg_arr)
        # check range
        if np.nanmin(veg_arr) < 0 or np.nanmax(veg_arr) > 1:
            raise ValueError("Values must be between 0 and 1")
        # compute statistics
        flat = veg_arr.reshape(-1, self.n_classes)  # (Npix, C)
        counts = np.stack(
                [
                    np.histogram(flat[:, c], bins=self.bin_edges)[0]
                    for c in range(self.n_classes)
                ],
                axis=0,
            )  # (C, B)
        means = np.nanmean(flat, axis=0)  # (C,)
        # store statistics
        self.hist_history[time] = counts
        self.mean_history[time] = means
        # add information about fire events
        self.fire_counts[time] = fire_count
        self.burned_area[time] = burned_area
        self.extreme_events[time] = n_extreme
        # checkpoint storage > store a copy of the full array
        if checkpoint:
            self.checkpoint_veg[time] = veg_arr.copy()
            self.checkpoint_fire[time] = fire_scar.copy()

    def clean_checkpoints(self):
        self.checkpoint_veg.clear()
        self.checkpoint_fire.clear()

    def plot_timeseries(self, figsize=(10, 5)):
        """
        One line per class: domain mean fraction over time.
        """
        times = np.arange(self.timesteps)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 1, figure=fig)
        # subplot on vegetation timeseires
        ax = fig.add_subplot(gs[0:2, 0])
        for c, label in enumerate(self.class_labels):
            ax.plot(
                times, self.mean_history[:, c],
                label=label, color=self.class_colors[c]
            )
        ax.set_title("Mean vegetation fraction over time")
        ax.set_xlabel("Time")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, self.timesteps-1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        ax.label_outer()
        # subplot on fire events
        ax_events = fig.add_subplot(gs[2, 0], sharex=ax)
        ax_events.bar(
            times, self.burned_area,
            label="Burned area", color="tab:blue", alpha=0.7
        )
        ax_events.set_title("Burned area over time")
        ax_events.set_xlabel("Time")
        ax_events.set_ylabel("ha")
        ax_events.grid(True, alpha=0.3)
        ax_events.set_ylim(bottom=0)
        ax_events.set_xlim(0, self.timesteps-1)
        fig.tight_layout()
        return fig, ax

    def plot_domain_composition(self, figsize=(10, 5)):
        """
        Stacked area chart of domain composition over time.
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 1, figure=fig)
        ax = fig.add_subplot(gs[0:2, 0])
        # plot domain composition
        ax.stackplot(
            np.arange(self.timesteps), self.mean_history.T,
            labels=self.class_labels, colors=self.class_colors,
            alpha=0.7
        )
        ax.set_title("Mean vegetation fraction over time")
        ax.set_xlabel("Time")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, self.timesteps-1)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        # plot burned area on a separate subplot
        ax_events = fig.add_subplot(gs[2, 0], sharex=ax)
        ax_events.bar(
            np.arange(self.timesteps), self.burned_area,
            label="Burned area", color="tab:blue", alpha=0.7
        )
        ax_events.set_title("Burned area over time")
        ax_events.set_xlabel("Time")
        ax_events.set_ylabel("ha")
        ax_events.grid(True, alpha=0.3)
        ax_events.set_ylim(bottom=0)
        ax_events.set_xlim(0, self.timesteps-1)
        fig.tight_layout()
        return fig, ax

    def plot_histograms(self, times: list[int], figsize=(10, 5)):
        """
        Plot histograms for specified time steps.
        """
        fig, axes = plt.subplots(
            1, len(times),
            figsize=figsize, constrained_layout=True
        )
        if len(times) == 1:
            axes = [axes]
        for t, ax in zip(times, axes):
            for c in range(self.n_classes):
                counts = self.hist_history[t, c]
                centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
                widths = np.diff(self.bin_edges)
                ax.bar(
                    centers,
                    counts,
                    width=widths,
                    alpha=0.7,
                    label=self.class_labels[c],
                    color=self.class_colors[c]
                )
            ax.set_title(f"Histogram of vegetation fractions at time {t}")
            ax.set_xlabel("Fraction")
            ax.set_ylabel("Pixel count")
            ax.grid(True, alpha=0.3)
            ax.legend()
        return fig, axes

    def plot_checkpoint_map(self, time: int, figsize=(10, 5)):
        # check if time is in checkpoints
        if time not in self.checkpoint_veg or time not in self.checkpoint_fire:
            raise ValueError(f"Checkpoint for time {time} not found")
        veg = self.checkpoint_veg[time]
        fire_scars = self.checkpoint_fire[time]
        n_classes = self.n_classes
        nrows = 2
        ncols = max(n_classes//nrows, 1)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.ravel()
        for idx, c in enumerate(self.class_labels):
            axis = axes[idx]
            axis.set_title(f"{c}")
            veg_slice = veg[:, :, idx]
            image = axis.imshow(veg_slice, cmap="Greens", vmin=0.0, vmax=1.0)
            axis.contour(
                np.ma.masked_invalid(fire_scars), [0.5],
                colors=["red"], linewidths=0.5
            )
            # remove ticks on both axis
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis)
        fig.tight_layout()
        return fig, axes


# %%
###############################################################################
# MAIN SIMULATION LOGIC
###############################################################################

def main() -> SimulationSummary:
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # load rasters
    dem, raw_veg, susceptibility, mask = load_rasters()
    masked_veg = np.where(mask, raw_veg, 0)

    # setting initial Batllori state based on the land-cover map
    # and applying initial noise
    print("Initializing Batllori model ...")
    batllori_initial = veg_propagator_to_batllori(masked_veg)
    batllori_initial = apply_initial_noise(batllori_initial, rng)
    batllori_model = Batllori6CL(initial_map=batllori_initial)
    # warm-up to let the model stabilize before starting the simulation
    warm_up_model(batllori_model, steps=WARMUP_STEPS)

    # setting up data structure to track outputs
    summary = SimulationSummary(timesteps=TIMESTEPS+1, hist_bins=30)

    # get current vegetation map -> initial condition
    batllori_veg = batllori_model.get_vegetation_map()
    # add initial information
    summary.update(
        time=0,
        veg_arr=batllori_veg,
        fire_count=0,
        n_extreme=0,
        burned_area=0,
        checkpoint=True,  # store checkpoints for all timesteps
        fire_scar=np.zeros(batllori_veg.shape[0:2], dtype=np.uint8)
    )
    # plot > initial conditions
    fig, _ = summary.plot_timeseries()
    fig.savefig(OUTPUT_DIR / "timeseries.png")
    plt.close(fig)
    fig, _ = summary.plot_domain_composition()
    fig.savefig(OUTPUT_DIR / "domain_composition.png")
    plt.close(fig)
    fig, _ = summary.plot_checkpoint_map(time=0)
    fig.savefig(OUTPUT_DIR / f"timeseries_timestep_{0}.png")
    plt.close(fig)
    summary.clean_checkpoints()  # clear checkpoints to save memory

    # main simulation loop
    for timestep in range(1, TIMESTEPS+1):
        print("-------------------------------------------------------")
        print(f"Timestep {timestep}/{TIMESTEPS}")

        # translate vegetation map into PROPAGATOR land-cover classes
        propagator_veg = veg_batllori_to_propagator(batllori_veg)

        # generate fire events for the current timestep based on the mask
        fire_events = generate_fire_events(rng, mask, susceptibility)
        # number of fire events in the current timestep
        fire_count = len(fire_events)
        # number of extreme events in the current timestep
        n_extreme = sum(event.is_extreme for event in fire_events)

        print(f"Ignitions: {len(fire_events)} - extreme events: {n_extreme}")

        # run the fire simulation for the current vegetation state and
        # fire events, and get the resulting fire scar map
        fire_scars, _ = run_fire_events(
            fire_events, dem, propagator_veg,
            verbose=False
        )
        # count of burned pixels
        cells_burnt = np.where(mask, fire_scars > 0, False).sum()
        burned_area = cells_burnt * (CELL_SIZE ** 2) / 10000  # in hectares

        print(f"burned area [ha]: {burned_area}")

        # update the Batllori model with the fire scars as disturbances
        batllori_model.step(fire_scars)
        # get new vegetation map to be saved > used in the next simulation
        batllori_veg = batllori_model.get_vegetation_map()

        # update summary statistics
        summary.update(
            time=timestep,
            veg_arr=batllori_veg,
            fire_count=fire_count,
            n_extreme=n_extreme,
            burned_area=burned_area,
            checkpoint=True,  # store checkpoints for all timesteps
            fire_scar=fire_scars
        )

        # plot
        fig, _ = summary.plot_timeseries()
        fig.savefig(OUTPUT_DIR / "timeseries.png")
        plt.close(fig)
        fig, _ = summary.plot_domain_composition()
        fig.savefig(OUTPUT_DIR / "domain_composition.png")
        plt.close(fig)
        fig, _ = summary.plot_checkpoint_map(time=timestep)
        fig.savefig(OUTPUT_DIR / f"timeseries_timestep_{timestep}.png")
        plt.close(fig)
        summary.clean_checkpoints()  # clear checkpoints to save memory

    return summary


# %%

if __name__ == "__main__":
    main()

# %%
