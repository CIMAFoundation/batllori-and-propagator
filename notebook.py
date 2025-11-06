import marimo

__generated_with = "0.17.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from propagator_module import get_initial_data, get_simulator, create_boundary_conditions, start_simulation, get_fire_scar
    from data import create_colormap
    from batllori_4cl import Batllori4CL
    return (
        Batllori4CL,
        create_boundary_conditions,
        create_colormap,
        dataclass,
        get_fire_scar,
        get_initial_data,
        get_simulator,
        mo,
        np,
        plt,
        start_simulation,
    )


@app.cell
def _(create_colormap, get_initial_data):
    cmap, norm = create_colormap()

    # load initial vegetation from propagator utils
    dem, initial_veg = get_initial_data(
        "data/dem_monti_pisani.tif",
        "data/clc_2018.tif"
    )
    return cmap, dem, initial_veg, norm


@app.cell
def _(mo, np):
    timesteps_slider = mo.ui.slider(
        steps=np.array([5, 10, 20, 50]),
        label="Simulation Steps",
        value=10,
    )
    ignition_slider = mo.ui.slider(
        start=1,
        stop=25,
        step=1,
        label="Number of Ignitions",
        value=5,
    )
    mean_wind_speed_slider = mo.ui.slider(
        start=0,
        stop=30,
        step=1,
        label="Mean Wind Speed (m/s)",
        value=5,
    )
    std_wind_speed_slider = mo.ui.slider(
        start=0,
        stop=10,
        step=1,
        label="Wind Speed Std Dev (m/s)",
        value=1,
    )
    mean_fuel_moisture_slider = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        label="Mean Fuel Moisture (%)",
        value=10,
    )
    std_fuel_moisture_slider = mo.ui.slider(
        start=0,
        stop=20,
        step=1,
        label="Fuel Moisture Std Dev",
        value=2,
    )
    rng_seed_slider = mo.ui.slider(
        start=0,
        stop=9999,
        step=1,
        label="Random Seed",
        value=0,
    )
    controls = mo.vstack(
        [
            timesteps_slider,
            ignition_slider,
            mean_wind_speed_slider,
            std_wind_speed_slider,
            mean_fuel_moisture_slider,
            std_fuel_moisture_slider,
            rng_seed_slider,
        ],
        gap=0,
    )
    controls
    return (
        ignition_slider,
        mean_fuel_moisture_slider,
        mean_wind_speed_slider,
        rng_seed_slider,
        std_fuel_moisture_slider,
        std_wind_speed_slider,
        timesteps_slider,
    )


@app.cell
def _(dataclass):
    @dataclass(frozen=True)
    class SimulationConfig:
        timesteps: int
        n_ignitions_rate: float
        mean_wind_speed: float
        std_wind_speed: float
        mean_fuel_moisture: float
        std_fuel_moisture: float
        seed: int | None
    return (SimulationConfig,)


@app.cell
def _(
    SimulationConfig,
    ignition_slider,
    mean_fuel_moisture_slider,
    mean_wind_speed_slider,
    rng_seed_slider,
    std_fuel_moisture_slider,
    std_wind_speed_slider,
    timesteps_slider,
):
    config = SimulationConfig(
        timesteps=int(timesteps_slider.value),
        n_ignitions_rate=float(ignition_slider.value),
        mean_wind_speed=float(mean_wind_speed_slider.value),
        std_wind_speed=float(std_wind_speed_slider.value),
        mean_fuel_moisture=float(mean_fuel_moisture_slider.value),
        std_fuel_moisture=float(std_fuel_moisture_slider.value),
        seed=int(rng_seed_slider.value) if rng_seed_slider.value else None,
    )
    return (config,)


@app.cell
def _(
    BatlloriModel,
    config,
    create_boundary_conditions,
    dem,
    get_fire_scar,
    get_simulator,
    initial_veg,
    mo,
    np,
    start_simulation,
):
    rng = np.random.default_rng(config.seed)
    batllori_model = BatlloriModel(veg=initial_veg, timesteps=config.timesteps)
    veg = initial_veg.copy()

    def generate_ignition_coords() -> list[tuple[int, int]]:
        """Sample ignition points from a Poisson process."""
        n_ignitions = int(rng.poisson(config.n_ignitions_rate))
        if n_ignitions == 0:
            return []
        n_rows, n_cols = veg.shape
        xs = rng.integers(0, n_rows, size=n_ignitions)
        ys = rng.integers(0, n_cols, size=n_ignitions)
        return list(zip(xs, ys))

    def simulate_single_fire(coord: tuple[int, int]):
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

    frames = []
    progress = mo.status.progress_bar(range(config.timesteps))
    for timestep in progress:
        ignition_coords = generate_ignition_coords()
        print(f"Timestep {timestep + 1}: {len(ignition_coords)} ignitions.")

        fire_scars_list = []
        fire_intensities_list = []
        for coord in ignition_coords:
            fire_scar, intensity = simulate_single_fire(coord)
            fire_scars_list.append(fire_scar)
            fire_intensities_list.append(intensity)

        if not fire_scars_list:
            shape = veg.shape
            fire_scars_list = [np.zeros(shape, dtype=np.uint8)]
            fire_intensities_list = [np.zeros(shape, dtype=np.float32)]

        fire_scars = np.max(np.stack(fire_scars_list), axis=0)
        fire_intensities = np.max(np.stack(fire_intensities_list), axis=0)
        batllori_model.step(fire_scars, fire_intensities)

        veg = batllori_model.get_vegetation_map()

        frames.append(
            {
                "vegetation": veg.copy(),
                "fire_scars": fire_scars_list,
                "fire_intensity": fire_intensities,
                "fire_scar_union": fire_scars,
                "ignition_coords": ignition_coords,
            }
        )
    return (frames,)


@app.cell
def _():
    return


@app.cell
def _(frames):
    frame_count = len(frames)
    return (frame_count,)


@app.cell
def _(frames, mo):
    get_pos, set_pos = mo.state(len(frames))
    return get_pos, set_pos


@app.cell
def _(frame_count, get_pos, mo, set_pos):
    # A standard slider, which will be updated by the frame_state
    max_index = max(frame_count - 1, 0)
    current_value = min(get_pos(), max_index)
    frame_slider = mo.ui.slider(
        start=0,
        stop=max_index,
        value=current_value,
        on_change=set_pos,
        show_value=True,
        label="Frame",
    )
    return (frame_slider,)


@app.cell
def _(frame_count, get_pos, mo, set_pos):
    if frame_count == 0:
        next_button = mo.ui.button(label="Next Frame", disabled=True)
    else:
        next_button = mo.ui.button(
            label="Next Frame",
            value=get_pos(),
            on_change=set_pos,
            on_click=lambda value: (value + 1) % frame_count,
        )
    return (next_button,)


@app.cell
def _(cmap, frame_count, frame_slider, frames, norm, plt):
    fig, ax = plt.subplots(figsize=(6, 5))
    frame = frames[frame_slider.value]
    veg_raster = frame["vegetation"]
    fire_scars_rasters = frame["fire_scars"]
    img = ax.imshow(veg_raster, cmap=cmap, norm=norm)
    for scar in fire_scars_rasters:
        ax.contour(scar, levels=[0.5], colors="red", linewidths=2)
    ax.set_title(f"Frame {frame_slider.value + 1} / {frame_count}")
    plt.colorbar(img, ax=ax, label="Vegetation Class")
    return fig, frame


@app.cell
def _(fig, frame, frame_slider, mo, next_button):
    ignitions = len(frame["ignition_coords"])
    burned_pixels = int(frame["fire_scar_union"].sum())
    mean_intensity = float(frame["fire_intensity"].mean())
    stats = mo.md(
        f"**Ignitions**: {ignitions}  \n"
        f"**Burned Pixels**: {burned_pixels}  \n"
        f"**Mean Intensity**: {mean_intensity:.2f}"
    )
    # A marimo cell to display the UI and reactive plot
    mo.vstack(
        [
            mo.hstack([frame_slider, next_button, stats], justify="start", gap=12),
            mo.mpl.interactive(fig),
        ]
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
