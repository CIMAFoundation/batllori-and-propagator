import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    from propagator_module import get_initial_data, get_simulator, create_boundary_conditions, start_simulation, get_fire_scar
    from data import create_colormap
    from batllori import BatlloriModel
    return (
        BatlloriModel,
        create_boundary_conditions,
        create_colormap,
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
    return dem, initial_veg


@app.cell
def _(mo, np):
    timesteps_slider = mo.ui.slider(steps=np.array([ 5, 10, 20, 50]), label="Simulation Steps", value=5)
    ignition_slider = mo.ui.slider(start=1, stop=20, step=1, label="Number of Ignitions", value=5)
    mean_wind_speed_slider = mo.ui.slider(start=0, stop=50, step=5, label="Mean Wind Speed (m/s)", value=5.0)
    std_wind_speed_slider = mo.ui.slider(start=0, stop=5, step=1, label="Wind Speed Std Dev (km/h)", value=0.0)
    timesteps_slider
    mean_fuel_moisture_slider = mo.ui.slider(start=0, stop=100, step=5, label="Mean Fuel Moisture", value=5)
    std_fuel_moisture_slider = mo.ui.slider(start=0, stop=5, step=1, label="Fuel Moisture Std Dev", value=0)
    mo.vstack([
        timesteps_slider,
        ignition_slider,
        mean_wind_speed_slider, 
        std_wind_speed_slider,
        mean_fuel_moisture_slider, 
        std_fuel_moisture_slider
    ])
    return (
        ignition_slider,
        mean_fuel_moisture_slider,
        mean_wind_speed_slider,
        std_fuel_moisture_slider,
        std_wind_speed_slider,
        timesteps_slider,
    )


@app.cell
def _(
    ignition_slider,
    mean_fuel_moisture_slider,
    mean_wind_speed_slider,
    std_fuel_moisture_slider,
    std_wind_speed_slider,
    timesteps_slider,
):
    # initialize models
    # Parametri
    timesteps = timesteps_slider.value
    n_ignitions_coefficient = ignition_slider.value
    mean_wind_speed = mean_wind_speed_slider.value
    std_wind_speed = std_wind_speed_slider.value
    mean_fuel_moisture = mean_fuel_moisture_slider.value
    std_fuel_moisture = std_fuel_moisture_slider.value
    return (
        mean_fuel_moisture,
        mean_wind_speed,
        n_ignitions_coefficient,
        std_fuel_moisture,
        std_wind_speed,
        timesteps,
    )


@app.cell
def _(
    BatlloriModel,
    create_boundary_conditions,
    dem,
    get_fire_scar,
    get_simulator,
    initial_veg,
    mean_fuel_moisture,
    mean_wind_speed,
    mo,
    n_ignitions_coefficient,
    np,
    start_simulation,
    std_fuel_moisture,
    std_wind_speed,
    timesteps,
):
    batllori_model = BatlloriModel(veg=initial_veg, timesteps=timesteps)
    veg = initial_veg.copy()

    frames = []
    # iterate on the number of steps and get the information about extreme events from data
    # use info on extreme events to generate: number of ignition (vary poissons' distribution parameter).
    # for each event

    # Simulazione
    for t in mo.status.progress_bar(range(timesteps)):
        print(f"Simulating timestep {t+1}...")

        # generate random ignition points, using poisson distribution (ideally should be based on extreme events number)
        n_ignitions = np.random.poisson(n_ignitions_coefficient)
        ignition_coords = []
        n_rows, n_cols = veg.shape
        for _ in range(n_ignitions):
            x = np.random.randint(0, n_rows)
            y = np.random.randint(0, n_cols)
            ignition_coords.append((x, y))

        print(f"Generated {n_ignitions} ignition points.")

        fire_scars_list = []
        fire_intensities_list = []
        # now we will simulate the fires, indipendently
        for coord in ignition_coords:
            print(f"Simulating fire at ignition point {coord}...")
            simulator = get_simulator(dem, veg)
            # extract wind speed, wind direction, fuel moisture from data 
            # for now, set them to constant values
            wind_speed = np.random.normal(mean_wind_speed, std_wind_speed)
            wind_direction = np.random.rand() * 360
            fuel_moisture = np.random.normal(mean_fuel_moisture, std_fuel_moisture)

            boundary_conditions = create_boundary_conditions(veg, wind_speed, wind_direction, fuel_moisture, coord)
            start_simulation(simulator, boundary_conditions, time_limit=6*60*60)
            fire_scar, intensity = get_fire_scar(simulator, threshold=0.3)
            fire_scars_list.append(fire_scar)
            fire_intensities_list.append(intensity)

        if len(fire_scars_list) == 0:
            fire_scars_list = [np.zeros((n_rows, n_cols), dtype=np.uint8)]
            fire_intensities_list = [np.zeros((n_rows, n_cols), dtype=np.float32)]

        fire_scars = np.max((fire_scars_list), axis=0)
        fire_intensities = np.max(np.stack(fire_intensities_list), axis=0)
        # count number of pixels in fire_scar and print it
        fire_scar_count = np.sum(fire_scars[:])
        print(f"Number of pixels in fire scar: {fire_scar_count}")

        batllori_model.step(fire_scars, fire_intensities)

        # get vegetation after batllori model step   
        veg = batllori_model.get_vegetation_map()

        frames.append((veg.copy(), fire_scars_list))
    return frames, t


@app.cell
def _():
    return


@app.cell
def _(mo, t):
    get_pos, set_pos = mo.state(t)
    return get_pos, set_pos


@app.cell
def _(get_pos, mo, set_pos, timesteps):
    # A standard slider, which will be updated by the frame_state
    frame_slider = mo.ui.slider(
        start=0,
        stop=timesteps - 1,
        value=get_pos(),
        on_change=lambda v: set_pos(v),
        show_value=True,
        label="Frame",
    )
    return (frame_slider,)


@app.cell
def _(get_pos, mo, set_pos, timesteps):
    next_button = mo.ui.button(
        value=get_pos(),
        on_change=set_pos,
        on_click=lambda value: (value+1)%timesteps
    )
    return (next_button,)


@app.cell
def _(frame_slider, frames, plt):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.gca()
    veg_raster, fire_scars_rasters = frames[frame_slider.value]
    img = ax.imshow(veg_raster, cmap="viridis")
    for scar in fire_scars_rasters:
        ax.contour(scar, levels=[0.5], colors='red', linewidths=2)
    ax.set_title(f"Frame {frame_slider.value}")
    plt.colorbar(img, ax=ax, label="Intensity")
    return (fig,)


@app.cell
def _(fig, frame_slider, mo, next_button):
    # A marimo cell to display the UI and reactive plot
    mo.vstack([
        mo.hstack([frame_slider, next_button], justify='start'),
        mo.mpl.interactive(fig)
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
