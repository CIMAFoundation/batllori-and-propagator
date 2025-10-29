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
        np,
        plt,
        start_simulation,
    )


@app.cell
def _(create_colormap, get_initial_data, plt):

    cmap, norm = create_colormap()

    # load initial vegetation from propagator utils
    dem, initial_veg = get_initial_data(
        "data/dem.tif",
        "data/veg.tif"
    )

    def plot_raster(raster, t):
        plt.figure()
        plt.imshow(raster, cmap=cmap, norm=norm)
        plt.title(f"Timestep {t+1}")
        plt.show()

    return dem, initial_veg, plot_raster


@app.cell
def _(
    BatlloriModel,
    create_boundary_conditions,
    dem,
    get_fire_scar,
    get_simulator,
    initial_veg,
    np,
    plot_raster,
    start_simulation,
):
    # initialize models
    # Parametri
    timesteps = 10


    batllori_model = BatlloriModel(veg=initial_veg, timesteps=timesteps)
    veg = initial_veg.copy()
    # Simulazione
    for t in range(timesteps):
        print(f"Simulating timestep {t+1}...")
    
        # generate random ignition points, using poisson distribution (ideally should be based on extreme events number)
        n_ignitions = np.random.poisson(5)
        ignition_coords = []
        n_rows, n_cols = veg.shape
        for _ in range(n_ignitions):
            x = np.random.randint(0, n_rows)
            y = np.random.randint(0, n_cols)
            ignition_coords.append((x, y))

        fire_scars = np.zeros((n_rows, n_cols), dtype=np.uint8)
        # now we will simulate the fires, indipendently
        for coord in ignition_coords:
            simulator = get_simulator(dem, veg)
            # extract wind speed, wind direction, fuel moisture from data 
            # for now, set them to constant values
            wind_speed = 10.0  # km/h
            wind_direction = 45.0  # degrees
            fuel_moisture = 5.0  # %

            boundary_conditions = create_boundary_conditions(veg, wind_speed, wind_direction, fuel_moisture, coord)
            start_simulation(simulator, boundary_conditions, time_limit=6*60)
            fire_scar = get_fire_scar(simulator, threshold=0.3)

            fire_scars = np.maximum(fire_scars, fire_scar.astype(np.uint8))


        # count number of pixels in fire_scar and print it
        fire_scar_count = np.sum(fire_scars[:])
        print(f"Number of pixels in fire scar: {fire_scar_count}")

        batllori_model.step(fire_scars)

        # get vegetation after batllori model step   
        veg = batllori_model.get_vegetation_map()

        veg_with_fire = veg.copy()
        veg_with_fire[fire_scars > 0] = 4
        plot_raster(veg_with_fire, t)
    


    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
