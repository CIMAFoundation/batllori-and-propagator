import numpy as np
import matplotlib.pyplot as plt
from propagator_module import get_initial_data, get_simulator, create_boundary_conditions, start_simulation, get_fire_scar
from data import G_INDEX, S_INDEX, C_INDEX, B_INDEX, create_colormap
from batllori import BatlloriModel

cmap, norm = create_colormap()
# Parametri
timesteps = 10

# load initial vegetation from propagator utils
dem, veg = get_initial_data(
    "data/dem.tif",
    "data/clc_2018.tif"
)
n_rows, n_cols = veg.shape



def plot_raster(raster, t):
    plt.figure()
    plt.imshow(raster, cmap=cmap, norm=norm)
    plt.title(f"Timestep {t+1}")
    plt.show()



batllori_model = BatlloriModel(veg=veg, timesteps=timesteps)

# Simulazione
for t in range(timesteps):
    print(f"Simulating timestep {t+1}...")
    
    
    # generate random ignition points, using poisson distribution (ideally should be based on extreme events number)
    n_ignitions = np.random.poisson(5)
    ignition_coords = []
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
    

    # Grafici dei risultati
    # Grafico 1

plt.figure()

# plt.plot(range(timesteps), G_system, label="Grassland", color="green", linewidth=1.5)
# plt.plot(range(timesteps), S_system, label="Shrubland", color="yellow", linewidth=1.5)
# plt.plot(range(timesteps), C_system, label="Coniferous", color="yellowgreen", linewidth=1.5)
# plt.plot(range(timesteps), B_system, label="Broadleaved", color=(0.5, 0.25, 0), linewidth=1.5)
# plt.xlabel("Timestep")
# plt.ylabel("Proportion")
# plt.title("System-Level Vegetation Proportions Over Time")
# plt.legend()
# plt.grid()


plt.tight_layout()
plt.show()
