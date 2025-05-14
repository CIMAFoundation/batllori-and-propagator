import numpy as np
import matplotlib.pyplot as plt
from propagator_module import get_simulator, get_initial_veg, create_boundary_conditions, start_simulation, get_fire_scar
from data import grid_from_raster, G_INDEX, S_INDEX, C_INDEX, B_INDEX, create_colormap

types_to_corine = [G_INDEX, S_INDEX, C_INDEX, B_INDEX]


cmap, norm = create_colormap()
# Parametri

timesteps = 10
TSF_classes = 3
beta_g3 = 0.1
beta_g = [0, 0.5 * beta_g3, beta_g3]
beta_s = [0, 0.05 * beta_g3, 0.1 * beta_g3]
beta_c = [0, 0.005 * beta_g3, 0.01 * beta_g3]
f = 0.4
alpha_s = 0.2
alpha_c = 0.3
alpha_b = 1 - (alpha_s+alpha_c)
flammability = [0.8, 0.5, 0.2, 0.1]
feedback_type = "strong"

# load initial vegetation from propagator utils
veg = get_initial_veg()
n_rows, n_cols = veg.shape

# Dinamiche a livello di sistema
G_system = np.zeros(timesteps)
S_system = np.zeros(timesteps)
C_system = np.zeros(timesteps)
B_system = np.zeros(timesteps)

def get_dominant_type(grid):
    raster = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            cell = grid[i, j]
            proportions = cell["proportions"]
            proportions = np.sum(proportions, axis=0)

            dominant_type = np.argmax(proportions)

            raster[i, j] = types_to_corine[dominant_type]

    raster = raster.astype(np.uint8)
    return raster

def plot_raster(raster, t):
    plt.figure()
    plt.imshow(raster, cmap=cmap, norm=norm)
    plt.title(f"Timestep {t+1}")
    plt.show()


grid = grid_from_raster(veg)
# Simulazione
for t in range(timesteps):
    print(f"Simulating timestep {t+1}...")
    simulator = get_simulator(veg)
    boundary_conditions = create_boundary_conditions(veg, probability_of_ignition=0.00001)
    start_simulation(simulator, boundary_conditions, time_limit=6*60)
    fire_scar = get_fire_scar(simulator)
    # count number of pixels in fire_scar and print it
    fire_scar_count = np.sum(fire_scar[:])
    print(f"Number of pixels in fire scar: {fire_scar_count}")

    G_total, S_total, C_total, B_total = 0, 0, 0, 0
    for i in range(n_rows):
        for j in range(n_cols):
            cell = grid[i, j]
            proportions = cell["proportions"]
            TSF = cell["TSF"]
            
            # Determina la classe TSF e i tassi di successione
            if TSF < 5:
                level = 0
                betag, betas, betac = beta_g[0], beta_s[0], beta_c[0]
                
            elif 5 <= TSF < 15:
                level = 1
                betag, betas, betac = beta_g[1], beta_s[1], beta_c[1]
                
            else:
                level = 2
                betag, betas, betac = beta_g[2], beta_s[2], beta_c[2]
                
            
            Gold, Sold, Cold, Bold = proportions[level]
            
            
            
            # Probabilità di incendio
            if feedback_type == "strong":
                dominant_type = np.argmax([Gold, Sold, Cold, Bold])
                if dominant_type == 0:
                    F = f * Gold * flammability[0]
                elif dominant_type == 1:
                    F = f * Sold * flammability[1]
                elif dominant_type == 2:
                    F = f * Cold * flammability[2]
                else:
                    F = f * Bold * flammability[3]

            fire = fire_scar[i, j] > 0
            
            if fire:
                TSF = 0
                B = Bold * (1 - alpha_b * F)  + Cold* betac * (1 - F)
                C = Cold * (1 - alpha_c * F - betac * (1 - F))  + Bold * alpha_b * F / 3 + Sold * betas * (1 - F) 
                S = Sold * (1 - alpha_s * F - betas * (1 - F)) + Bold * alpha_b * F / 3 + Gold * betag * (1 - F) + Cold * alpha_c * F
                G = Gold * (1 - betag * (1 - F))  + Bold * alpha_b * F / 3 + Sold * alpha_s * F 

            else:
                TSF += 1
                B = Bold + betac * Cold
                C = Cold * (1 - betac) + betas * Sold
                S = Sold * (1 - betas) + betag * Gold
                G = Gold * (1 - betag)
            
            proportions[level] = [G, S, C, B]
            cell["proportions"] = proportions
            cell["TSF"] = TSF
            
   
            # Accumula i totali per G, S, C e B
            G_total += np.sum(grid[i, j]["proportions"][:, 0])  # Somma la colonna G per tutti i livelli TSF
            S_total += np.sum(grid[i, j]["proportions"][:, 1])  # Somma la colonna S per tutti i livelli TSF
            C_total += np.sum(grid[i, j]["proportions"][:, 2])  # Somma la colonna C per tutti i livelli TSF
            B_total += np.sum(grid[i, j]["proportions"][:, 3])  # Somma la colonna B per tutti i livelli TSF
        
    veg = get_dominant_type(grid)
    veg_with_fire = veg.copy()
    veg_with_fire[fire_scar > 0] = 7
    plot_raster(veg_with_fire, t)
    
    # Calcola proporzioni a livello di sistema
    G_system[t] = G_total / (n_rows * n_cols)
    S_system[t] = S_total / (n_rows * n_cols)
    C_system[t] = C_total / (n_rows * n_cols)
    B_system[t] = B_total / (n_rows * n_cols)
            
    print(f"Timestep {t+1} - Grass: {G_system[t]:.4f}, Shrub: {S_system[t]:.4f}, Coniferous: {C_system[t]:.4f}, Broadleaved: {B_system[t]:.4f}")

    # Grafici dei risultati
    # Grafico 1

plt.figure()

plt.plot(range(timesteps), G_system, label="Grassland", color="green", linewidth=1.5)
plt.plot(range(timesteps), S_system, label="Shrubland", color="yellow", linewidth=1.5)
plt.plot(range(timesteps), C_system, label="Coniferous", color="yellowgreen", linewidth=1.5)
plt.plot(range(timesteps), B_system, label="Broadleaved", color=(0.5, 0.25, 0), linewidth=1.5)
plt.xlabel("Timestep")
plt.ylabel("Proportion")
plt.title("System-Level Vegetation Proportions Over Time")
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()
