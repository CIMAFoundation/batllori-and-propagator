import numpy as np
import matplotlib.pyplot as plt
import random

# Parametri
grid_size = 30
timesteps = 5000
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


# Inizializzazione della griglia
grid = np.empty((grid_size, grid_size), dtype=object)
for i in range(grid_size):
    for j in range(grid_size):
        proportions = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        TSF = random.randint(1, 100)
        grid[i, j] = {"proportions": proportions, "TSF": TSF}

# Dinamiche a livello di sistema
G_system = np.zeros(timesteps)
S_system = np.zeros(timesteps)
C_system = np.zeros(timesteps)
B_system = np.zeros(timesteps)



# Simulazione
for t in range(timesteps):
    G_total, S_total, C_total, B_total = 0, 0, 0, 0
    
    for i in range(grid_size):
        for j in range(grid_size):
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
            else:
                F = f * (Gold * flammability[0] + Sold * flammability[1] + Cold * flammability[2]+ Bold * flammability[3])
            
            # Ignizione del fuoco
            ignition = np.random.binomial(1, F)
            
            if ignition:
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

        
    
    # Calcola proporzioni a livello di sistema
    G_system[t] = G_total / (grid_size ** 2)
    S_system[t] = S_total / (grid_size ** 2)
    C_system[t] = C_total / (grid_size ** 2)
    B_system[t] = B_total / (grid_size ** 2)
            
    print(f"Timestep {t+1} - Grass: {G_system[t]:.4f}, Shrub: {S_system[t]:.4f}, Coniferous: {C_system[t]:.4f}, Broadleaved: {B_system[t]:.4f}")

# Grafici dei risultati
# Grafico 1
plt.figure(figsize=(10, 6))
plt.plot(range(timesteps), G_system, label="Grassland", color="green", linewidth=1.5)
plt.plot(range(timesteps), S_system, label="Shrubland", color="yellow", linewidth=1.5)
plt.plot(range(timesteps), C_system, label="Coniferous", color=(0.5, 0.25, 0), linewidth=1.5)
plt.plot(range(timesteps), B_system, label="Broadleaved", color="black", linewidth=1.5)
plt.xlabel("Timestep")
plt.ylabel("Proportion")
plt.title("System-Level Vegetation Proportions Over Time")
plt.legend()
plt.grid()




# Layout
plt.tight_layout()
plt.show()
