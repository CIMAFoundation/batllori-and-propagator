from dataclasses import dataclass, field
import random
import numpy as np
from data import G_INDEX, S_INDEX, C_INDEX, B_INDEX


TSF_classes = 3
BETA_G3 = 0.1
BETA_G = [0, 0.5 * BETA_G3, BETA_G3]
BETA_S = [0, 0.05 * BETA_G3, 0.1 * BETA_G3]
BETA_C = [0, 0.005 * BETA_G3, 0.01 * BETA_G3]
FIRE_PROBABILITY_FACTOR = 0.4
ALPHA_S = 0.2
ALPHA_C = 0.3
ALPHA_B = 1 - (ALPHA_S+ALPHA_C)
FLAMMABILITY = [0.8, 0.5, 0.2, 0.1]
FEEDBACK_TYPE = "strong"

TYPES_TO_CORINE = [G_INDEX, S_INDEX, C_INDEX, B_INDEX]


def grid_from_raster(raster: np.ndarray) -> np.ndarray:
    grid = np.empty(raster.shape, dtype=object)
    for i in range(raster.shape[0]):
        for j in range(raster.shape[1]):
            index = raster[i, j]
            if index == G_INDEX or index == 5:  # Grassland
                proportions = np.array(
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                )
            elif index == S_INDEX:  # Shrubland
                proportions = np.array(
                    [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                )
            elif index == C_INDEX:  # Coniferous
                proportions = np.array(
                    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
                )
            elif index == B_INDEX or index == 6:  # Broadleaved
                proportions = np.array(
                    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
                )
            else:
                proportions = np.array(
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )

            TSF = random.randint(1, 100)
            grid[i, j] = {"proportions": proportions, "TSF": TSF}
    grid = np.array(grid)
    return grid


@dataclass
class BatlloriModel:
    veg: np.ndarray
    timesteps: int
    

    # Dinamiche a livello di sistema
    G_system: np.ndarray = field(init=False)
    S_system: np.ndarray = field(init=False)
    C_system: np.ndarray = field(init=False)
    B_system: np.ndarray = field(init=False)

    t: int = 0


    def __post_init__(self):
        self.G_system = np.zeros(self.timesteps)
        self.S_system = np.zeros(self.timesteps)
        self.C_system = np.zeros(self.timesteps)
        self.B_system = np.zeros(self.timesteps)
        self.grid = grid_from_raster(self.veg)
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])


    def step(self, fire_scars: np.ndarray):
        G_total, S_total, C_total, B_total = 0, 0, 0, 0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                cell = self.grid[i, j]
                proportions = cell["proportions"]
                TSF = cell["TSF"]
                
                # Determina la classe TSF e i tassi di successione
                if TSF < 5:
                    level = 0
                    betag, betas, betac = BETA_G[0], BETA_S[0], BETA_C[0]
                    
                elif 5 <= TSF < 15:
                    level = 1
                    betag, betas, betac = BETA_G[1], BETA_S[1], BETA_C[1]
                    
                else:
                    level = 2
                    betag, betas, betac = BETA_G[2], BETA_S[2], BETA_C[2]
                    
                
                Gold, Sold, Cold, Bold = proportions[level]
                
                
                F: float = 0.0                
                # Probabilità di incendio
                if FEEDBACK_TYPE == "strong":
                    dominant_type = np.argmax([Gold, Sold, Cold, Bold])
                    if dominant_type == 0:
                        F = FIRE_PROBABILITY_FACTOR * Gold * FLAMMABILITY[0]
                    elif dominant_type == 1:
                        F = FIRE_PROBABILITY_FACTOR * Sold * FLAMMABILITY[1]
                    elif dominant_type == 2:
                        F = FIRE_PROBABILITY_FACTOR * Cold * FLAMMABILITY[2]
                    else:
                        F = FIRE_PROBABILITY_FACTOR * Bold * FLAMMABILITY[3]
                
                fire = fire_scars[i, j] > 0
                
                if fire:
                    TSF = 0
                    B = Bold * (1 - ALPHA_B * F)  + Cold* betac * (1 - F)
                    C = Cold * (1 - ALPHA_C * F - betac * (1 - F))  + Bold * ALPHA_B * F / 3 + Sold * betas * (1 - F) 
                    S = Sold * (1 - ALPHA_S * F - betas * (1 - F)) + Bold * ALPHA_B * F / 3 + Gold * betag * (1 - F) + Cold * ALPHA_C * F
                    G = Gold * (1 - betag * (1 - F))  + Bold * ALPHA_B * F / 3 + Sold * ALPHA_S * F 

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
                G_total += np.sum(self.grid[i, j]["proportions"][:, 0])  # Somma la colonna G per tutti i livelli TSF
                S_total += np.sum(self.grid[i, j]["proportions"][:, 1])  # Somma la colonna S per tutti i livelli TSF
                C_total += np.sum(self.grid[i, j]["proportions"][:, 2])  # Somma la colonna C per tutti i livelli TSF
                B_total += np.sum(self.grid[i, j]["proportions"][:, 3])  # Somma la colonna B per tutti i livelli TSF

        
        # Calcola proporzioni a livello di sistema
        n_cells = self.n_rows * self.n_cols
        self.G_system[self.t] = G_total / (n_cells)
        self.S_system[self.t] = S_total / (n_cells)
        self.C_system[self.t] = C_total / (n_cells)
        self.B_system[self.t] = B_total / (n_cells)

        print(f"Timestep {self.t+1} - Grass: {self.G_system[self.t]:.4f}, Shrub: {self.S_system[self.t]:.4f}, Coniferous: {self.C_system[self.t]:.4f}, Broadleaved: {self.B_system[self.t]:.4f}")
        self.t += 1



    def get_vegetation_map(self) -> np.ndarray:
        raster = np.zeros((self.n_rows, self.n_cols))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                cell = self.grid[i, j]
                proportions = cell["proportions"]
                proportions = np.sum(proportions, axis=0)

                dominant_type = np.argmax(proportions)

                raster[i, j] = TYPES_TO_CORINE[dominant_type]

        raster = raster.astype(np.uint8)
        return raster
