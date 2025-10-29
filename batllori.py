from dataclasses import dataclass, field
import numpy as np
from data import G_INDEX, S_INDEX, C_INDEX, B_INDEX


TSF_CLASSES = 3
BETA_G3 = 0.1
BETA_G = np.array([0.0, 0.5 * BETA_G3, BETA_G3], dtype=np.float64)
BETA_S = np.array([0.0, 0.05 * BETA_G3, 0.1 * BETA_G3], dtype=np.float64)
BETA_C = np.array([0.0, 0.005 * BETA_G3, 0.01 * BETA_G3], dtype=np.float64)
FIRE_PROBABILITY_FACTOR = 0.4
ALPHA_S = 0.2
ALPHA_C = 0.3
ALPHA_B = 1 - (ALPHA_S + ALPHA_C)
FLAMMABILITY = np.array([0.8, 0.5, 0.2, 0.1], dtype=np.float64)

TYPES_TO_CORINE = np.array([G_INDEX, S_INDEX, C_INDEX, B_INDEX], dtype=np.uint8)

GRASS_TEMPLATE = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (TSF_CLASSES, 1))
SHRUB_TEMPLATE = np.tile(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float64), (TSF_CLASSES, 1))
CONIFER_TEMPLATE = np.tile(np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float64), (TSF_CLASSES, 1))
BROAD_TEMPLATE = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (TSF_CLASSES, 1))
UNIFORM_TEMPLATE = np.tile(
    np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float64), (TSF_CLASSES, 1)
)


def grid_from_raster(raster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised conversion from land-cover raster to succession proportions and TSF."""
    n_rows, n_cols = raster.shape
    proportions = np.empty((n_rows, n_cols, TSF_CLASSES, 4), dtype=np.float64)
    proportions[:] = UNIFORM_TEMPLATE

    tsf = np.random.randint(1, 101, size=(n_rows, n_cols), dtype=np.int32)

    grass_mask = (raster == G_INDEX) | (raster == 5)
    shrub_mask = raster == S_INDEX
    conifer_mask = raster == C_INDEX
    broad_mask = (raster == B_INDEX) | (raster == 6)

    proportions[grass_mask] = GRASS_TEMPLATE
    proportions[shrub_mask] = SHRUB_TEMPLATE
    proportions[conifer_mask] = CONIFER_TEMPLATE
    proportions[broad_mask] = BROAD_TEMPLATE

    return proportions, tsf


@dataclass
class BatlloriModel:
    veg: np.ndarray
    timesteps: int
    feedback_type: str = "vegetation"


    # Dinamiche a livello di sistema
    G_system: np.ndarray = field(init=False)
    S_system: np.ndarray = field(init=False)
    C_system: np.ndarray = field(init=False)
    B_system: np.ndarray = field(init=False)

    t: int = 0


    def __post_init__(self):
        if self.feedback_type not in {"vegetation", "none", "external"}:
            raise ValueError(
                "feedback_type must be one of {'vegetation', 'external', 'none'}"
            )
        if self.feedback_type == "external":
            self.feedback_type = "none"

        self.G_system = np.zeros(self.timesteps)
        self.S_system = np.zeros(self.timesteps)
        self.C_system = np.zeros(self.timesteps)
        self.B_system = np.zeros(self.timesteps)
        self.proportions, self.tsf = grid_from_raster(self.veg)
        self.n_rows, self.n_cols = self.veg.shape


    def step(self, fire_scars: np.ndarray, fire_intensity: np.ndarray | None = None):
        """Advance one timestep using an externally generated fire footprint.

        Parameters
        ----------
        fire_scars:
            Boolean or integer mask where non-zero entries denote cells that burned.
        fire_intensity:
            Optional array (same shape as `fire_scars`) with fire severity in [0, 1].
            When provided, it overrides vegetation-based feedback and is only applied
            to burned cells.
        """
        fire_mask = fire_scars.astype(bool)
        fire_flat = fire_mask.reshape(-1)

        tsf_flat = self.tsf.reshape(-1)
        levels = np.where(tsf_flat < 5, 0, np.where(tsf_flat < 15, 1, 2))

        proportions_flat = self.proportions.reshape(-1, TSF_CLASSES, 4)
        cell_idx = np.arange(proportions_flat.shape[0])
        current = proportions_flat[cell_idx, levels]

        Gold = current[:, 0]
        Sold = current[:, 1]
        Cold = current[:, 2]
        Bold = current[:, 3]

        betag = BETA_G[levels]
        betas = BETA_S[levels]
        betac = BETA_C[levels]

        if fire_intensity is not None:
            provided = np.asarray(fire_intensity, dtype=np.float64)
            if provided.shape != (self.n_rows, self.n_cols):
                raise ValueError(
                    "fire_intensity must have shape (n_rows, n_cols); "
                    f"received {provided.shape} instead of {(self.n_rows, self.n_cols)}"
                )
            F = np.zeros_like(Gold)
            F[fire_flat] = np.clip(provided.reshape(-1)[fire_flat], 0.0, 1.0)
        elif self.feedback_type == "vegetation":
            dominant = np.argmax(current, axis=1)
            dominant_prop = np.take_along_axis(
                current, dominant[:, None], axis=1
            ).squeeze(-1)
            flammability = FLAMMABILITY[dominant]
            F = FIRE_PROBABILITY_FACTOR * dominant_prop * flammability
        else:
            F = np.zeros_like(Gold)

        B_fire = Bold * (1 - ALPHA_B * F) + Cold * betac * (1 - F)
        C_fire = (
            Cold * (1 - ALPHA_C * F - betac * (1 - F))
            + Bold * ALPHA_B * F / 3
            + Sold * betas * (1 - F)
        )
        S_fire = (
            Sold * (1 - ALPHA_S * F - betas * (1 - F))
            + Bold * ALPHA_B * F / 3
            + Gold * betag * (1 - F)
            + Cold * ALPHA_C * F
        )
        G_fire = (
            Gold * (1 - betag * (1 - F))
            + Bold * ALPHA_B * F / 3
            + Sold * ALPHA_S * F
        )

        B_nofire = Bold + betac * Cold
        C_nofire = Cold * (1 - betac) + betas * Sold
        S_nofire = Sold * (1 - betas) + betag * Gold
        G_nofire = Gold * (1 - betag)

        B_new = np.where(fire_flat, B_fire, B_nofire)
        C_new = np.where(fire_flat, C_fire, C_nofire)
        S_new = np.where(fire_flat, S_fire, S_nofire)
        G_new = np.where(fire_flat, G_fire, G_nofire)

        new_values = np.column_stack((G_new, S_new, C_new, B_new))
        proportions_flat[cell_idx, levels] = new_values

        tsf_flat[fire_flat] = 0
        tsf_flat[~fire_flat] += 1

        # Calcola proporzioni a livello di sistema
        n_cells = self.n_rows * self.n_cols
        G_total = np.sum(self.proportions[..., 0])
        S_total = np.sum(self.proportions[..., 1])
        C_total = np.sum(self.proportions[..., 2])
        B_total = np.sum(self.proportions[..., 3])

        self.G_system[self.t] = G_total / n_cells
        self.S_system[self.t] = S_total / n_cells
        self.C_system[self.t] = C_total / n_cells
        self.B_system[self.t] = B_total / n_cells

        print(f"Timestep {self.t+1} - Grass: {self.G_system[self.t]:.4f}, Shrub: {self.S_system[self.t]:.4f}, Coniferous: {self.C_system[self.t]:.4f}, Broadleaved: {self.B_system[self.t]:.4f}")
        self.t += 1



    def get_vegetation_map(self) -> np.ndarray:
        aggregated = np.sum(self.proportions, axis=2)
        dominant = np.argmax(aggregated, axis=-1)
        raster = TYPES_TO_CORINE[dominant]
        return raster.astype(np.uint8)
