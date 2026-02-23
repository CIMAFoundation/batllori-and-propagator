from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numba import njit


INVALID_VALUES = (-9999, -3333, -6666)


@dataclass(frozen=True)
class ModelParams:
    k_sy_sm: float = 0.1
    k_ry_rm: float = 0.1
    k_au: float = 0.03
    rho_s: float = 0.05
    rho_sy: float = 0.0125
    rho_r: float = 0.015
    rho_rm: float = 0.0125
    fraction: float = 0.75
    w_ry: float = 0.3
    w_rm: float = 0.15
    w_sy: float = 0.4
    w_sm: float = 0.25
    w_u: float = 0.1
    omega_l: float = 0.7

    @property
    def omega_cell(self) -> float:
        return 1.0 - self.omega_l


class Batllori6CL:
    """Encapsulates the vegetation–fire dynamics so callers can reuse it without I/O."""

    def __init__(self, initial_map: np.ndarray, params: ModelParams | None = None) -> None:
        self.params = params or ModelParams()
        self.proportions = np.asarray(initial_map, dtype=float).copy()
        if self.proportions.ndim != 3 or self.proportions.shape[2] != 6:
            raise ValueError("initial_map must have shape (rows, cols, 6)")

        self.grid_size = self.proportions.shape[0]
        if self.proportions.shape[0] != self.proportions.shape[1]:
            raise ValueError("initial_map must be square")

        self.tsf = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.total_cells = self.grid_size * self.grid_size

        # Pre-compute coefficients that only depend on parameters.
        self.mu_s = self.params.rho_s * self.params.fraction
        self.mu_sy = self.params.rho_sy * self.params.fraction
        self.mu_r = self.params.rho_r * self.params.fraction
        self.mu_rm = self.params.rho_rm * self.params.fraction

    def step(self, fire_mask: np.ndarray|None = None) -> Dict[str, np.ndarray | float | int]:
        """Advance the model by one timestep using a boolean fire mask."""
        if fire_mask is None:
            fire_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        mask = np.ascontiguousarray(np.asarray(fire_mask, dtype=bool))
        if mask.shape != (self.grid_size, self.grid_size):
            raise ValueError("fire_mask must match the model grid size")

        totals, fires_this_step = _step_kernel(
            self.proportions,
            self.tsf,
            mask,
            self.params.k_sy_sm,
            self.params.k_ry_rm,
            self.params.k_au,
            self.params.rho_s,
            self.params.rho_sy,
            self.params.rho_r,
            self.params.rho_rm,
            self.params.w_ry,
            self.params.w_rm,
            self.params.w_sy,
            self.params.w_sm,
            self.params.w_u,
            self.params.omega_l,
            self.params.omega_cell,
            self.mu_s,
            self.mu_sy,
            self.mu_r,
            self.mu_rm,
        )

        averages = totals / self.total_cells
        return {
            "totals": totals,
            "averages": averages,
            "fires_this_step": fires_this_step
        }

    def update_vegetation_map(self, new_map: np.ndarray, reset_tsf: bool = False) -> None:
        """Replace the full vegetation map, optionally resetting TSF counters."""
        new_map = np.asarray(new_map, dtype=float)
        if new_map.shape != self.proportions.shape:
            raise ValueError("new_map must match the current map shape")
        self.proportions = new_map.copy()
        if reset_tsf:
            self.tsf.fill(0)

    def update_cell(self, row: int, col: int, new_values: np.ndarray, reset_tsf: bool = False) -> None:
        """Update a single cell with custom proportions."""
        new_values = np.asarray(new_values, dtype=float)
        if new_values.shape != (6,):
            raise ValueError("new_values must have shape (6,)")
        self.proportions[row, col] = new_values
        if reset_tsf:
            self.tsf[row, col] = 0

    def get_vegetation_map(self, copy: bool = True) -> np.ndarray:
        """Return the current vegetation proportions."""
        return self.proportions.copy() if copy else self.proportions

    def encode_grid(self) -> np.ndarray:
        """Return the encoded integer grid used for exports."""
        encoded = np.full((self.grid_size, self.grid_size), -9999, dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.proportions[i, j]
                if np.all(cell == -9999):
                    encoded[i, j] = -9999
                elif np.all(cell == -3333):
                    encoded[i, j] = 3
                else:
                    s23 = cell[2] + cell[3]
                    s45 = cell[4] + cell[5]
                    candidates: Dict[float, int] = {
                        cell[0]: 4,
                        cell[1]: 2,
                        s23: 5,
                        s45: 1,
                    }
                    encoded[i, j] = candidates[max(candidates)]
        return encoded

    def grid_shape(self) -> Tuple[int, int]:
        return self.proportions.shape[:2]


@njit(cache=True)
def clip(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value

@njit(cache=True)
def _step_kernel(
    proportions,
    tsf,
    fire_mask,
    k_sy_sm,
    k_ry_rm,
    k_au,
    rho_s,
    rho_sy,
    rho_r,
    rho_rm,
    w_ry,
    w_rm,
    w_sy,
    w_sm,
    w_u,
    omega_l,
    omega_cell,
    mu_s,
    mu_sy,
    mu_r,
    mu_rm,
):
    grid_size = proportions.shape[0]

    sm_sum = 0.0
    sm_count = 0
    rm_sum = 0.0
    rm_count = 0

    for i in range(grid_size):
        for j in range(grid_size):
            sm_val = proportions[i, j, 3]
            if sm_val != -9999.0 and sm_val != -3333.0 and sm_val != -6666.0:
                sm_sum += sm_val
                sm_count += 1

            rm_val = proportions[i, j, 5]
            if rm_val != -9999.0 and rm_val != -3333.0 and rm_val != -6666.0:
                rm_sum += rm_val
                rm_count += 1

    Sm_mean = 0.0
    if sm_count > 0:
        Sm_mean = sm_sum / sm_count

    Rm_mean = 0.0
    if rm_count > 0:
        Rm_mean = rm_sum / rm_count

    totals = np.zeros(6, dtype=np.float64)
    fires_this_step = 0

    for i in range(grid_size):
        for j in range(grid_size):
            A_old = proportions[i, j, 0]

            if A_old == -9999.0 or A_old == -3333.0 or A_old == -6666.0:
                continue

            U_old = proportions[i, j, 1]
            Sy_old = proportions[i, j, 2]
            Sm_old = proportions[i, j, 3]
            Ry_old = proportions[i, j, 4]
            Rm_old = proportions[i, j, 5]

            tsf[i, j] += 1
            tsf_val = tsf[i, j]

            if not fire_mask[i, j]:
                F_s = omega_cell * Sm_old + omega_l * Sm_mean
                F_r = omega_cell * Rm_old + omega_l * Rm_mean

                K_u_sy = mu_s + (rho_s - mu_s) * F_s
                K_ry_sy = mu_sy + (rho_sy - mu_sy) * F_s
                K_u_ry = mu_r + (rho_r - mu_r) * F_r
                K_sm_rm = mu_rm + (rho_rm - mu_rm) * F_s

                A = (1 - k_au) * A_old
                U = (1 - K_u_sy - K_u_ry) * U_old + k_au * A_old
                Sy = (1 - k_sy_sm) * Sy_old + K_u_sy * U_old + K_ry_sy * Ry_old
                Sm = (1 - K_sm_rm) * Sm_old + k_sy_sm * Sy_old
                Ry = (1 - K_ry_sy - k_ry_rm) * Ry_old + K_u_ry * U_old
                Rm = Rm_old + k_ry_rm * Ry_old + K_sm_rm * Sm_old
            else:
                fires_this_step += 1

                power_val = 0.35 ** (3.367 - 0.306 * (tsf_val - 1.0))
                if power_val < 1.0:
                    P_ry = power_val
                else:
                    P_ry = 1.0

                frac_val = tsf_val / 5.0
                if frac_val < 1.0:
                    P_rm = frac_val
                else:
                    P_rm = 1.0

                tmp_sy = (1 - w_sy) * Sy_old
                if tmp_sy < Sm_old:
                    min_sy_sm = tmp_sy
                else:
                    min_sy_sm = Sm_old

                G = (1 - w_sm) * Sm_old + min_sy_sm
                T = P_rm * (1 - w_rm) * Rm_old + P_ry * (1 - w_ry) * Ry_old

                C_g = w_sm * Sm_old + (Sy_old - min_sy_sm)

                term_rm = (1 - w_rm) * P_rm
                term_ry = (1 - w_ry) * P_ry
                C_t = (1 - term_rm) * Rm_old + (1 - term_ry) * Ry_old

                C_u = w_u * U_old
                C_u_half = C_u / 2.0
                if C_u_half < Sm_old:
                    C_u_sy = C_u_half
                else:
                    C_u_sy = Sm_old
                C_u_a = C_u - C_u_sy

                A = A_old + C_u_a
                U = U_old - C_u_sy - C_u_a + C_g + C_t
                Sy = G + C_u_sy
                Sm = 0.0
                Ry = T
                Rm = 0.0

                tsf[i, j] = 0

            proportions[i, j, 0] = clip(A, 0, 1)
            proportions[i, j, 1] = clip(U, 0, 1)
            proportions[i, j, 2] = clip(Sy, 0, 1)
            proportions[i, j, 3] = clip(Sm, 0, 1)
            proportions[i, j, 4] = clip(Ry, 0, 1)
            proportions[i, j, 5] = clip(Rm, 0, 1)

            totals[0] += A
            totals[1] += U
            totals[2] += Sy
            totals[3] += Sm
            totals[4] += Ry
            totals[5] += Rm

    return totals, fires_this_step
