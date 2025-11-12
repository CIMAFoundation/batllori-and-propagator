from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio

MAX_DURATION = 12
DATA_DIR = Path("data")
EXTREME_EVENTS_PATH = DATA_DIR / "extreme_events.csv"
MASK_PATH = DATA_DIR / "mask.tif"
SUSC_PATH = DATA_DIR / "susc_monti_pisani.tif"


@dataclass(frozen=True)
class ProbabilityInputs:
    max_daily_duration_df: pd.DataFrame
    mask: np.ndarray
    susc: np.ndarray


@lru_cache(maxsize=1)
def load_probability_inputs() -> ProbabilityInputs:
    df = pd.read_csv(EXTREME_EVENTS_PATH, parse_dates=["start", "end"])
    max_daily_duration_df = pd.DataFrame(
        {"duration_h": df.groupby(df["start"].dt.date).duration_h.max()}
    ).reset_index()

    with rio.open(MASK_PATH) as src:
        mask = src.read(1) > 0

    with rio.open(SUSC_PATH) as src:
        susc = src.read(1)

    return ProbabilityInputs(
        max_daily_duration_df=max_daily_duration_df,
        mask=mask,
        susc=susc,
    )

def extract_ignition_points(
    n_events: int,
    rng: np.random.Generator | None = None,
    data: ProbabilityInputs | None = None,
) -> list[tuple[int, int]]:
    """Sample ignition coordinates using the susceptibility map as weights."""
    rng = rng or np.random.default_rng()
    data = data or load_probability_inputs()

    susc_masked = np.where(data.mask, data.susc, 0)
    susc_flat = susc_masked.flatten() ** 4  # emphasize high susc areas
    susc_prob = susc_flat / susc_flat.sum()

    ignition_points = []
    for _ in range(n_events):
        point = rng.choice(np.arange(susc_prob.size), p=susc_prob)
        point = np.unravel_index(point, susc_masked.shape)
        ignition_points.append((int(point[0]), int(point[1])))
    return ignition_points


def sample_event_durations(
    rng: np.random.Generator | None = None,
    data: ProbabilityInputs | None = None,
) -> list[int]:
    """
    Ritorna una lista di durate (in ore) simulata come un anno tipico del dataset.
    Il numero di eventi è campionato dalla distribuzione dei conteggi annuali.
    Le durate sono campionate dalla distribuzione osservata globale.

    Parametri
    ----------
    df : DataFrame con colonne ['start', 'duration_h']
    random_state : opzionale, per riproducibilità

    Ritorna
    -------
    list[int] : durate simulate in ore
    """
    rng = rng or np.random.default_rng()
    data = data or load_probability_inputs()

    # estrai anno da start
    df = data.max_daily_duration_df.copy()
    df["year"] = pd.to_datetime(df["start"]).dt.year

    # distribuzione dei conteggi annuali
    event_counts = df.groupby("year").size()
    n_events = int(rng.choice(event_counts))

    # distribuzione globale delle durate
    durations = df["duration_h"].to_numpy()
    sampled_durations = rng.choice(durations, size=n_events, replace=True)

    # create a probability based on duration (duration/MAX_DURATION)
    probabilities = (sampled_durations / MAX_DURATION)**2
    # resample based on probabilities
    mask = rng.random(size=n_events) < probabilities
    sampled_durations = sampled_durations[mask]

    return sampled_durations.tolist()
