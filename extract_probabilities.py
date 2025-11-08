#%%
import pandas as pd
import rasterio as rio
import os
import numpy as np
import matplotlib.pyplot as plt

MAX_DURATION = 12


df = pd.read_csv('data/extreme_events.csv', parse_dates=['start', 'end'])
MAX_DAILY_DURATION_DF = pd.DataFrame({
    'duration_h': df.groupby(df['start'].dt.date).duration_h.max()
}).reset_index()

with rio.open('data/mask.tif') as src:
    mask = src.read(1) > 0

with rio.open('data/clc_2018.tif') as src:
    veg = src.read(1)

with rio.open('data/susc_monti_pisani.tif') as src:
    susc = src.read(1)

#%%
gen = np.random.Generator(np.random.PCG64(42))

def extract_ignition_points(n_events: int) -> list[tuple[int, int]]:
    # sample a random ignition point within the mask according to the susc map
    ignition_points = []
    for _ in range(n_events):
        susc_masked = np.where(mask, susc, 0)
        susc_flat = susc_masked.flatten() ** 4  # emphasize high susc areas
        susc_flat = susc_flat / susc_flat.sum()  # normalize to sum to 1

        point = gen.choice(np.arange(susc_flat.size), p=susc_flat)
        point = np.unravel_index(point, susc_masked.shape)
        ignition_points.append((int(point[0]), int(point[1])))
    print(ignition_points)
    return ignition_points


def sample_event_durations(random_state: int | None = None) -> list[int]:
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
    rng = np.random.default_rng(random_state)

    # estrai anno da start
    df = MAX_DAILY_DURATION_DF.copy()
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

