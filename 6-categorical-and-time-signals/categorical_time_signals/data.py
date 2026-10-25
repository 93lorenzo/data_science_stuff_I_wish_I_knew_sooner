from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

NYC_TAXI_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"

PAYMENT_TYPE_LABELS = {
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
}

DISTANCE_TIER_BINS = [0, 1, 2, 4, 8, np.inf]
DISTANCE_TIER_LABELS = ["0-1 mi", "1-2 mi", "2-4 mi", "4-8 mi", "8+ mi"]


def download_trip_data(url: str = NYC_TAXI_URL) -> pd.DataFrame:
    """Download one month of NYC yellow-taxi trip records.

    Public data hosted by the NYC Taxi & Limousine Commission, no
    authentication required (~50 MB Parquet file).
    """
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw_bytes = response.read()
    return pd.read_parquet(io.BytesIO(raw_bytes))


def prepare_trip_data(
    raw_df: pd.DataFrame,
    sample_size: int = 20_000,
    seed: int = 7,
) -> pd.DataFrame:
    """Clean, label, and sample the raw trip data down to a notebook-friendly size.

    Every column used below is real, nothing here is a synthetic label. The
    only randomness is which rows land in the sample, controlled by
    ``seed`` so the notebook is fully reproducible despite starting from a
    ~3 million row file.

    * ``well_tipped``: 1 if the tip was at least 20% of the fare. Only
      credit-card payments reliably record a tip at all, cash tips are
      essentially never logged by the metering system, so this label is
      most meaningful read alongside ``payment_type``, not despite it.
    * ``distance_tier``: an ordinal bucketing of ``trip_distance`` into 5
      ordered bands, short trips first.
    * ``hour``: the pickup hour, 0-23.
    * ``airport_fee_flag``: 1 if ``Airport_fee`` is recorded as anything
      other than NaN or 0 (a nonzero airport fee was actually charged), 0
      otherwise.
    * ``store_and_fwd_flag_bin``: 1 if ``store_and_fwd_flag == "Y"``, 0 for
      everything else (``"N"`` or missing). Y (Store and forward trip): the
      trip record was held in vehicle memory because the taxi lacked a live
      connection to the server at the time, and was sent out later once
      connectivity was restored.
    """
    df = raw_df[
        (raw_df["fare_amount"] > 0)
        & (raw_df["trip_distance"] > 0)
        & (raw_df["trip_distance"] < 100)
        & (raw_df["passenger_count"].between(1, 6))
        & (raw_df["payment_type"].isin(PAYMENT_TYPE_LABELS))
    ].copy()

    df["payment_type"] = df["payment_type"].map(PAYMENT_TYPE_LABELS)
    df["tip_pct"] = df["tip_amount"] / df["fare_amount"]
    df["well_tipped"] = (df["tip_pct"] >= 0.20).astype(int)
    df["distance_tier"] = pd.cut(
        df["trip_distance"], bins=DISTANCE_TIER_BINS, labels=DISTANCE_TIER_LABELS
    )
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["airport_fee_flag"] = (
        df["Airport_fee"].notna() & (df["Airport_fee"] != 0)
    ).astype(int)
    df["store_and_fwd_flag_bin"] = (df["store_and_fwd_flag"] == "Y").astype(int)

    columns = [
        "tpep_pickup_datetime",
        "hour",
        "payment_type",
        "distance_tier",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "tip_pct",
        "passenger_count",
        "well_tipped",
        "airport_fee_flag",
        "store_and_fwd_flag_bin",
    ]
    return df[columns].sample(n=sample_size, random_state=seed).reset_index(drop=True)


def add_cyclical_hour_features(
    df: pd.DataFrame, hour_col: str = "hour", period: int = 24
) -> pd.DataFrame:
    """Add sine/cosine coordinates for an hour-of-day (or any cyclical) column."""
    radians = 2 * np.pi * df[hour_col] / period
    return df.assign(
        **{f"{hour_col}_sin": np.sin(radians), f"{hour_col}_cos": np.cos(radians)}
    )
