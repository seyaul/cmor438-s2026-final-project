"""Class-imbalance utilities for DataFrame-based datasets."""

from __future__ import annotations

import pandas as pd


def undersample_majority(
    df: pd.DataFrame,
    label_col: str = "class_label",
    majority_label: int = 0,
    cap_multiplier: float = 2.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Undersample the majority class to cap_multiplier × the largest minority class.

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
        Column containing integer class labels.
    majority_label : int
        The label to undersample (default 0, silence in GuitarSet).
    cap_multiplier : float
        How many times larger the majority class is allowed to be relative
        to the largest minority class (default 2.0).
    random_state : int or None

    Returns
    -------
    pd.DataFrame
        Shuffled DataFrame with majority class undersampled.
    """
    minority = df[df[label_col] != majority_label]
    majority = df[df[label_col] == majority_label]

    largest_minority = minority[label_col].value_counts().iloc[0]
    cap = int(cap_multiplier * largest_minority)

    if len(majority) <= cap:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    majority_sampled = majority.sample(n=cap, random_state=random_state)
    return (
        pd.concat([minority, majority_sampled])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
