"""Data-split and monotonicity properties that must hold for the model to be trustworthy.

The ordering checks are the ones that catch a real defect: a shuffled split or a
preprocessing bug would surface here as a comparison failure, not as a silent
metric degradation.
"""

from __future__ import annotations

import pytest

from src.loan_default_risk.config import settings

pytestmark = pytest.mark.usefixtures("api")


@pytest.fixture(scope="module")
def processed():
    """Skip data-dependent tests when the parquet splits are absent (e.g. CI)."""
    from pathlib import Path

    import pandas as pd

    paths = [Path(f"data/processed/{split}.parquet") for split in ("train", "val", "test")]
    if not all(path.is_file() for path in paths):
        pytest.skip("data/processed splits not built; run src.data.make_dataset")
    return {split: pd.read_parquet(path) for split, path in zip(("train", "val", "test"), paths)}


def test_splits_are_temporally_ordered(processed):
    train, val, test = processed["train"], processed["val"], processed["test"]
    assert train["issue_year"].max() < val["issue_year"].min()
    assert val["issue_year"].max() < test["issue_year"].min()


def test_splits_partition_the_resolved_corpus(processed):
    """Every split is non-empty and the years do not overlap."""
    years = {split: set(processed[split]["issue_year"].unique()) for split in processed}
    for split, seen in years.items():
        assert seen, f"{split} is empty"
    assert years["train"] & years["val"] == set()
    assert years["val"] & years["test"] == set()


def test_default_rate_is_stationary_across_splits(processed):
    """The default rate must not collapse across time; a drop means the
    resolution filter changed or the label construction is wrong."""
    rates = {split: processed[split]["target"].mean() for split in processed}
    assert all(0.0 < rate < 1.0 for rate in rates.values()), rates
    assert max(rates.values()) - min(rates.values()) < 0.15, rates


def test_no_leakage_columns_reach_the_splits(processed):
    columns = set().union(*(set(frame.columns) for frame in processed.values()))
    leaked = set(settings.leakage_blocklist).intersection(columns)
    assert not leaked, f"leakage columns present in processed splits: {sorted(leaked)}"


def test_scoring_is_monotone_in_dti(api, sample_payload):
    """DTI is the synthetic fixture's monotone risk driver, so the fixture model
    must rank it in order. This is the property that a broken preprocessing
    step or a shuffled feature matrix would violate."""
    low = api.post("/predict", json={**sample_payload, "dti": 5}).json()
    mid = api.post("/predict", json={**sample_payload, "dti": 18}).json()
    high = api.post("/predict", json={**sample_payload, "dti": 35}).json()
    assert low["pd"] < mid["pd"] < high["pd"], (low["pd"], mid["pd"], high["pd"])
