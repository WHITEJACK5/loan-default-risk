"""Runtime settings, loaded once from the environment and validated at import.

Deployment configuration is injected through environment variables so the same
image runs locally, in Docker and on Hugging Face without code changes. Values
are parsed by pydantic-settings rather than read with ``os.environ`` so a
mistyped variable fails fast at startup instead of surfacing as a wrong
threshold in production.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

# Columns resolved only after the loan is booked. Anything that becomes known
# post-decision leaks the label, so these are excluded from the feature set.
LEAKAGE_BLOCKLIST: tuple[str, ...] = (
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
)


class Settings(BaseSettings):
    """Environment-driven configuration for data, model and policy stages."""

    model_config = SettingsConfigDict(
        env_prefix="LDR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- data windows ---
    target: str = "loan_status"
    date_col: str = "issue_d"
    train_end: int = Field(default=2014, description="last inclusive training year")
    cal_year: int = Field(default=2015)
    policy_year: int = Field(default=2016)
    test_start: int = Field(default=2017)

    # --- policy economics ---
    threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    lgd: float = Field(default=0.6, gt=0.0, le=1.0, description="loss given default")
    op_cost: float = Field(default=500.0, ge=0.0, description="cupfront operating cost per loan")

    # --- reproducibility ---
    seed: int = Field(default=42, ge=0)
    calibration_ensemble: bool = Field(
        default=False,
        description=(
            "True keeps one estimator per calibration fold and averages them, "
            "costing 5.13x single-request latency for 0.0009 ROC; see "
            "scripts/calibration_tradeoff.py"
        ),
    )

    # --- artifacts ---
    artifacts_dir: Path = Field(default=REPO_ROOT / "artifacts")
    model_path: Path | None = Field(default=None, description="override LDR_MODEL_PATH")

    leakage_blocklist: tuple[str, ...] = LEAKAGE_BLOCKLIST
    model_params: dict[str, Any] = Field(default={"n_estimators": 500, "learning_rate": 0.05})

    @field_validator("model_params")
    @classmethod
    def _validate_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = set(value) - {"n_estimators", "learning_rate", "max_depth", "num_leaves"}
        if unknown:
            raise ValueError(f"model_params keys outside the LightGBM grid: {sorted(unknown)}")
        if int(value.get("n_estimators", 0)) <= 0 or float(value.get("learning_rate", 0)) <= 0:
            raise ValueError("model_params requires n_estimators > 0 and learning_rate > 0")
        return value

    @property
    def model_binary(self) -> Path:
        """Path to the model binary the API serves."""
        return self.model_path or self.artifacts_dir / "model.joblib"

    @property
    def model_meta(self) -> Path:
        return self.artifacts_dir / "model_meta.json"

    @property
    def metrics(self) -> Path:
        return self.artifacts_dir / "metrics.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton; call ``get_settings.cache_clear()`` in tests."""
    return Settings()


settings = get_settings()

__all__ = ["LEAKAGE_BLOCKLIST", "Settings", "get_settings", "settings"]
