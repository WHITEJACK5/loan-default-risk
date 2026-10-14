"""Hyperparameter search space and cross-validation scheme.

Both were module-level statements that printed on import, which made
``from src.loan_default_risk.tuning import ...`` write to stdout and made the
search space impossible to vary or test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.model_selection import TimeSeriesSplit

# Expanding-window splits on the 2007-2014 training block, per the work plan.
N_SPLITS = 3

# Grid logged as the search space. Bounded so a search rerun is reproducible.
SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [300, 500],
    "learning_rate": [0.05, 0.1],
    "max_depth": [6, 8],
}


@dataclass(frozen=True)
class TuningScheme:
    """Cross-validation scheme and the grid it is evaluated over."""

    n_splits: int = N_SPLITS
    search_space: dict[str, list[Any]] = field(default_factory=lambda: dict(SEARCH_SPACE))

    @property
    def splitter(self) -> TimeSeriesSplit:
        """Expanding-window splitter; a new instance each call, since sklearn
        splitters are stateful during ``split()``."""
        return TimeSeriesSplit(n_splits=self.n_splits)

    def combinations(self) -> int:
        """Number of parameter combinations in the grid."""
        total = 1
        for values in self.search_space.values():
            total *= len(values)
        return total


DEFAULT = TuningScheme()


def main() -> None:
    print(f"TimeSeriesSplit {DEFAULT.n_splits} expanding windows ready")
    print(f"grid {DEFAULT.search_space} -> {DEFAULT.combinations()} combinations")


if __name__ == "__main__":
    main()
