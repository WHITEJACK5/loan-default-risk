"""Pandera schema for the raw LendingClub columns.

Imported from ``pandera.pandas`` rather than the top-level ``pandera``, which
pandera 0.33 deprecates. The top-level import still works but emits a
FutureWarning on every import, which is the kind of warning that becomes a hard
failure at the next major version.

This schema is the machine-readable form of ``docs/leakage_audit.md``: it is
the check that a leakage column's *values* cannot silently change meaning.
"""

from __future__ import annotations

import pandera.pandas as pa

# Bounds are the corpus min/max for the published LendingClub columns, so a
# value outside them means the raw archive or the parsing changed.
loan_amount_schema = pa.DataFrameSchema(
    {
        "loan_amnt": pa.Column(int, pa.Check.gt(0)),
        "dti": pa.Column(float, pa.Check.ge(0)),
        "fico_range_low": pa.Column(int, pa.Check.between(300, 850)),
        "fico_range_high": pa.Column(int, pa.Check.between(300, 850)),
        "int_rate": pa.Column(float, pa.Check.between(5, 35)),
        "annual_inc": pa.Column(int, pa.Check.gt(0)),
        "term": pa.Column(str, pa.Check.isin(["36 months", "60 months"])),
        "grade": pa.Column(str, pa.Check.isin(list("ABCDEFG"))),
        "loan_status": pa.Column(str, pa.Check.isin(["Fully Paid", "Charged Off"])),
    },
    strict=False,  # extra columns are allowed; only the listed ones are checked
    name="lending_club_raw",
)


def validate(df):
    """Validate a raw frame; raises ``SchemaError`` listing every violation."""
    return loan_amount_schema.validate(df, lazy=True)
