"""Descriptive summaries (Table-1 style) — pure pandas, no Shiny.

- continuous: N, Mean, SD, Median, Q1, Q3
- categorical: n (% of group); binary flags report only the "Yes" row
"""

from __future__ import annotations

import pandas as pd

from services.variables import LABELS, RCOUNT_LEVELS, VARIABLES

BINARY = {v.name for v in VARIABLES if v.kind == "binary"}


def _groups(df: pd.DataFrame, by: str | None) -> list[tuple[str, pd.DataFrame]]:
    """[(group label, sub-frame)] — a single 'Overall' group when `by` is None."""
    if not by:
        return [("Overall", df)]
    keys = sorted(df[by].unique(), key=lambda k: RCOUNT_LEVELS.index(k) if by == "rcount" else k)
    return [(str(k), df[df[by] == k]) for k in keys]


def continuous_summary(df: pd.DataFrame, variables: list[str], by: str | None = None) -> pd.DataFrame:
    """One row per (group, variable): N, Mean, SD, Median, Q1, Q3 (missing ignored)."""
    rows = []
    for group, sub in _groups(df, by):
        for var in variables:
            s = sub[var]
            rows.append({
                "Group": group, "Variable": LABELS[var], "N": int(s.count()),
                "Mean": s.mean(), "SD": s.std(), "Median": s.median(),
                "Q1": s.quantile(0.25), "Q3": s.quantile(0.75),
            })
    out = pd.DataFrame(rows)
    return out if by else out.drop(columns="Group")


def categorical_long(df: pd.DataFrame, variables: list[str], by: str | None = None) -> pd.DataFrame:
    """Long form: Group, Variable, Level, n, pct (percent of the group's rows)."""
    rows = []
    for group, sub in _groups(df, by):
        total = len(sub)
        for var in variables:
            counts = sub[var].value_counts()
            if var in BINARY:                       # flag -> single "Yes" row
                levels = [(1, "Yes")]
            else:
                order = RCOUNT_LEVELS if var == "rcount" else sorted(df[var].unique())
                levels = [(lv, str(lv)) for lv in order]
            for raw, label in levels:
                n = int(counts.get(raw, 0))
                rows.append({"Group": group, "Variable": LABELS[var], "Level": label,
                             "n": n, "pct": 100 * n / total if total else float("nan")})
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame, variables: list[str], by: str | None = None) -> pd.DataFrame:
    """Wide display table: Variable, Level, then one 'n (pct%)' column per group."""
    long = categorical_long(df, variables, by)
    long["cell"] = long.apply(lambda r: f"{r['n']:,} ({r['pct']:.1f}%)", axis=1)
    order = list(dict.fromkeys(zip(long["Variable"], long["Level"])))   # keep input order
    wide = long.pivot(index=["Variable", "Level"], columns="Group", values="cell")
    wide = wide.reindex(pd.MultiIndex.from_tuples(order, names=["Variable", "Level"]))
    wide = wide[[g for g, _ in _groups(df, by)]]                         # keep group order
    return wide.reset_index().rename_axis(columns=None)
