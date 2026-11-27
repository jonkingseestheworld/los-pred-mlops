"""Descriptive summaries (Table-1 style) — pure pandas, no Shiny.

- continuous: N, Mean, SD, Median, Q1, Q3
- categorical: n (% of group); binary flags report only the "Yes" row
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from services.variables import DISPLAY_LABELS, GROUP_BY_CHOICES, RCOUNT_LEVELS, VARIABLES, choices_of

BINARY = {v.name for v in VARIABLES if v.kind == "binary"}
LOS = "lengthofstay"
SPLIT_ORDER = ["train", "test"]


def _groups(df: pd.DataFrame, by: str | None) -> list[tuple[str, pd.DataFrame]]:
    """[(group label, sub-frame)] — a single 'Overall' group when `by` is None."""
    if not by:
        return [("Overall", df)]
    fixed = {"rcount": RCOUNT_LEVELS, "split": SPLIT_ORDER}.get(by)   # natural order, not alphabetical
    keys = sorted(df[by].unique(), key=lambda k: fixed.index(k) if fixed else k)
    return [(str(k), df[df[by] == k]) for k in keys]


def continuous_summary(df: pd.DataFrame, variables: list[str], by: str | None = None) -> pd.DataFrame:
    """One row per (group, variable): N, Mean, SD, Median, Q1, Q3 (missing ignored)."""
    rows = []
    for group, sub in _groups(df, by):
        for var in variables:
            s = sub[var]
            rows.append({
                "Group": group, "Variable": DISPLAY_LABELS[var], "N": int(s.count()),
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
                rows.append({"Group": group, "Variable": DISPLAY_LABELS[var], "Level": label,
                             "n": n, "pct": 100 * n / total if total else float("nan")})
    return pd.DataFrame(rows)


def _wide(long: pd.DataFrame) -> pd.DataFrame:
    """Long (Group, Variable, Level, n, pct) -> display table, one 'n (pct%)' column per group."""
    long = long.assign(cell=long.apply(lambda r: f"{r['n']:,} ({r['pct']:.1f}%)", axis=1))
    order = list(dict.fromkeys(zip(long["Variable"], long["Level"])))   # keep input order
    groups = list(dict.fromkeys(long["Group"]))                          # keep group order
    wide = long.pivot(index=["Variable", "Level"], columns="Group", values="cell")
    wide = wide.reindex(pd.MultiIndex.from_tuples(order, names=["Variable", "Level"]))
    return wide[groups].reset_index().rename_axis(columns=None)


def categorical_summary(df: pd.DataFrame, variables: list[str], by: str | None = None) -> pd.DataFrame:
    """Wide display table: Variable, Level, then one 'n (pct%)' column per group."""
    return _wide(categorical_long(df, variables, by))


# --- pre-computed tables ----------------------------------------------------
# The dashboard does not summarise the raw rows on every request: `build_summary_tables`
# is run once (offline, `db_app/build_summary.py`) for every grouping the app offers, the
# result is saved next to the app, and the `*_from_tables` functions below only filter and
# reshape those small tables. The cost per request no longer depends on the number of patients.
@dataclass(frozen=True)
class SummaryTables:
    continuous: pd.DataFrame     # By, Group, Variable, N, Mean, SD, Median, Q1, Q3
    categorical: pd.DataFrame    # By, Group, Variable, Level, n, pct
    los: pd.DataFrame            # Days, n  (all admissions per length-of-stay day; the histogram)


def los_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Admissions per length-of-stay day (whole population): Days, n.

    Length of stay is a whole number of days, so a count per day *is* the histogram. Every
    day from the minimum to the maximum is listed, zero counts included.
    """
    counts = df[LOS].value_counts()
    days = range(int(df[LOS].min()), int(df[LOS].max()) + 1)
    return pd.DataFrame({"Days": list(days), "n": [int(counts.get(d, 0)) for d in days]})


def build_summary_tables(
    df: pd.DataFrame,
    continuous: list[str] | None = None,
    categorical: list[str] | None = None,
    group_bys: list[str] | None = None,
) -> SummaryTables:
    """Summarise every variable for every grouping (key 'none' = overall)."""
    continuous = continuous or list(choices_of("continuous"))
    categorical = categorical or list(choices_of("categorical", "binary"))
    group_bys = group_bys or list(GROUP_BY_CHOICES)
    cont, cat = [], []
    for key in group_bys:
        by = None if key == "none" else key
        c = continuous_summary(df, continuous, by)
        if by is None:
            c.insert(0, "Group", "Overall")
        cont.append(c.assign(By=key))
        # a variable tabulated by itself would only show a 100% / 0% diagonal
        cat.append(categorical_long(df, [v for v in categorical if v != by], by).assign(By=key))

    def by_first(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[["By", *[c for c in frame.columns if c != "By"]]].reset_index(drop=True)

    return SummaryTables(by_first(pd.concat(cont)), by_first(pd.concat(cat)), los_counts(df))


def continuous_from_tables(t: SummaryTables, variables: list[str], by: str | None = None) -> pd.DataFrame:
    """Same result as `continuous_summary`, read from the pre-computed table."""
    labels = [DISPLAY_LABELS[v] for v in variables]
    sub = t.continuous[(t.continuous["By"] == (by or "none")) & t.continuous["Variable"].isin(labels)]
    groups = list(dict.fromkeys(sub["Group"]))
    sub = sub.assign(
        _g=pd.Categorical(sub["Group"], groups, ordered=True),
        _v=pd.Categorical(sub["Variable"], labels, ordered=True),
    )
    out = sub.sort_values(["_g", "_v"], kind="stable").drop(columns=["By", "_g", "_v"])
    out = out.reset_index(drop=True)
    return out if by else out.drop(columns="Group")


def continuous_display(table: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Display layout of a continuous summary: [Group,] Variable, Mean (SD), Median (Q1, Q3).

    The N column is left out for now (see the commented `"N"` below).
    """
    def fmt(x: float) -> str:
        return "–" if pd.isna(x) else f"{x:.{decimals}f}"

    out = table[[c for c in ("Group", "Variable") if c in table.columns]].copy()     #, "N"
    out["Mean (SD)"] = [f"{fmt(m)} ({fmt(s)})" for m, s in zip(table["Mean"], table["SD"])]
    out["Median (Q1, Q3)"] = [
        f"{fmt(m)} ({fmt(q1)}, {fmt(q3)})" for m, q1, q3 in zip(table["Median"], table["Q1"], table["Q3"])
    ]
    return out


def categorical_table1(wide: pd.DataFrame) -> pd.DataFrame:
    """Table-1 layout of a wide categorical summary (`categorical_summary` / `_from_tables`).

    One bold row per variable with its levels indented underneath; a binary flag (only a
    "Yes" level) is a single row carrying its count. Columns: Variable, kind, then one per
    group, where kind is "variable" (header / single row) or "level" (indented row).
    """
    groups = [c for c in wide.columns if c not in ("Variable", "Level")]
    rows = []
    for var, sub in wide.groupby("Variable", sort=False):
        if list(sub["Level"]) == ["Yes"]:
            rows.append({"Variable": var, "kind": "variable", **{g: sub.iloc[0][g] for g in groups}})
            continue
        rows.append({"Variable": var, "kind": "variable", **{g: "" for g in groups}})
        rows += [{"Variable": r["Level"], "kind": "level", **{g: r[g] for g in groups}}
                 for _, r in sub.iterrows()]
    return pd.DataFrame(rows, columns=["Variable", "kind", *groups])


def categorical_from_tables(t: SummaryTables, variables: list[str], by: str | None = None) -> pd.DataFrame:
    """Same result as `categorical_summary`, read from the pre-computed table."""
    labels = [DISPLAY_LABELS[v] for v in variables]
    sub = t.categorical[(t.categorical["By"] == (by or "none")) & t.categorical["Variable"].isin(labels)]
    sub = sub.assign(_v=pd.Categorical(sub["Variable"], labels, ordered=True))
    return _wide(sub.sort_values("_v", kind="stable").drop(columns=["By", "_v"]))


def los_from_tables(t: SummaryTables) -> pd.DataFrame:
    """The pre-computed admissions-per-day table (Days, n)."""
    return t.los.reset_index(drop=True)
