"""Description > Data cleaning: what was done to the raw data before modelling, and why.

Everything shown is measured from the real data by `services.cleaning`, which runs
the same `los_pred.features` pipeline the model was trained with.
"""

from __future__ import annotations

import pandas as pd
from shiny import module, render, ui

from components.tables import html_table
from los_pred.features import FLAGS
from services.cleaning import CleaningReport, cleaning_report

ALL_ROWS = "All rows"
COLUMN_REMOVED = "Column removed"


## [PACEHOLDER]       
 # ui.p(
 #     ui.tags.b("Tip for modelling: "),
 #     "drop the visit and discharge dates — they give the answer away "
 #     "(discharge − visit = length of stay).",
 # ),

def _n_pct(n: int, total: int) -> str:
    return f"{n:,} ({n / total:.1%})"


def _code(text: str) -> ui.Tag:
    return ui.tags.code(text)


def _month(period: str) -> str:
    return pd.Period(period, freq="M").strftime("%b %Y")


# --- tables built from the report ------------------------------------------
def steps_table(r: CleaningReport) -> ui.Tag:
    eff = r.effect.set_index("Variable")
    p, total = r.params, r.n_rows
    rows = [
        ["Duplicates & missing values", "all columns",
         f"{r.n_duplicates} duplicate rows (every episode ID is unique); {r.n_missing} missing values",
         "None needed", "0"],
        ["Leakage check", [_code("eid"), ", ", _code("vdate"), ", ", _code("discharged")],
         f"Discharge date − visit date equals length of stay in {r.leakage_match_pct:.1f}% of rows "
         "— the answer is in the data",
         ["Excluded from the model inputs. ", _code("vdate"),
          " is kept only to build the monthly train/test batches"],
         COLUMN_REMOVED],
        ["Monthly batches", _code("vdate"),
         "A single boundary day (1 Jan 2013) sits outside the 12 monthly batches",
         "Relabelled into the December 2012 batch (the raw date is untouched)",
         _n_pct(r.n_stub_rows, total)],
        ["Impossible value", _code("glucose"),
         f"Negative glucose is physiologically impossible (min = {eff.loc['glucose', 'Raw min']:.2f})",
         [f"Values ≤ 0 replaced by the training-set median ({p['glucose_median']:.1f}). "
          "High glucose is clinically real, so it is not capped"],
         _n_pct(r.n_glucose_fixed, total)],
        ["Extreme outliers", _code("neutrophils"),
         f"Implausible right tail (max = {eff.loc['neutrophils', 'Raw max']:.1f})",
         f"Capped (winsorised) at the training-set 99th percentile ({p['neutrophils_cap']:.1f}); rows kept",
         _n_pct(r.n_neutrophils_capped, total)],
        ["Extreme outliers", _code("bloodureanitro"),
         f"Implausible right tail (max = {eff.loc['bloodureanitro', 'Raw max']:.1f})",
         f"Capped (winsorised) at the training-set 99th percentile ({p['bloodureanitro_cap']:.1f}); rows kept",
         _n_pct(r.n_bun_capped, total)],
        ["Near-constant variable", _code("respiration"),
         f"Almost no variation (interquartile range = {r.respiration_iqr:.1f}, median {r.respiration_median:.1f})",
         "Dropped — negligible signal", COLUMN_REMOVED],
        ["No signal", _code("secondarydiagnosisnonicd9"),
         f"No relationship with length of stay (correlation r = {r.secondary_dx_corr:.2f})",
         "Dropped", COLUMN_REMOVED],
        ["Recoding", _code("gender"), "Text category (M / F)",
         ["Recoded to a single 0/1 column, ", _code("isMale")], ALL_ROWS],
        ["Recoding", _code("rcount"), "Numeric count stored as text, with a “5+” level",
         "“5+” recoded to 5 so the column is numeric", ALL_ROWS],
        ["Recoding", _code("facid"), "Facility ID is a label, not a quantity",
         ["One-hot encoded over the training facilities (" + ", ".join(p["facid_categories"]) + ") plus an ",
          _code("other"), " column for any facility not seen in training"],
         ALL_ROWS],
        ["New variable", _code("comorbidity_count"), "11 separate 0/1 comorbidity flags",
         "Sum of the flags (0–11) added as one column; the individual flags are kept", ALL_ROWS],
    ]
    numbered = [[str(i), *row] for i, row in enumerate(rows, start=1)]
    return html_table(
        ["#", "Step", "Variable(s)", "What was found", "Treatment", "Rows affected"], numbered,
        numeric_cols=[],
    )


def effect_table(r: CleaningReport) -> ui.Tag:
    rows = [
        [_code(e["Variable"]), f"{e['Raw min']:.2f}", f"{e['Raw max']:.2f}",
         f"{e['Cleaned min']:.2f}", f"{e['Cleaned max']:.2f}", f"{e['Rows changed']:,}"]
        for e in r.effect.to_dict("records")
    ]
    return html_table(
        ["Variable", "Raw min", "Raw max", "Cleaned min", "Cleaned max", "Rows changed"],
        rows, numeric_cols=[1, 2, 3, 4, 5],
    )


def split_table(r: CleaningReport) -> ui.Tag:
    rows = [
        [s["split"].capitalize(), f"{_month(s['first'])} – {_month(s['last'])}",
         f"{s['rows']:,}", f"{s['share']:.1%}"]
        for s in r.split.to_dict("records")
    ]
    return html_table(["Set", "Admission months", "Rows", "Share"], rows, numeric_cols=[2, 3])


def params_table(r: CleaningReport) -> ui.Tag:
    p = r.params
    rows = [
        ["Facility list", ", ".join(p["facid_categories"]), "One-hot encoding of the facility"],
        ["Glucose median", f"{p['glucose_median']:.2f}", "Replaces impossible glucose values"],
        ["Neutrophils cap", f"{p['neutrophils_cap']:.2f}", "Upper limit for neutrophils"],
        ["Blood urea nitrogen cap", f"{p['bloodureanitro_cap']:.2f}", "Upper limit for blood urea nitrogen"],
    ]
    return html_table(["Learned from training data", "Value", "Used for"], rows, numeric_cols=[1])


def features_table(r: CleaningReport) -> ui.Tag:
    flags = set(FLAGS)
    groups = {
        "Admission & demographics": [c for c in r.features if c in ("rcount", "isMale", "comorbidity_count")],
        "Comorbidity flags": [c for c in r.features if c in flags],
        "Labs & vitals": [c for c in r.features if c in
                          ("hematocrit", "neutrophils", "sodium", "glucose",
                           "bloodureanitro", "creatinine", "bmi", "pulse")],
        "Facility (one-hot)": [c for c in r.features if c.startswith("fac_")],
    }
    rows = [
        [name, str(len(cols)),
         ui.TagList(*[ui.span(c, class_="badge text-bg-light border me-1 mb-1 font-monospace") for c in cols])]
        for name, cols in groups.items()
    ]
    return html_table(["Group", "Columns", "Variables"], rows, numeric_cols=[1])


# --- module ----------------------------------------------------------------
@module.ui
def data_cleaning_ui():
    return ui.div(
        ui.card(
            ui.card_header("Approach"),
            ui.tags.ul(
                ui.tags.li(ui.tags.b("Split first. "),
                           "Data are split by time before anything is learned (earlier months for "
                           "training, later months for testing), so the test set stays unseen."),
                ui.tags.li(ui.tags.b("Learn on training data only. "),
                           "Medians, caps and the facility list are fitted on the training rows, then "
                           "applied unchanged to the test set and to new patients — no test information leaks in."),
                ui.tags.li(ui.tags.b("Fix, don't delete. "),
                           "Impossible values are replaced and extreme values capped; no rows are removed."),
                ui.tags.li(ui.tags.b("One code path. "),
                           "The same cleaning function is used for training and for predictions, "
                           "so the model never sees differently-prepared data."),
            ),
        ),
        ui.layout_columns(
            ui.value_box("Rows checked", ui.output_text("n_rows")),
            ui.value_box("Duplicate rows", ui.output_text("n_dups")),
            ui.value_box("Missing values", ui.output_text("n_missing")),
            ui.value_box("Values corrected", ui.output_text("n_corrected")),
            ui.value_box("Rows removed", ui.output_text("n_removed")),
            fill=False,
        ),
        ui.card(
            ui.card_header("Cleaning steps, in the order they are applied"),
            ui.output_ui("steps"),
        ),
        #ui.card(ui.card_header("Effect on the corrected variables"), ui.output_ui("effect")),
        ui.layout_columns(
            ui.card(ui.card_header("Train / test split"), ui.output_ui("split")),
            ui.card(ui.card_header("Parameters learned from the training data"), ui.output_ui("params")),
        ),
        ui.card(ui.card_header("Final model inputs"), ui.output_ui("features")),
        ui.p(
            ui.tags.b("Left as recorded: "),
            "the outcome (length of stay, 1–17 days). The upper limit of 17 is a feature of this "
            "dataset rather than a clinical ceiling, so the model under-estimates the longest stays "
            "(see the model card).",
            class_="small text-muted mt-2",
        ),
    )


@module.server
def data_cleaning_server(input, output, session):
    report = cleaning_report()  # computed once per process

    @render.text
    def n_rows():
        return f"{report.n_rows:,}"

    @render.text
    def n_dups():
        return f"{report.n_duplicates:,}"

    @render.text
    def n_missing():
        return f"{report.n_missing:,}"

    @render.text
    def n_corrected():
        return f"{report.n_glucose_fixed + report.n_neutrophils_capped + report.n_bun_capped:,}"

    @render.text
    def n_removed():
        return f"{report.n_removed:,}"

    @render.ui
    def steps():
        return steps_table(report)

    #@render.ui
    #def effect():
    #    return effect_table(report)

    @render.ui
    def split():
        return split_table(report)

    @render.ui
    def params():
        return params_table(report)

    @render.ui
    def features():
        return features_table(report)
