"""Forecaster tab: placeholder for the predictor, plus the model's input variables."""

from __future__ import annotations

from shiny import module, render, ui

from components.placeholder import placeholder_card
from components.tables import html_table
from los_pred.features import FLAGS
from services.cleaning import CleaningReport, cleaning_report


def features_table(r: CleaningReport) -> ui.Tag:
    """The 28 columns the model is trained on, grouped by type."""
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


@module.ui
def forecaster_ui():
    return ui.div(
        placeholder_card(
            "Length-of-stay forecaster",
            "Enter a patient's details and get a predicted length of stay, with an "
            "explanation of what drove the prediction.",
        ),
        ui.card(ui.card_header("Final model inputs"), ui.output_ui("features")),
    )


@module.server
def forecaster_server(input, output, session):
    report = cleaning_report()  # cached; the same audit the Data cleaning tab uses

    @render.ui
    def features():
        return features_table(report)
