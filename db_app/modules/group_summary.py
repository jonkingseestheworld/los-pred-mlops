"""Cohort summary tab: descriptive tables for continuous and categorical variables.

Reads the pre-computed tables saved by `build_summary.py` (see `services/summary.py`);
nothing here summarises raw patient rows.
"""

from __future__ import annotations

from shiny import module, reactive, render, ui

from components.tables import html_table, table1
from services.charts import histogram_html
from services.summary import (
    categorical_from_tables,
    categorical_table1,
    continuous_display,
    continuous_from_tables,
    los_from_tables,
)
from services.summary_store import load_summary_meta, load_summary_tables, summary_is_stale
from services.variables import GROUP_BY_CHOICES, choices_of

# The variable pickers are switched off for now: every variable is shown, as if all were
# selected. To bring them back, uncomment the two `input_selectize` calls in the sidebar below
# and read `input.continuous_vars()` / `input.categorical_vars()` in the server instead of
# these two lists.
CONTINUOUS = choices_of("continuous")
CATEGORICAL = choices_of("categorical", "binary")


def _continuous_html(table):
    """Variable, Mean (SD), Median (Q1, Q3) as a plain table (all rows visible)."""
    shown = continuous_display(table)
    return html_table(list(shown.columns), shown.values.tolist())


@module.ui
def group_summary_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("by", "Group by", GROUP_BY_CHOICES, selected="none"),
            # ui.input_selectize(
            #     "continuous_vars", "Continuous variables", CONTINUOUS,
            #     selected=list(CONTINUOUS), multiple=True,
            # ),
            # ui.input_selectize(
            #     "categorical_vars", "Categorical variables", CATEGORICAL,
            #     selected=list(CATEGORICAL), multiple=True,
            # ),
            ui.hr(),
            ui.p(
                "Q1 / Q3 = 25th / 75th percentile. Categorical cells show "
                "n (% of the group); binary flags show the “Yes” row only.",
                class_="small text-muted",
            ),
            ui.output_ui("stamp"),
            width=320,
        ),
        ui.output_ui("stale_warning"),
        ui.card(
            ui.card_header("Length of stay distribution"),
            ui.output_ui("los_histogram"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Continuous variables"),
            ui.output_ui("continuous_tables"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Categorical variables"),
            ui.output_ui("categorical_table"),
            full_screen=True,
        ),
        ui.p(
            ui.tags.b("Left as recorded: "),
            "the outcome (length of stay, 1–17 days). The upper limit of 17 is a feature of this "
            "dataset rather than a clinical ceiling, so the model under-estimates the longest stays "
            "(see the model card).",
            class_="small text-muted mt-2",
        ),
    )


@module.server
def group_summary_server(input, output, session):
    tables = load_summary_tables()  # pre-computed, cached for the life of the process
    meta = load_summary_meta()

    @reactive.calc
    def group_by() -> str | None:
        return None if input.by() == "none" else input.by()

    @render.ui
    def stamp():
        return ui.p(
            f"Pre-computed from {meta['n_rows']:,} admissions "
            f"({meta['first_visit']} to {meta['last_visit']}). ",
            ui.tags.br(),
            f"Data fingerprint ", ui.tags.code(meta["data_sha256"][:8]),
            f", built {meta['built_at'][:10]}.",
            class_="small text-muted",
        )

    @render.ui
    def stale_warning():
        if summary_is_stale():
            return ui.div(
                "The source data has changed since these tables were built, so they may be out "
                "of date. Rebuild them with ",
                ui.tags.code("python db_app/build_summary.py"), ".",
                class_="alert alert-warning py-2",
            )

    @render.ui
    def los_histogram():
        # whole population: deliberately independent of the "Group by" selector
        return ui.HTML(histogram_html(los_from_tables(tables)))

    @render.ui
    def continuous_tables():
        by = group_by()
        table = continuous_from_tables(tables, list(CONTINUOUS), by)
        # one table per group (e.g. train, then test), each with its own heading and size;
        # ungrouped, a single "All entries" table
        if by is None:
            groups = [(GROUP_BY_CHOICES["none"], table)]
        else:
            groups = [(str(g).capitalize(), sub.drop(columns="Group")) for g, sub in table.groupby("Group", sort=False)]
        parts = []
        for i, (label, sub) in enumerate(groups):
            parts.append(ui.p(
                ui.tags.b(label),
                ui.span(f"n = {int(sub['N'].max()):,}", class_="text-muted ms-2"),
                class_="mb-1" + ("" if i == 0 else " mt-4"),
            ))
            parts.append(_continuous_html(sub))
        return ui.TagList(*parts)

    @render.ui
    def categorical_table():
        # the grouping variable itself would just show a 100% / 0% diagonal
        variables = [v for v in CATEGORICAL if v != group_by()]
        wide = categorical_from_tables(tables, variables, group_by())
        return table1(categorical_table1(wide))
