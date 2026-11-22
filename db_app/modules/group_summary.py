"""Group summary tab: descriptive tables for continuous and categorical variables."""

from __future__ import annotations

import pandas as pd
from shiny import module, reactive, render, req, ui

from services.summary import categorical_summary, continuous_summary
from services.variables import GROUP_BY_CHOICES, choices_of


@module.ui
def group_summary_ui():
    continuous = choices_of("continuous")
    categorical = choices_of("categorical", "binary")
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("by", "Group by", GROUP_BY_CHOICES, selected="none"),
            ui.input_selectize(
                "continuous_vars", "Continuous variables", continuous,
                selected=list(continuous), multiple=True,
            ),
            ui.input_selectize(
                "categorical_vars", "Categorical variables", categorical,
                selected=list(categorical), multiple=True,
            ),
            ui.hr(),
            ui.p(
                "Q1 / Q3 = 25th / 75th percentile. Categorical cells show "
                "n (% of the group); binary flags show the “Yes” row only.",
                class_="small text-muted",
            ),
            width=320,
        ),
        ui.card(
            ui.card_header("Continuous variables"),
            ui.output_data_frame("continuous_table"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Categorical variables"),
            ui.output_data_frame("categorical_table"),
            full_screen=True,
        ),
    )


@module.server
def group_summary_server(input, output, session, data: pd.DataFrame):
    @reactive.calc
    def group_by() -> str | None:
        return None if input.by() == "none" else input.by()

    @render.data_frame
    def continuous_table():
        req(input.continuous_vars())
        table = continuous_summary(data, list(input.continuous_vars()), group_by())
        return render.DataGrid(table.round(2), width="100%")

    @render.data_frame
    def categorical_table():
        # the grouping variable itself would just show a 100% / 0% diagonal
        variables = [v for v in input.categorical_vars() if v != group_by()]
        req(variables)
        table = categorical_summary(data, variables, group_by())
        return render.DataGrid(table, width="100%")
