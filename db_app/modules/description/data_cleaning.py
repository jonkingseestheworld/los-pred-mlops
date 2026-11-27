"""Description > Data processing: a short, educational walk-through of how the sample data were prepared.

Static text only. (The measured audit of the real pipeline still lives in
`services.cleaning`; the Forecaster tab uses it for the list of model inputs.)
"""

from __future__ import annotations

from shiny import module, ui


def _code(text: str) -> ui.Tag:
    return ui.tags.code(text)


def _step(number: int, title: str, *text) -> ui.Tag:
    """One processing step as its own small card."""
    return ui.card(ui.card_header(f"{number}. {title}"), ui.div(*text))


@module.ui
def data_cleaning_ui():
    return ui.div(
        ui.card(
            ui.card_header("The data"),
            ui.p(
                "This open-source dataset, published by Microsoft, contains 100,000 records "
                "with 28 columns."
            ),
            ui.p(
                "Each row represents one hospital admission and is associated with a unique "
                "episode ID (", _code("eid"), "). There are no duplicates (each ",
                _code("eid"), " is unique) and no missing values. For this exercise, every "
                "record is regarded as an independent case."
            ),
        ),
        ui.card(
            ui.card_header("Train / test split"),
            ui.p(
                "For model building in this example, the data were split ",
                ui.tags.b("by time"),
                " before anything was learned: earlier months were used for training and later "
                "months were retained for testing, so the test set stays unseen. One reason for "
                "splitting the data by time is that it prepares the pipeline for drift "
                "monitoring, should this be needed in the future."
            ),
        ),
        ui.card(
            ui.card_header("Data processing steps"),
            ui.layout_column_wrap(
                _step(
                    1, "Remove information that gives the answer away",
                    "The difference between the ", _code("discharged"), " date and the ",
                    _code("vdate"), " (visit date) equals the length of stay (i.e. the answer is "
                    "in the data), so these date fields are excluded as features in model training.",
                ),
                _step(
                    2, "Check for impossible values and extreme outliers",
                    "Impossible values are replaced and extreme values are capped. Examples:",
                    ui.tags.ul(
                        ui.tags.li(
                            "The ", _code("glucose"), " field has a negative value (impossible), "
                            "which is replaced by the training median."
                        ),
                        ui.tags.li(
                            "The ", _code("neutrophils"), " and ", _code("bloodureanitro"),
                            " fields have extremely high values, which are capped at the "
                            "training 99th percentile."
                        ),
                        class_="mt-2 mb-0 ps-3",
                    ),
                ),
                _step(
                    3, "Recode categorical fields",
                    _code("gender"), " and ", _code("facid"), " (facility) are recoded into "
                    "numeric indicator columns:",
                    ui.tags.ul(
                        ui.tags.li(
                            _code("gender"), " becomes a single 0/1 column named ", _code("isMale"),
                            " (1 = male, 0 = female)."
                        ),
                        ui.tags.li(
                            _code("facid"), " is one-hot coded: one 0/1 column per facility."
                        ),
                        class_="mt-2 mb-0 ps-3",
                    ),
                ),
                _step(
                    4, "Derive a new variable",
                    _code("comorbidity_count"), " counts the number of comorbidities each "
                    "patient has.",
                ),
                width="180px",
                fill=False,
            ),
            ui.p(
                ui.tags.b("Preventing data leakage. "),
                "All values used in the steps above (for example, medians and caps) are learnt "
                "from the training data only.",
                class_="mt-3 mb-0",
            ),
            ui.p(
                ui.tags.b("One code path. "),
                "The same cleaning function is used for both training and prediction, "
                "so the model never sees differently treated data.",
                class_="mt-3 mb-0",
            ),
        ),
        ui.card(
            ui.card_header("Other considerations in real-life practice"),
            ui.tags.ul(
                ui.tags.li(
                    "A unique ", ui.tags.b("patient ID"), " would usually come with this kind of "
                    "data. It would let us verify, and handle differently, cases where multiple "
                    "records belong to the same patient."
                ),
                ui.tags.li(
                    "We would also usually know the ", ui.tags.b("inclusion and exclusion criteria"),
                    " of the specific study cohort. These are essential for understanding how a "
                    "model should be used, and they point to the model's limitations."
                ),
                ui.tags.li(
                    "For this demo, impossible values are replaced and extreme values are capped; ",
                    ui.tags.b("no rows are removed"), ". "
                    "In reality, implausible records might instead lead to the entire row being "
                    "dropped, or a different imputation method might be applied."
                ),
            ),
            class_="border-info",
        ),
    )


@module.server
def data_cleaning_server(input, output, session):
    """No reactivity — kept so every tab has the same ui/server pair."""
