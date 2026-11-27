"""Description > Background: where the data came from, how to use the dashboard, data dictionary."""

from __future__ import annotations

import pandas as pd
from shiny import module, reactive, render, ui
from traitlets import This

from services.variables import dictionary_frame
from settings import APP_TITLE, KAGGLE_URL, MICROSOFT_DOC_URL


def _link(text: str, href: str) -> ui.Tag:
    return ui.a(text, href=href, target="_blank", rel="noopener")


@module.ui
def background_ui():
    return ui.div(
        ui.card(
            ui.p(
                "This is a mock dashboard set up to illustrate how it might be used "
                "to help researchers and decision-makers (1) understand the prevalence and "
                "characteristics of a clinical cohort of interest, and (2) estimate an "
                "outcome (here, length of stay) from a patient's baseline profile using "
                "a trained statistical model."
            ),
            ui.div(
                ui.input_action_button(
                    "disclaimer", "Disclaimer", class_="btn-light btn-sm disclaimer-btn",
                ),
                class_="text-center",
            ),
            class_="bg-secondary",  # Add this
        ),
        ui.div(
            ui.h2(APP_TITLE),

        ),
        ui.card(
            ui.card_header("Background"),
            ui.p(
                "A patient's length of stay (LOS) is central to hospital management, "
                "shaping a healthcare institution's operational efficiency, financial "
                "outcomes and clinical outcomes. Efficient patient flow, including streamlined "
                "admissions, timely transfers and well-coordinated discharges, shortens "
                "stays and waiting times, which can in turn improve patient outcomes and "
                "satisfaction (1,2)."
            ),
            ui.p(
                "In healthcare research, LOS is a key measure of healthcare resource "
                "utilisation (HCRU). HCRU is typically used to quantify the baseline "
                "economic burden of a disease, or to estimate cost offsets: the savings "
                "that arise when a new treatment reduces downstream medical events."
            ),
        ),
        # ui.layout_columns(
        #    ui.value_box("Admissions", ui.output_text("n_rows")),
        #    ui.value_box("Columns", ui.output_text("n_cols")),
        #    ui.value_box("Visit dates", ui.output_text("date_range")),
        #    ui.value_box("Facilities", ui.output_text("n_facilities")),
        #    fill=False,
        #),
        ui.layout_columns(
            ui.card(
                            ui.card_header("About this dashboard"),
                            ui.p("The app contains three main tabs:"),
                            ui.tags.ol(
                                ui.tags.li(ui.tags.b("Description"), " - background and data processing details"),
                                # add new line here
                                ui.br(),
                                ui.tags.li(
                                    ui.tags.b("Cohort summary"),
                                    " — summary statistics of the population cohort",
                                ),
                                ui.br(),
                                ui.tags.li(ui.tags.b("Forecaster"), " - Length of Stay predictions (based on the patient's characteristics)"),
                            ),

                        ),
            ui.card(
                 ui.card_header("Where the data comes from"),
                 ui.p(
                     "The Hospital Length of Stay dataset is open-sourced by Microsoft "
                     "as a synthetic sample for demonstrating length-of-stay prediction. "
                     "It is also available on Kaggle."
                 ),
                 ui.tags.ul(
                     ui.tags.li(_link("Kaggle dataset page", KAGGLE_URL)),
                     ui.tags.li(_link("Microsoft's data description", MICROSOFT_DOC_URL)),
                 ),
                 ui.p(
                     "Each row represents one hospital episode, covering patient demographics, comorbidity "
                     "history, lab values and vitals, the facility, and "
                     "the outcome of interest — ",
                     ui.tags.b("length of stay in days"),
                     ".",
                 ),
                 ui.p(
                     ui.tags.b("Note: "),
                     "Check the source pages for licence terms. Synthetic data may be cleaner than real hospital records; do not use anything "
                     "here for clinical decisions. ",
                     class_="data_disclaimer",
                 ),
                 
             ),
        ), 
        ui.layout_columns(
            

        ),
        #ui.card(
        #    ui.card_header("Data dictionary"),
        #    ui.output_data_frame("dictionary"),
        #    full_screen=True,
        #),
        ui.card(
            ui.card_header("References"),
            ui.tags.ol(
                ui.tags.li(
                    "Stone K, Zwiggelaar R, Jones P, et al. A systematic review of the "
                    "prediction of hospital length of stay: Towards a unified framework. "
                    "PLOS Digit Health 2022;1:e0000017."
                ),
                ui.tags.li(
                    "Sonis JD, Aaronson EL, Lee RY, et al. Emergency Department Patient "
                    "Experience: A Systematic Review of the Literature. "
                    "J Patient Exp 2018;5:101-6."
                ),
                class_="small mb-0",
            ),
        ),
    )


@module.server
def background_server(input, output, session, data: pd.DataFrame):
    @reactive.effect
    @reactive.event(input.disclaimer)
    def _show_disclaimer():
        ui.modal_show(
            ui.modal(
                ui.p("This is only a demo.Do not use anything here for making clinical decisions."),
                title="!!! WARNING !!!",
                easy_close=True,
                footer=ui.modal_button("Close"),
                class_="disclaimer-modal text-center",
            )
        )

    #@render.text
    #def n_rows():
    #    return f"{len(data):,}"

    #@render.text
    #def n_cols():
    #    return str(len(dictionary_frame()) - 1)
    #    # the 28 original columns (excludes the ones this app derives)

    #def date_range():
    ##@render.text
    #    return f"{data['vdate'].min():%b %Y} – {data['vdate'].max():%b %Y}"

    #@render.text
    #def n_facilities():
    #    return str(data["facid"].nunique())

    #@render.data_frame
    #def dictionary():
    #    return render.DataGrid(dictionary_frame(), width="100%", filters=True)
