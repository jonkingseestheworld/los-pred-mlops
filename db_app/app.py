"""Entry point — hospital length-of-stay dashboard (Shiny for Python).

Run from the repo root:
    uv run --group dashboard shiny run --reload db_app/app.py

This file only assembles the app: the page shell and loading screen, then one
module per tab (see `modules/`). Computation lives in `services/`.
"""

from __future__ import annotations

from shiny import App, ui

from components.loading import loading_assets, loading_screen
from modules.description import description_server, description_ui
from modules.forecaster import forecaster_server, forecaster_ui
from modules.group_summary import group_summary_server, group_summary_ui
from services.data import load_dataset
from settings import APP_DIR, APP_TITLE, LOGO_SRC

app_ui = ui.page_navbar(
    ui.nav_panel("Description", description_ui("description")),
    ui.nav_panel("Group Summary", group_summary_ui("group_summary")),
    ui.nav_panel("Forecaster", forecaster_ui("forecaster")),
    ui.nav_spacer(),  # pushes everything after it to the right-hand side of the navbar
    ui.nav_control(ui.img(src=LOGO_SRC, alt="Length of stay logo", class_="navbar-logo")),
    title="HCRU Explorer",
    id="main_nav",
    navbar_options=ui.navbar_options(bg="#001a64", theme="dark"),
    header=ui.TagList(
        ui.include_css(APP_DIR / "www" / "styles.css"),
        loading_assets(),
        loading_screen(),
    ),
    window_title=APP_TITLE,
)


def server(input, output, session):
    data = load_dataset()  # read once per process, shared by every session
    description_server("description", data)
    group_summary_server("group_summary", data)
    forecaster_server("forecaster")


# Shiny only serves www/ by default; expose img/ at /img so the navbar logo resolves.
app = App(app_ui, server, static_assets={"/img": APP_DIR / "img"})
