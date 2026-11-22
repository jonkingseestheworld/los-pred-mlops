"""Description tab: a thin shell holding two sub-tabs, one module each.

    Description
      ├─ Background     (background.py)
      └─ Data cleaning  (data_cleaning.py)

`app.py` only knows about `description_ui` / `description_server`; the sub-tabs
are an implementation detail of this package. Child module ids are namespaced
inside this module automatically (e.g. "description-background-n_rows").
"""

from __future__ import annotations

import pandas as pd
from shiny import module, ui

from modules.description.background import background_server, background_ui
from modules.description.data_cleaning import data_cleaning_server, data_cleaning_ui


@module.ui
def description_ui():
    return ui.navset_underline(
        ui.nav_panel("Background", background_ui("background")),
        ui.nav_panel("Data processing", data_cleaning_ui("data_cleaning")),
        id="sub_tab",
    )


@module.server
def description_server(input, output, session, data: pd.DataFrame):
    background_server("background", data)
    data_cleaning_server("data_cleaning")  # audits the shared, cached dataset itself
