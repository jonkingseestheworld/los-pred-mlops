"""Full-screen loading page shown until Shiny has connected and rendered.

The overlay is plain HTML in the initial page, so it paints immediately — before
the websocket connects and before the (potentially slow) server work finishes.
`www/loading.js` removes it on Shiny's first `shiny:idle` event.
"""

from __future__ import annotations

from shiny import ui

from settings import APP_DIR, APP_TITLE, LOADING_MESSAGE, LOGO_SRC, ORG_ABBR, ORG_NAME, TEAM_NAME

WWW = APP_DIR / "www"


def loading_screen() -> ui.Tag:
    return ui.div(
        ui.h1(APP_TITLE, class_="app-loading__title"),
        ui.div(class_="app-loading__rule"),
        ui.div(
            ui.div(ORG_ABBR, class_="app-loading__logo"),
            ui.img(src=LOGO_SRC, alt="", class_="app-loading__img"),
            ui.div(ORG_NAME, class_="app-loading__org"),
            class_="app-loading__brand",
        ),
        #ui.div(TEAM_NAME, class_="app-loading__team"),
        ui.div(class_="app-loading__spinner", role="progressbar", aria_label="Loading"),
        ui.p(LOADING_MESSAGE, class_="app-loading__message"),
        id="app-loading",
        class_="app-loading",
    )


def loading_assets() -> ui.TagList:
    """CSS + JS for the overlay (inlined into the page head)."""
    return ui.TagList(ui.include_css(WWW / "loading.css"), ui.include_js(WWW / "loading.js"))
