"""A "coming soon" card for tabs that aren't built yet."""

from __future__ import annotations

from shiny import ui


def placeholder_card(title: str, text: str, badge: str = "Coming soon") -> ui.Tag:
    return ui.card(
        ui.div(
            ui.span(badge, class_="badge text-bg-secondary mb-3"),
            ui.h3(title),
            ui.p(text),
            class_="placeholder-card",
        ),
    )
