"""Forecaster tab: placeholder until the prediction model is wired in."""

from __future__ import annotations

from shiny import module

from components.placeholder import placeholder_card


@module.ui
def forecaster_ui():
    return placeholder_card(
        "Length-of-stay forecaster",
        "Enter a patient's details and get a predicted length of stay, with an "
        "explanation of what drove the prediction.",
    )


@module.server
def forecaster_server(input, output, session):
    """No reactivity yet — kept so every tab has the same ui/server pair."""
