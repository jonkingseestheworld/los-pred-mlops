"""A plain HTML table for short, text-heavy content (wraps text, accepts UI elements).

`render.DataGrid` truncates long cells; documentation tables need full wrapping and
inline formatting (e.g. <code> names), so this builds a Bootstrap table directly.
"""

from __future__ import annotations

from typing import Sequence

from shiny import ui


def html_table(
    header: Sequence[str],
    rows: Sequence[Sequence],
    *,
    numeric_cols: Sequence[int] = (),
) -> ui.Tag:
    """Bootstrap table. Cells may be str, number or any shiny UI element.

    `numeric_cols` are right-aligned (column indexes, 0-based).
    """
    def cell(tag, i, content):
        return tag(content, class_="text-end" if i in numeric_cols else None)

    head = ui.tags.thead(ui.tags.tr(*[cell(ui.tags.th, i, h) for i, h in enumerate(header)]))
    body = ui.tags.tbody(*[
        ui.tags.tr(*[cell(ui.tags.td, i, c) for i, c in enumerate(row)]) for row in rows
    ])
    return ui.div(
        ui.tags.table(head, body, class_="table table-sm table-striped align-middle mb-0"),
        class_="table-responsive",
    )
