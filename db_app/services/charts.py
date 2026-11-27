"""Length-of-stay histogram as inline SVG — plain Python, no plotting library.

The input is the pre-computed table of admissions per day (`summary.los_counts`, all patients), so
drawing is a few dozen rectangles no matter how many patients sit behind them. Output is
an HTML string; the Shiny module just wraps it in `ui.HTML`.

Design: one hue (the value is magnitude, so colour carries no identity), thin bars with a
2px surface gap and a rounded data end on a square baseline, hairline grid, the tallest bar
labelled at its tip and every bar's exact value in its hover tooltip, text in text tokens.
Colours and fonts come from CSS (`www/styles.css`, classes `hist-*`).
"""

from __future__ import annotations

import math
from html import escape

import pandas as pd

W, H = 480, 250                          # SVG viewBox
LEFT, RIGHT, TOP, BOTTOM = 50, 10, 22, 44
MAX_BAR, GAP, RADIUS = 24, 2, 4          # px in viewBox units


def nice_step(max_value: float, target_ticks: int = 4) -> int:
    """A round tick step (1, 2, 5 x 10^k, at least 1) giving about `target_ticks` ticks."""
    if max_value <= 0:
        return 1
    raw = max_value / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * magnitude:
            return max(1, int(m * magnitude))
    return max(1, int(10 * magnitude))


def _tick(value: int, compact: bool) -> str:
    """Full numbers below 10,000 (one style per axis); 10K-style once the axis reaches it."""
    return f"{value / 1000:g}K" if compact and value else f"{value:,}"


def _plural(days: int) -> str:
    return f"{days} day" if days == 1 else f"{days} days"


def _bar_path(x: float, base: float, width: float, height: float) -> str:
    """Square at the baseline, rounded at the data end."""
    r = min(RADIUS, width / 2, height)
    return (
        f"M{x:.1f},{base:.1f}V{base - height + r:.1f}"
        f"a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{-r:.1f}H{x + width - r:.1f}"
        f"a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f}V{base:.1f}Z"
    )


def histogram_svg(
    days: list[int],
    counts: list[int],
    *,
    y_max: int | None = None,
    x_label: str = "Length of stay (days)",
    y_label: str = "Admissions",
    aria_label: str = "Length of stay histogram",
) -> str:
    """One histogram panel. Pass a shared `y_max` to put several panels on one scale."""
    total = sum(counts)
    top_value = max(y_max if y_max is not None else max(counts, default=0), 1)
    step = nice_step(top_value)
    axis_max = step * math.ceil(top_value / step)
    compact = axis_max >= 10_000

    plot_w, plot_h = W - LEFT - RIGHT, H - TOP - BOTTOM
    base = TOP + plot_h
    slot = plot_w / len(days)
    bar_w = min(MAX_BAR, slot - GAP)
    y_of = lambda v: base - plot_h * v / axis_max            # noqa: E731

    parts = [
        f'<svg class="hist" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria_label)}" '
        'xmlns="http://www.w3.org/2000/svg">'
    ]
    for v in range(0, axis_max + 1, step):                    # hairline grid + y ticks
        y = y_of(v)
        parts.append(f'<line class="hist-grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="hist-tick" x="{LEFT - 6}" y="{y + 3.5:.1f}" text-anchor="end">{_tick(v, compact)}</text>')

    peak = max(range(len(counts)), key=counts.__getitem__) if counts and max(counts) > 0 else None
    label_every = 1 if len(days) <= 20 else math.ceil(len(days) / 15)
    for i, (d, n) in enumerate(zip(days, counts)):
        x0 = LEFT + i * slot
        cx = x0 + slot / 2
        pct = f" ({100 * n / total:.1f}%)" if total else ""
        tip = f"{_plural(d)}: {n:,}{pct}"
        h = base - y_of(n)
        bar = f'<path class="hist-bar" d="{_bar_path(cx - bar_w / 2, base, bar_w, h)}"/>' if n > 0 else ""
        parts.append(
            f'<g class="hist-col"><rect class="hist-hit" x="{x0:.1f}" y="{TOP}" width="{slot:.1f}" '
            f'height="{plot_h}"/>{bar}<title>{escape(tip)}</title></g>'
        )
        if i % label_every == 0:
            parts.append(f'<text class="hist-tick" x="{cx:.1f}" y="{base + 15}" text-anchor="middle">{d}</text>')
        if i == peak:                                         # label only the extreme
            parts.append(f'<text class="hist-value" x="{cx:.1f}" y="{y_of(n) - 5:.1f}" text-anchor="middle">{n:,}</text>')

    parts.append(f'<line class="hist-axis" x1="{LEFT}" x2="{W - RIGHT}" y1="{base}" y2="{base}"/>')
    parts.append(f'<text class="hist-label" x="{LEFT + plot_w / 2:.1f}" y="{H - 6}" text-anchor="middle">{escape(x_label)}</text>')
    parts.append(
        f'<text class="hist-label" transform="translate(12,{TOP + plot_h / 2:.1f}) rotate(-90)" '
        f'text-anchor="middle">{escape(y_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def histogram_html(counts: pd.DataFrame) -> str:
    """The histogram for a `Days, n` table, plus the same numbers as a table (a chart is
    never the only way to get at the values)."""
    days = [int(d) for d in counts["Days"]]
    ns = [int(n) for n in counts["n"]]
    # transposed: one column per day (header), one row of counts underneath
    head = "".join(f"<th class='text-end'>{d}</th>" for d in days)
    cells = "".join(f"<td class='text-end'>{n:,}</td>" for n in ns)
    return (
        f'<div class="hist-wrap">' #<p class="hist-caption"> All admissions <span class="hist-n">n = {sum(ns):,}</span></p>'
        f'{histogram_svg(days, ns)}</div>'
        '<details class="hist-table mt-3"><summary>View as table</summary>'
        '<div class="table-responsive"><table class="table table-sm table-striped mb-0">'
        f"<thead><tr><th>Length of stay (days)</th>{head}</tr></thead>"
        f"<tbody><tr><th scope='row'>Admissions</th>{cells}</tr></tbody></table></div></details>"
    )
