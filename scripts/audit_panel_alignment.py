"""Portable plot-rectangle QA for the public source snapshot.

This project-local implementation checks equal-span subplot dimensions and
alignment, and panel label presence. It is not the workstation's more extensive
layout auditor; it does not check text collisions or scientific validity.
No numerical analysis is changed by this plotting-only replacement.
"""
from pathlib import Path
import json
from html import escape


def require_matplotlib_panel_alignment(fig, json_out=None, overlay_svg=None,
        axes=None, panel_ids=None, row_groups=None, require_panel_labels=False,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True):
    fig.canvas.draw()
    panels = list(axes) if axes is not None else [
        ax for ax in fig.axes if ax.get_subplotspec() is not None]
    width, height = fig.get_size_inches() * 72
    entries, failures = [], []
    for i, ax in enumerate(panels):
        ident = panel_ids.get(ax, chr(97+i)) if panel_ids else chr(97+i)
        box = ax.get_position()
        spec = ax.get_subplotspec()
        entry = dict(id=ident, x=box.x0*width, y=box.y0*height,
            width=box.width*width, height=box.height*height,
            row=[spec.rowspan.start, spec.rowspan.stop],
            column=[spec.colspan.start, spec.colspan.stop])
        entries.append(entry)
        if require_panel_labels and not any(
                t.get_text().strip() == ident for t in ax.texts):
            failures.append(f'Panel {ident}: missing standalone label')
    for i, a in enumerate(entries):
        for b in entries[i+1:]:
            if a['row'] == b['row']:
                for key in ('y', 'height'):
                    if abs(a[key]-b[key]) > tolerance_pt:
                        failures.append(f"{a['id']}/{b['id']}: mismatched {key}")
                if a['column'][1]-a['column'][0] == b['column'][1]-b['column'][0]:
                    if abs(a['width']-b['width']) > tolerance_pt:
                        failures.append(f"{a['id']}/{b['id']}: mismatched width")
            if a['column'] == b['column']:
                for key in ('x', 'width'):
                    if abs(a[key]-b[key]) > tolerance_pt:
                        failures.append(f"{a['id']}/{b['id']}: mismatched {key}")
    report = dict(auditor='public-rectangle-QA', panels=entries,
        tolerance_pt=tolerance_pt, failures=failures,
        scope='Rectangle alignment and panel labels; no text collision or gutter audit')
    if json_out:
        target = Path(json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if overlay_svg:
        target = Path(overlay_svg)
        target.parent.mkdir(parents=True, exist_ok=True)
        shapes = ''.join(f'<rect x="{p["x"]}" y="{height-p["y"]-p["height"]}" '
            f'width="{p["width"]}" height="{p["height"]}" fill="none" stroke="red"/>'
            f'<title>{escape(p["id"])}</title>' for p in entries)
        target.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}">{shapes}</svg>', encoding='utf-8')
    if strict and failures:
        raise ValueError('Panel alignment failed: ' + '; '.join(failures))
    return report
