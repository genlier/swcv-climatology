"""Make the winter trend argument readable without changing its analysis.

Question: what temporal structure underlies the winter increase?
a establishes frequency; b/c bound pressure-level intensity; d decomposes months.
Python quantitative grid, 183 x 125 mm. Every observation retained.
Reconstruct frozen fitted curves; assert original breakpoints/BIC and slopes.
No new bootstrap, p value, significant-subset selection or causal inference.
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audit_panel_alignment import require_matplotlib_panel_alignment
from analyze_djf_change_structure import design, select_break_poisson, select_break_ols

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/processed'
RESPONSES = ['frequency', 'q90_height_lifecycle', 'q90_vorticity_lifecycle']
MONTHS = ['dec_frequency', 'jan_frequency', 'feb_frequency']

def fitted_sources(annual, breaks, trends):
    if list(annual.season_year) != list(range(1980, 2024)):
        raise ValueError('Expected sorted 44 complete winters')
    if not np.isfinite(annual[RESPONSES+MONTHS].to_numpy()).all():
        raise ValueError('Missing observations cannot be silently excluded')
    if not np.array_equal(annual[MONTHS].sum(axis=1).to_numpy(), annual.frequency.to_numpy()):
        raise ValueError('Month decomposition must reproduce every winter count')
    source = annual[['season_year']+RESPONSES+MONTHS].copy()
    for response in RESPONSES:
        row = breaks.loc[breaks.response.eq(response)]
        if len(row) != 1:
            raise ValueError('Expected one frozen breakpoint row per endpoint')
        row = row.iloc[0]
        selector = select_break_poisson if response == 'frequency' else select_break_ols
        base, hinge, selected, _, bic = selector(annual.season_year.to_numpy(float), annual[response].to_numpy(float))
        if selected != row.selected_breakpoint or not np.isclose(bic, row.delta_bic_linear_minus_segmented, rtol=1e-8, atol=1e-8):
            raise ValueError('Display fit does not reproduce frozen selection')
        slope = trends.loc[trends.response.eq(response), 'coefficient_per_decade']
        if len(slope) != 1 or not np.isclose(base.params[1]*10, slope.iloc[0], rtol=1e-8, atol=1e-10):
            raise ValueError('Single-slope display differs from retained trend')
        source[response+'_single_slope'] = base.predict(design(annual.season_year.to_numpy(float)))
        source[response+'_searched_hinge'] = hinge.predict(design(annual.season_year.to_numpy(float), selected))
    return source

def make_figure(annual, breaks, trends):
    source = fitted_sources(annual, breaks, trends)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(7.2047244094, 4.9212598425))
    fig.subplots_adjust(left=.10, right=.965, bottom=.15, top=.88, wspace=.35, hspace=.58)
    titles = ['Winter event frequency', 'Terrain-safe height-deficit tail', 'Terrain-safe vorticity tail']
    labels = ['DJF event count', 'P90 height deficit (gpm)', 'P90 vorticity']
    for ax, response, title, label in zip(axes.ravel(), RESPONSES, titles, labels):
        scale = 1e5 if response == 'q90_vorticity_lifecycle' else 1
        ax.plot(source.season_year, source[response]*scale, color='#999999', marker='o', ms=2.2, lw=.6)
        ax.plot(source.season_year, source[response+'_single_slope']*scale, color='#205A89', lw=1.6,
                ls='--', label='Single-slope fit')
        ax.plot(source.season_year, source[response+'_searched_hinge']*scale, color='#555555', lw=1.0,
                label='Searched hinge (descriptive)')
        ax.set_title(title, loc='left', pad=11)
        ax.set_ylabel(label)
        ax.set_xlim(1979, 2025)
    axes[1,0].set_ylabel(r'P90 vorticity ($10^{-5}$ s$^{-1}$)', fontsize=9)
    colors = ['#888888', '#555555', '#205A89']
    display_names = ['Dec', 'Jan', 'Feb']
    end_positions = [10, 12.5, 5.7]
    monthly_ax = axes[1,1]
    for column, name, color, position in zip(MONTHS, display_names, colors, end_positions):
        monthly_ax.plot(source.season_year, source[column], color=color,
                        lw=1.9 if name == 'Feb' else .8, marker='o', ms=2.3 if name == 'Feb' else 1.8,
                        ls='--' if name == 'Jan' else '-')
        monthly_ax.plot([2023, 2024], [source[column].iloc[-1], position], color=color, lw=.6)
        if name == 'Feb':
            rate = trends.loc[trends.response.eq(column), 'effect_percent_per_decade'].iloc[0]
            text = f'Feb\n+{rate:.2f}%/dec'
        else:
            text = name
        monthly_ax.text(2024.5, position, text, color=color, fontsize=8, va='center')
    monthly_ax.set_xlim(1979, 2039)
    monthly_ax.set_ylim(0, float(source[MONTHS].max().max())+2)
    monthly_ax.set_ylabel('Monthly genesis count')
    monthly_ax.set_title('Monthly contributions to winter growth', loc='left', pad=11)
    for letter, ax in zip('abcd', axes.ravel()):
        ax.set_xticks([1980, 2000, 2020])
        ax.set_xlabel('DJF season year')
        ax.spines[['top', 'right']].set_visible(False)
        ax.annotate(letter, xy=(0,1), xycoords='axes fraction', xytext=(-13,11),
                    textcoords='offset points', fontsize=9, weight='bold', va='bottom')
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.10,.99),
               ncol=2, frameon=False, fontsize=8)
    fig.text(.10, .035, 'Searched hinges are descriptive: selection-adjusted tests do not reject single-slope models.', fontsize=7.5)
    target = ROOT/'figures/djf_change_structure'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=target.with_suffix('.alignment.json'), require_panel_labels=True, strict=True)
    fig.savefig(target.with_suffix('.pdf'))
    fig.savefig(target.with_suffix('.svg'))
    fig.savefig(target.with_suffix('.png'), dpi=600)
    plt.close(fig)
    source.to_csv(OUT/'djf_revision_figure_source.csv', index=False)
    audit = {'source_sha256': {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                              for name in ['djf_change_structure_annual.csv', 'djf_selected_breakpoint_tests.csv', 'djf_change_source_trends.csv']},
             'winter_years': 44, 'observed_values': 264, 'dropped_observations': 0,
             'curve_reconstruction': 'Frozen selected breakpoint, BIC and single-slope coefficients reproduced; no bootstrap rerun',
             'vorticity_display_scale': 1e5, 'monthly_label_offsets': 'Leader labels only; data coordinates unchanged',
             'February_emphasis': 'User-requested highlight; neither largest growth rate nor all winter growth',
             'uncertainty': 'Observed annual values and descriptive fits; inference retained in frozen tables and manuscript'}
    (OUT/'djf_revision_figure_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    make_figure(pd.read_csv(OUT/'djf_change_structure_annual.csv'), pd.read_csv(OUT/'djf_selected_breakpoint_tests.csv'),
                pd.read_csv(OUT/'djf_change_source_trends.csv'))
