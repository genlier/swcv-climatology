"""Display frozen paired height responses, without recomputing inference.

Question: does raw height rise imply a coherent regional dynamical response?
Claim: removing the zonal background weakens the GHG-only height response.
Primary panel: GHG-only paired decomposition; control: historical all forcing.
Quantitative paired grid, Python, 183 x 70 mm (main) / 183 x 130 mm (complete).
User-requested main subset: GHG and historical; all 44 pairs in complete view.
Original source retains all three components and all control-noise statistics.
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

ROOT = Path(__file__).resolve().parents[1]
EXPS = ['hist-GHG', 'historical', 'hist-aer', 'hist-nat']
METRICS = ['raw_z850_m', 'zonal_z850_m', 'eddy_z850_m']
EXPECTED = dict(zip(EXPS, [11, 13, 10, 10]))
COLORS = {'hist-GHG': '#BF5B16', 'historical': '#205A89',
          'hist-aer': '#555555', 'hist-nat': '#555555'}
NAMES = {'hist-GHG': 'GHG only', 'historical': 'Historical',
         'hist-aer': 'Aerosol only', 'hist-nat': 'Natural only'}

def validate_source(comparison):
    q = comparison.copy()
    if set(q.metric) != set(METRICS) or set(q.experiment) != set(EXPS):
        raise ValueError('Unexpected metric or experiment allocation')
    if q.duplicated(['source', 'experiment', 'metric']).any():
        raise ValueError('Repeated model-experiment-component rows')
    if not np.isfinite(q[['forced_response', 'picontrol_noise_sd', 'signal_to_noise']].to_numpy()).all():
        raise ValueError('Nonfinite observations cannot be silently excluded')
    if (q.picontrol_noise_sd <= 0).any():
        raise ValueError('Invalid control-noise denominator')
    if not np.allclose(q.forced_response/q.picontrol_noise_sd, q.signal_to_noise, rtol=1e-10, atol=1e-12):
        raise ValueError('Control-scaled source values disagree')
    counts = q.groupby(['experiment', 'metric']).size().to_dict()
    if counts != {(e, m): n for e, n in EXPECTED.items() for m in METRICS}:
        raise ValueError('Frozen paired cohort changed')
    units = q.groupby(['source', 'experiment'])
    if not units.metric.nunique().eq(3).all() or not units.member.nunique().eq(1).all():
        raise ValueError('Components must use identical model members')
    p = q.pivot(index=['source', 'experiment', 'member'], columns='metric', values='forced_response')
    if not np.allclose(p.raw_z850_m-p.zonal_z850_m, p.eddy_z850_m, rtol=1e-10, atol=1e-10):
        raise ValueError('Raw minus zonal does not equal eddy for matched members')
    return q.sort_values(['experiment', 'source', 'metric']).reset_index(drop=True)

def render(q, experiments, basename, full=False):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    if full:
        fig, grid = plt.subplots(2, 2, figsize=(7.2047244094, 5.1181102362), sharey=True)
        fig.subplots_adjust(left=.09, right=.98, bottom=.13, top=.88, hspace=.45, wspace=.22)
    else:
        fig, grid = plt.subplots(1, 2, figsize=(7.2047244094, 2.7559055118), sharey=True)
        fig.subplots_adjust(left=.09, right=.98, bottom=.25, top=.77, wspace=.22)
    axes = np.asarray(grid).ravel()
    for letter, ax, exp in zip('abcd', axes, experiments):
        data = q.loc[q.experiment.eq(exp)].pivot(index=['source', 'member'], columns='metric', values='forced_response')
        for values in data[['raw_z850_m', 'eddy_z850_m']].to_numpy():
            ax.plot([0, 1], values, color='#999999', alpha=.6, lw=.65, marker='o', ms=2.5, zorder=2)
        medians = data[['raw_z850_m', 'eddy_z850_m']].median().to_numpy()
        ax.plot([0, 1], medians, color=COLORS[exp], lw=2.1, marker='D', ms=4.5, zorder=4)
        # Reference stops at the observed categories; labels occupy a separate margin.
        ax.plot([-.04, 1.04], [0, 0], color='#333333', lw=.9, zorder=1)
        ax.text(-.10, medians[0], f'{medians[0]:+.2f} m', color=COLORS[exp], ha='right', va='center', fontsize=8)
        ax.text(1.10, medians[1], f'{medians[1]:+.2f} m', color=COLORS[exp], ha='left', va='center', fontsize=8)
        ax.set_xlim(-.72, 1.80)
        ax.set_ylim(-4.5, 8.8)
        ax.set_yticks([-4, 0, 4, 8])
        ax.set_xticks([0, 1], ['Raw height', 'Eddy height'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(f'{NAMES[exp]} (n = {len(data)} models)', loc='left', pad=10)
        ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-13, 10),
                    textcoords='offset points', fontsize=9, weight='bold', va='bottom')
    axes[0].set_ylabel('Z850 response (m)')
    if full:
        axes[2].set_ylabel('Z850 response (m)')
    fig.text(.09, .94, 'JJA 1985–2014 minus 1951–1980; paired selected model members', fontsize=8)
    fig.text(.09, .065 if full else .07, 'Grey: individual models. Thick line and diamonds: descriptive model medians.', fontsize=8)
    fig.text(.09, .025, 'Eddy height = raw height minus the 15–25°N zonal-mean height.', fontsize=8)
    target = ROOT/'figures'/basename
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=target.with_suffix('.alignment.json'),
                                       require_panel_labels=True, strict=True)
    fig.savefig(target.with_suffix('.pdf'))
    fig.savefig(target.with_suffix('.svg'))
    fig.savefig(target.with_suffix('.png'), dpi=600)
    plt.close(fig)

def make_figure(comparison):
    q = validate_source(comparison)
    render(q, EXPS[:2], 'cmip6_raw_vs_eddy_z850_response')
    render(q, EXPS, 'cmip6_raw_vs_eddy_z850_all_forcings', full=True)
    # All components retained, including zonal component not drawn as another line.
    q.to_csv(ROOT/'data/processed/cmip6_height_figure_source.csv', index=False)
    source = ROOT/'data/processed/cmip6_eddy_z850_forced_vs_control.csv'
    audit = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'source_rows': len(q), 'source_pairs': 44, 'main_display_pairs': 24,
             'complete_display_pairs': 44, 'scientific_exclusions': 0,
             'main_allocation_rule': 'User-requested hist-GHG and historical; aerosol/natural in complete view and S10',
             'component_allocation': 'Raw and eddy displayed; zonal retained in source and S10; noise metrics retained in S10',
             'counts': EXPECTED, 'center': 'marginal model medians, not median paired difference',
             'spread': 'every individual paired model response shown; no inferential interval for descriptive median',
             'members': 'one identical selected member for raw/zonal/eddy within each model-experiment',
             'inference': 'No new statistical tests or direct vortex-count attribution'}
    (ROOT/'data/processed/cmip6_height_figure_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    make_figure(pd.read_csv(ROOT/'data/processed/cmip6_eddy_z850_forced_vs_control.csv'))
