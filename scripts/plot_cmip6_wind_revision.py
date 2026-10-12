"""Plot frozen CMIP6 wind responses; no re-estimation or model selection.

Question: are forced circulation responses coherent across models?
Panels c/d in the joint manuscript figure: response and descriptive noise scale.
Archetype: paired categorical evidence, 183 x 100 mm, vector PDF/SVG + PNG.
Every finite WNPAC row is plotted: 15/15/14/15 for historical/GHG/aer/nat.
No inferential threshold or new statistical test is introduced.
"""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[1]
EXPS = ['historical', 'hist-GHG', 'hist-aer', 'hist-nat']
COLORS = {'BCC-CSM2-MR': '#B33D35', 'CNRM-CM6-1': '#246BA4'}

def validate_source(comparison):
    q = comparison.loc[comparison.metric.eq('wnpac_u850_ms')].copy()
    if q.duplicated(['source', 'experiment']).any():
        raise ValueError('Repeated model-experiment rows')
    if set(q.experiment) != set(EXPS):
        raise ValueError('Unexpected experiment allocation')
    if not np.isfinite(q[['forced_response', 'signal_to_noise']].to_numpy()).all():
        raise ValueError('Nonfinite observations may not be silently excluded')
    counts = q.groupby('experiment').size().to_dict()
    if counts != dict(zip(EXPS, [15, 15, 14, 15])):
        raise ValueError(f'Frozen cohort changed: {counts}')
    extremes = q.loc[q.experiment.eq('hist-GHG') & q.signal_to_noise.abs().ge(2)]
    if set(extremes.source) != set(COLORS):
        raise ValueError('Highlighted models no longer match frozen observations')
    return q.sort_values(['experiment', 'source']).reset_index(drop=True)

def make_figure(comparison):
    q = validate_source(comparison)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(7.2047244094, 3.9370078740))  # 183 x 100 mm
    fig.subplots_adjust(left=.16, right=.975, bottom=.18, top=.83, wspace=.27)
    models = sorted(q.source.unique())
    offsets = dict(zip(models, np.linspace(-.16, .16, len(models))))
    yvalues = dict(zip(EXPS, [3, 2, 1, 0]))
    for ax, metric in zip(axes, ['forced_response', 'signal_to_noise']):
        for row in q.itertuples():
            highlight = row.experiment == 'hist-GHG' and row.source in COLORS
            ax.scatter(getattr(row, metric), yvalues[row.experiment]+offsets[row.source],
                       s=23 if highlight else 12, c=COLORS[row.source] if highlight else '#999999',
                       marker='D' if highlight else 'o', linewidths=0,
                       alpha=1 if highlight else .65, zorder=4 if highlight else 2)
        ax.axvline(0, color='#333333', lw=1.25, zorder=1)
        ax.set_ylim(-.45, 3.85)
        ax.set_yticks([3, 2, 1, 0])
        ax.set_yticklabels(['Historical (15)', 'GHG only (15)', 'Aerosol only (14)', 'Natural only (15)'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_xlim(-1.65, 1.95)
    axes[0].set_xticks([-1, 0, 1])
    axes[0].set_xlabel('Wind-index change (m/s)')
    axes[0].set_title('Responses span both signs', loc='left', pad=12)
    axes[1].set_xlim(-3.35, 3.35)
    axes[1].set_xticks([-3, -2, -1, 0, 1, 2, 3])
    axes[1].set_yticklabels([])
    axes[1].set_xlabel('Response / piControl noise SD')
    axes[1].set_title('GHG outliers have opposite signs', loc='left', pad=12)
    for value in [-2, 2]:
        axes[1].plot([value, value], [-.45, 3.3], color='#777777', ls=(0, (3, 3)), lw=.8, zorder=1)
        axes[1].text(value, 3.65, f'{value:+d}', ha='center', va='center', fontsize=8)
    for model, ypos, xpos, align in [('BCC-CSM2-MR', 2.55, .1, 'left'),
                                      ('CNRM-CM6-1', 1.45, -.1, 'right')]:
        row = q.loc[q.source.eq(model) & q.experiment.eq('hist-GHG')].iloc[0]
        axes[1].text(xpos, ypos, f'{model}\n{row.signal_to_noise:+.2f} SD',
                     color=COLORS[model], ha=align, va='center', fontsize=7.2)
    for letter, ax in zip('cd', axes):
        ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-13, 12),
                    textcoords='offset points', fontsize=9, weight='bold', va='bottom')
    fig.text(.16, .94, '1985–2014 minus 1951–1980; one selected member per model', fontsize=8)
    fig.text(.16, .035, 'Dashed lines: descriptive ±2 noise SD, not a significance test. All available model points shown.', fontsize=7.5)
    target = ROOT / 'figures/cmip6_wnpac_wind_fingerprints'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=target.with_suffix('.alignment.json'),
                                       require_panel_labels=True, strict=True)
    fig.savefig(target.with_suffix('.pdf'))
    fig.savefig(target.with_suffix('.svg'))
    fig.savefig(target.with_suffix('.png'), dpi=600)
    plt.close(fig)
    q.to_csv(ROOT/'data/processed/cmip6_wind_figure_source.csv', index=False)
    source = ROOT/'data/processed/cmip6_wnpac_wind_forced_vs_control.csv'
    audit = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'rows_before': len(q), 'rows_after': len(q), 'excluded_rows': 0,
             'counts': q.groupby('experiment').size().to_dict(),
             'center': 'individual selected model members; no aggregate',
             'intervals': 'none: points are individual responses, not estimated multimodel means',
             'threshold': 'descriptive ±2 piControl noise SD; no new hypothesis test',
             'offsets': 'fixed model-specific vertical offsets to reduce overplotting; no value-axis jitter'}
    (ROOT/'data/processed/cmip6_wind_figure_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    make_figure(pd.read_csv(ROOT/'data/processed/cmip6_wnpac_wind_forced_vs_control.csv'))
