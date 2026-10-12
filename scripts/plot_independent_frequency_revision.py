"""Render frozen summer counts without rerunning trend or mechanism analyses.

Question: does external frequency agreement imply catalogue interchangeability?
Panel a: short-overlap agreement and mean bias; b: longer coverage boundary.
Python quantitative grid, 183 x 115 mm, editable PDF/SVG and PNG preview.
All 44 short and 88 long-overlap values shown. Published 1979/2024 CRA-40
values remain in the input, outside the current catalogue's 1980--2023 overlap.
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
OUT = ROOT/'data/processed'
FROZEN = OUT/'published_independent_swcv_counts.csv'
SEASONAL = OUT/'seasonal_joint_metrics_1980_2023.csv'
SHORT_SOURCES = ['yearbook', 'cra40_fig2', 'era5_sun_fig2', 'cnn_era5_frequency']
COLORS = {'yearbook': '#333333', 'cra40_fig2': '#558899',
          'era5_sun_fig2': '#777777', 'cnn_era5_frequency': '#B75D17', 'cra40_fig3': '#558899'}

def prepare_sources(frozen, seasonal):
    if set(frozen.source) != set(SHORT_SOURCES+['cra40_fig3']) or len(frozen) != 90:
        raise ValueError('Unexpected published source allocation')
    if frozen.duplicated(['season_year', 'source']).any():
        raise ValueError('Duplicate published year-source observations')
    if not np.isfinite(frozen.frequency).all():
        raise ValueError('Missing published observations may not be silently removed')
    short = frozen.loc[frozen.source.isin(SHORT_SOURCES)].pivot(index='season_year', columns='source', values='frequency')
    if list(short.index) != list(range(2012, 2023)) or set(short.columns) != set(SHORT_SOURCES) or short.isna().any().any():
        raise ValueError('Expected complete 11-summer four-series comparison')
    cnn = seasonal.loc[seasonal.season.eq('JJA'), ['season_year', 'frequency']].set_index('season_year')
    if cnn.index.duplicated().any() or sorted(cnn.index) != list(range(1980, 2024)) or not np.isfinite(cnn.frequency).all():
        raise ValueError('Expected complete 44-summer current-catalogue series')
    if not np.array_equal(short.cnn_era5_frequency.to_numpy(), cnn.loc[short.index, 'frequency'].to_numpy()):
        raise ValueError('Frozen short current counts disagree with seasonal table')
    cra = frozen.loc[frozen.source.eq('cra40_fig3'), ['season_year', 'frequency']].set_index('season_year')
    if sorted(cra.index) != list(range(1979, 2025)):
        raise ValueError('Expected complete published long series, including non-overlap endpoints')
    long = cnn.rename(columns={'frequency': 'cnn_era5_frequency'}).join(cra.rename(columns={'frequency': 'cra40_fig3'}), validate='one_to_one')
    if long.isna().any().any():
        raise ValueError('Incomplete common overlap')
    if tuple(short[['yearbook', 'cra40_fig2', 'era5_sun_fig2']].sum()) != (233, 271, 256):
        raise ValueError('Published cross-checked totals changed')
    return short.sort_index(), long.sort_index()

def make_figure(short, long):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 1, figsize=(7.2047244094, 4.5275590551))
    fig.subplots_adjust(left=.11, right=.76, top=.88, bottom=.15, hspace=.60)
    names = {'yearbook': 'Yearbook', 'cra40_fig2': 'CRA-40',
             'era5_sun_fig2': 'ERA5 (published)', 'cnn_era5_frequency': 'CNN/ERA5', 'cra40_fig3': 'CRA-40 (Fig. 3)'}
    labels_y = {'yearbook': 12, 'cra40_fig2': 23, 'era5_sun_fig2': 17, 'cnn_era5_frequency': 33}
    for name in SHORT_SOURCES:
        axes[0].plot(short.index, short[name], color=COLORS[name], lw=1.65 if name == 'cnn_era5_frequency' else 1,
                     marker='o', ms=2.7, ls='--' if name == 'era5_sun_fig2' else '-')
        end = float(short[name].iloc[-1])
        # Leader stops before text; no opaque masks or altered count coordinates.
        axes[0].plot([2022, 2022.12], [end, labels_y[name]], color=COLORS[name], lw=.6, clip_on=False)
        axes[0].text(2022.25, labels_y[name], names[name], color=COLORS[name], va='center', fontsize=8)
    axes[0].set_xlim(2011.8, 2022.05)
    axes[0].set_xticks([2012, 2014, 2016, 2018, 2020, 2022])
    axes[0].set_ylim(9, 52)
    axes[0].set_title('Short overlap: agreement does not remove count bias', loc='left', pad=11)
    bias = float((short.cnn_era5_frequency-short.yearbook).mean())
    axes[0].annotate('', xy=(1.28, short.cnn_era5_frequency.mean()),
                     xytext=(1.28, short.yearbook.mean()), xycoords=('axes fraction', 'data'),
                     textcoords=('axes fraction', 'data'), arrowprops={'arrowstyle': '<->', 'color': '#666666', 'lw': 1})
    axes[0].text(1.28, 42, f'+{bias:.2f}', transform=axes[0].get_yaxis_transform(),
                 ha='center', va='center', color='#555555', fontsize=8)
    for name in ['cra40_fig3', 'cnn_era5_frequency']:
        axes[1].plot(long.index, long[name], color=COLORS[name], lw=1.2)
        axes[1].text(2023.8, long[name].iloc[-1], names[name], color=COLORS[name], va='center', fontsize=8)
    axes[1].set_xlim(1979.5, 2023.2)
    axes[1].set_ylim(15, 65)
    axes[1].set_xticks([1980, 1990, 2000, 2010, 2020])
    axes[1].set_title('Long overlap: catalogue definitions remain different', loc='left', pad=11)
    for letter, ax in zip('ab', axes):
        ax.set_ylabel('JJA event count')
        ax.set_xlabel('JJA season year')
        ax.spines[['top', 'right']].set_visible(False)
        ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-13, 11),
                    textcoords='offset points', fontsize=9, weight='bold', va='bottom')
    fig.text(.11, .035, 'Grey arrow: difference of 2012–2022 means (+15.55 events/summer), not the 2022 endpoint gap.', fontsize=7.5)
    target = ROOT/'figures/independent_frequency_validation'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=target.with_suffix('.alignment.json'), require_panel_labels=True, strict=True)
    fig.savefig(target.with_suffix('.pdf'))
    fig.savefig(target.with_suffix('.svg'))
    fig.savefig(target.with_suffix('.png'), dpi=600)
    plt.close(fig)
    rows = pd.concat([short.rename_axis('season_year').reset_index().melt('season_year', var_name='source', value_name='frequency').assign(panel='a'),
                      long.rename_axis('season_year').reset_index().melt('season_year', var_name='source', value_name='frequency').assign(panel='b')], ignore_index=True)
    rows.to_csv(OUT/'independent_frequency_figure_source.csv', index=False)
    audit = {'source_sha256': hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
             'seasonal_sha256': hashlib.sha256(SEASONAL.read_bytes()).hexdigest(),
             'short_values': 44, 'long_values': 88, 'mean_bias_events_per_summer': bias,
             'nonoverlap_published_years': [1979, 2024], 'missing_values_dropped': 0,
             'means_compared': {'yearbook': float(short.yearbook.mean()), 'cnn_era5': float(short.cnn_era5_frequency.mean())},
             'scope': 'Summer counts only; no new tests, winter validation, equivalence or causal inference',
             'published_2022_inconsistency': {'fig2': int(short.loc[2022,'cra40_fig2']), 'fig3': int(long.loc[2022,'cra40_fig3'])}}
    (OUT/'independent_frequency_figure_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    make_figure(*prepare_sources(pd.read_csv(FROZEN), pd.read_csv(SEASONAL)))
