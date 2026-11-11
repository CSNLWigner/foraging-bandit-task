"""
Plotting functions of Supplementary Fig. 10: session-wise parameters of the GLM of the neural value update.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from popy.config import FIGS_PATH


def plot_betas(res_df):
    # fontsize 8
    plt.rcParams.update({'font.size': 8})

    all_var_to_plot = [
        (
            "slope_pos",
            "slope_pos_withinbounds",
            r"Slope $\beta$ of $V(t)$ regressor" + "\n(positive outcomes)",
        ),
        (
            "slope_neg",
            "slope_neg_withinbounds",
            r"Slope $\beta$ of $V(t)$ regressor" + "\n(negative outcomes)",
        ),
    ]

    for var_to_plot, signif_slope, var_name in all_var_to_plot:
        monkeys   = res_df['monkey'].unique()
        subregions = res_df['subregion'].unique()
        n_monks   = len(monkeys)
        n_subs    = len(subregions)

        fig, axs = plt.subplots(n_monks, n_subs, figsize=(1.3 * n_subs, 1.8 * n_monks), sharey=True)
        if axs.ndim == 1:
            axs = axs[np.newaxis, :]

        VIOLIN_START = 0.38
        VIOLIN_WIDTH = 0.30

        for m, monkey in enumerate(monkeys):
            for s, subregion in enumerate(subregions):
                ax = axs[m, s]
                subset = res_df[(res_df['monkey'] == monkey) & (res_df['subregion'] == subregion)]
                vals = subset[var_to_plot].dropna().values
                signifs = subset[signif_slope].dropna().values
                
                if len(vals) == 0:
                    ax.set_visible(False)
                    continue

                kde = gaussian_kde(vals, bw_method='scott')
                y_range = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)
                density  = kde(y_range)
                density  = density / density.max() * VIOLIN_WIDTH

                ax.fill_betweenx(y_range, VIOLIN_START, VIOLIN_START + density,
                                alpha=0.5, color='steelblue', linewidth=0)
                ax.plot(VIOLIN_START + density, y_range, color='steelblue', linewidth=0.8)

                ax.boxplot(vals, positions=[0.25], widths=0.15,
                        vert=True, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5),
                        boxprops=dict(facecolor='steelblue', alpha=0.5),
                        whiskerprops=dict(color='black', linewidth=0.8),
                        capprops=dict(color='black', linewidth=0.8),
                        flierprops=dict(marker='', linestyle='none'),
                        showfliers=False)

                rng    = np.random.default_rng(42)
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))

                ax.scatter(jitter[signifs],  vals[signifs],
                        alpha=0.7, s=12, marker='o',
                        color='steelblue', zorder=3)
                ax.scatter(jitter[~signifs], vals[~signifs],
                        alpha=0.5, s=12, marker='o', facecolors='none',
                        color='steelblue', linewidths=0.7, zorder=3)

                ax.axhline(0, color='black',   linestyle='--', linewidth=0.8, zorder=1)
                ax.axhline(1, color='black', linestyle='--', linewidth=0.8, zorder=1)

                n_sig = signifs.sum()
                ax.set_title(f"{monkey.upper()} – {subregion}\n(n. sig.={n_sig}/{len(vals)})", fontsize=7)
                ax.set_ylabel(var_name, fontsize=7)
                ax.tick_params(labelsize=7)
                ax.set_xlim(-0.3, VIOLIN_START + VIOLIN_WIDTH + 0.05)
                ax.set_ylim(-.5, 1.01)
                ax.set_xticks([])
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        plt.tight_layout()
        # save figure
        plt.savefig(os.path.join(FIGS_PATH, f'figS10_{var_to_plot}.svg'), format='svg', dpi=300, bbox_inches='tight')
        plt.show()


def plot_intercept_differences(res_df):
    plt.rcParams.update({'font.size': 8})

    monkeys    = res_df['monkey'].unique()
    subregions = res_df['subregion'].unique()
    n_monks    = len(monkeys)
    n_subs     = len(subregions)

    fig, axs = plt.subplots(n_monks, n_subs, figsize=(1.3 * n_subs, 1.8 * n_monks), sharey=True)
    if axs.ndim == 1:
        axs = axs[np.newaxis, :]

    VIOLIN_START = 0.38
    VIOLIN_WIDTH = 0.30

    for m, monkey in enumerate(monkeys):
        for s, subregion in enumerate(subregions):
            ax = axs[m, s]
            subset = res_df[(res_df['monkey'] == monkey) & (res_df['subregion'] == subregion)]
            vals    = (subset['intercept_pos'] - subset['intercept_neg']).values
            signifs = subset['intercept_pos_higher'].values
            mask    = np.isfinite(vals)
            vals    = vals[mask]
            signifs = signifs[mask]

            if len(vals) == 0:
                ax.set_visible(False)
                continue

            # ── 1. Half-violin (KDE) ─────────────────────────────────────────────
            kde     = gaussian_kde(vals, bw_method='scott')
            y_range = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 300)
            density = kde(y_range)
            density = density / density.max() * VIOLIN_WIDTH

            ax.fill_betweenx(y_range, VIOLIN_START, VIOLIN_START + density,
                            alpha=0.5, color='steelblue', linewidth=0)
            ax.plot(VIOLIN_START + density, y_range, color='steelblue', linewidth=0.8)

            # ── 2. Boxplot ───────────────────────────────────────────────────────
            ax.boxplot(vals, positions=[0.25], widths=0.15,
                    vert=True, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(facecolor='steelblue', alpha=0.5),
                    whiskerprops=dict(color='black', linewidth=0.8),
                    capprops=dict(color='black', linewidth=0.8),
                    showfliers=False)

            # ── 3. Jittered scatter: filled circle = sig, open circle = nonsig ──
            rng    = np.random.default_rng(42)
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))

            ax.scatter(jitter[signifs],  vals[signifs],
                    alpha=0.7, s=12, marker='o',
                    color='steelblue', zorder=3)
            ax.scatter(jitter[~signifs], vals[~signifs],
                    alpha=0.5, s=12, marker='o', facecolors='none',
                    edgecolors='steelblue', linewidths=0.7, zorder=3)

            # ── Zero reference line only ─────────────────────────────────────────
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, zorder=1)

            # ── Cosmetics ────────────────────────────────────────────────────────
            n_sig = signifs.sum()
            ax.set_title(f"{monkey.upper()} – {subregion}\n(n. sig.={n_sig}/{len(vals)})", fontsize=7)
            ax.set_ylabel('Intercept difference\n(positive - negative)', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_xlim(-0.3, VIOLIN_START + VIOLIN_WIDTH + 0.05)
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_PATH, f'figS10c_intercept_diff.svg'), format='svg', dpi=300, bbox_inches='tight')
    plt.show()
