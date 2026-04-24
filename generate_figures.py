#!/usr/bin/env python3
"""
generate_figures.py
===================
Generates all five figures for:
  "Comet Halley Completes 15 Orbits in 1,151 Years:
   Evidence for Dynamical Stabilization by Jupiter and Saturn"

Output files (PDF + PNG at 200 dpi):
  fig1_periods_convergence.pdf / .png
  fig2_cumulative_perturbation.pdf / .png
  fig3_jupiter_correlation.pdf / .png
  fig4_residues_comparison.pdf / .png
  fig5_rolling_prediction.pdf / .png

Requirements: numpy, matplotlib
Optional:     skyfield  (for Fig. 3 with DE441/DE440 ephemeris;
                         falls back to Chirikov phases if absent)

Install:  pip install numpy matplotlib
          pip install skyfield          # optional

Author: C. Baiget Orts (2026)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED DATA
# ═══════════════════════════════════════════════════════════════════════════════

HALLEY_PERIHELIA = [
    (-239, 1633907.6180), (-163, 1661838.0660), (-86,  1689863.9617),
    (-11,  1717323.3485), (66,   1745189.4601), (141,  1772638.9340),
    (218,  1800819.2235), (295,  1828915.8984), (374,  1857707.8424),
    (451,  1885963.7491), (530,  1914909.6300), (607,  1942837.9758),
    (684,  1971164.2668), (760,  1998788.1713), (837,  2026830.7700),
    (912,  2054365.1743), (989,  2082538.1876), (1066, 2110493.4340),
    (1145, 2139377.0609), (1222, 2167664.3229), (1301, 2196546.0819),
    (1378, 2224686.1872), (1456, 2253022.1326), (1531, 2280492.7385),
    (1607, 2308304.0406), (1682, 2335655.7807), (1759, 2363592.5608),
    (1835, 2391598.9387), (1910, 2418781.6777), (1986, 2446470.9518),
]

HALLEY_YEARS = [p[0] for p in HALLEY_PERIHELIA]
HALLEY_JD    = [p[1] for p in HALLEY_PERIHELIA]

T_STAR    = 1151.0
P_EXACT   = T_STAR / 15          # 76.7333… yr
JD_PER_YR = 365.25

periods   = np.array([(HALLEY_JD[i+1] - HALLEY_JD[i]) / JD_PER_YR
                      for i in range(len(HALLEY_JD) - 1)])
mean_P    = np.mean(periods)
delta_P   = periods - mean_P     # yr
sigma_P   = np.std(delta_P, ddof=1)

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED STYLE
# ═══════════════════════════════════════════════════════════════════════════════

BG    = '#0d1117'
FG    = '#c9d1d9'
GRID  = '#21262d'
BLUE  = '#58a6ff'
RED   = '#f85149'
GREY  = '#8b949e'
GREEN = '#3fb950'
AMBER = '#d29922'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': BG,
    'axes.edgecolor':   GRID, 'axes.labelcolor': FG,
    'xtick.color':      GREY, 'ytick.color':     GREY,
    'text.color':       FG,   'grid.color':      GRID,
    'grid.linewidth':   0.5,  'font.size':       11,
    'axes.titlesize':   12,   'axes.titleweight': 'normal',
})

def save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=200, facecolor=BG)
    print(f'  Saved: {stem}.pdf / .png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Individual periods + running mean convergence
# ═══════════════════════════════════════════════════════════════════════════════

def fig1():
    running_mean = np.array([
        (HALLEY_JD[n] - HALLEY_JD[0]) / (n * JD_PER_YR)
        for n in range(1, len(HALLEY_JD))
    ])
    n_vals = np.arange(2, len(HALLEY_JD) + 1)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    fig.subplots_adjust(left=0.10, right=0.94, top=0.90, bottom=0.12)

    ax.plot(n_vals, running_mean, 'o-', color=BLUE,
            ms=5, lw=1.8, zorder=3, label='Running mean period')
    ax.axhline(P_EXACT, color=RED, lw=2.0, ls=':',
               label='$T^*/15 = ' + f'{P_EXACT:.3f}' + '$ yr (commensurable)')

    # Annotate final value
    label_mean = '$\\bar{P} = ' + f'{mean_P:.3f}' + '$ yr  (7.4 d below $T^*/15$)'
    ax.annotate(label_mean,
                xy=(30, mean_P), xytext=(21, mean_P - 0.38),
                color=BLUE, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))

    # Highlight the final data point in green to mark the convergence
    ax.plot(n_vals[-1], running_mean[-1], 'o', color=GREEN, ms=8, zorder=5)

    ax.set_xlim(1.5, 32)
    ax.set_ylim(75.4, 78.3)
    ax.set_xlabel('Number of apparitions included')
    ax.set_ylabel('Running mean period (yr)')
    ax.set_title("Convergence of Halley's mean orbital period toward $T^*/15$")
    ax.legend(loc='upper right', fontsize=10,
              facecolor='#161b22', edgecolor=GRID, framealpha=0.9)
    ax.grid(True, zorder=1)
    ax.tick_params(which='both', direction='in')

    save(fig, 'fig1_periods_convergence')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Cumulative perturbation sum vs random-walk envelope
# ═══════════════════════════════════════════════════════════════════════════════

def fig2():
    cumsum_days = np.cumsum(delta_P) * JD_PER_YR
    n_vals      = np.arange(1, len(delta_P) + 1)
    rw_upper    =  sigma_P * np.sqrt(n_vals) * JD_PER_YR
    rw_lower    = -sigma_P * np.sqrt(n_vals) * JD_PER_YR

    val_15 = cumsum_days[14]
    ratio  = abs(val_15) / rw_upper[14]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.11)

    ax.fill_between(n_vals, rw_lower, rw_upper, color=GREY, alpha=0.30,
                    zorder=1, label=r'$\pm\,\sigma\sqrt{n}$ random-walk envelope')
    ax.plot(n_vals, rw_upper, color=GREY, lw=0.8, alpha=0.6, zorder=2)
    ax.plot(n_vals, rw_lower, color=GREY, lw=0.8, alpha=0.6, zorder=2)
    ax.plot(n_vals, cumsum_days, 'o-', color=BLUE, ms=4.5, lw=1.8,
            zorder=3, label=r'$\sum_{i=1}^{n}\,\delta P_i$')
    ax.axhline(0,  color=FG,    lw=0.8, alpha=0.4, zorder=2)
    ax.axvline(15, color=GREEN, lw=1.4, ls=':', zorder=2,
               label='$n = 15$ (one commensurable cycle)')

    ax.annotate(f'$n=15$: {abs(val_15):.0f} days\n= {ratio*100:.1f}% of random walk',
                xy=(15, 0), xytext=(22, -1400),
                fontsize=9.5, color=GREEN,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2,
                                connectionstyle='arc3,rad=-0.2'))
    ax.annotate(f'$n=29$: {abs(cumsum_days[-1]):.1f} days\n(exact cancellation)',
                xy=(29, cumsum_days[-1]), xytext=(22, -480),
                fontsize=9.5, color=AMBER,
                arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.2))

    ax.set_xlim(0.5, 30.5)
    ax.set_xlabel('Number of orbits $n$')
    ax.set_ylabel(r'Cumulative $\sum \delta P_i$ (days)')
    ax.set_title('Perturbation cancellation over orbital cycles')
    ax.legend(loc='upper left', fontsize=9.5,
              facecolor='#161b22', edgecolor=GRID, framealpha=0.9)
    ax.grid(True, zorder=0)
    ax.tick_params(which='both', direction='in')

    save(fig, 'fig2_cumulative_perturbation')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Angular residues comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig4():
    # Empirical residues from DE441 metric (Baiget Orts 2026a, Table 1)
    planet_residues = [
        ('Mercury',  3.5),
        ('Neptune',  5.2),
        ('Earth',    6.6),
        ('Jupiter', 11.8),
        ('Mars',    15.1),
        ('Venus',   20.1),
        ('Saturn',  26.5),
    ]
    htcs = [
        ('12P/Pons–Brooks\n(1385–2024)',   70.942),
        ('12P/Pons–Brooks\n(1812–2024)',   70.533),
        ('27P/Crommelin\n(1928–2011)',     27.558),
        ('55P/Tempel–Tuttle\n(901–1998)', 33.225),
    ]

    def res_htc(P):
        return abs((T_STAR/P - round(T_STAR/P)) * 360)

    halley_res_val = abs((T_STAR / mean_P - round(T_STAR / mean_P)) * 360)

    ordered = (
        [('1P/Halley', halley_res_val)] +
        sorted(planet_residues, key=lambda x: x[1]) +
        [(n, res_htc(P)) for n, P in htcs] +
        [('Uranus', 108.3)]
    )

    names  = [b[0] for b in ordered]
    resids = [b[1] for b in ordered]
    colors = []
    htc_names = {h[0] for h in htcs}
    for name, _ in ordered:
        if name == '1P/Halley':         colors.append(RED)
        elif name == 'Uranus':          colors.append('#484f58')
        elif name in htc_names:         colors.append(AMBER)
        else:                           colors.append(BLUE)

    n     = len(ordered)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG)
    fig.subplots_adjust(left=0.26, right=0.96, top=0.92, bottom=0.09)

    ax.barh(y_pos, resids, color=colors, alpha=0.85, height=0.65, zorder=3)
    ax.axhline(n - 1.5, color=GREY, lw=0.8, ls='--', alpha=0.5)
    for i, val in enumerate(resids):
        ax.text(val + 1.5, i, f'{val:.1f}°', va='center', color=FG, fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel('Angular residue $|\\Delta\\theta|$ (°)')
    ax.set_title('Angular residues at $T^* = 1{,}151$ yr for Solar System bodies')
    ax.set_xlim(0, 155)
    ax.grid(True, axis='x', zorder=1)
    ax.tick_params(which='both', direction='in')
    ax.invert_yaxis()

    legend_handles = [
        mpatches.Patch(color=RED,     label='1P/Halley'),
        mpatches.Patch(color=BLUE,    label='Planets (participants)'),
        mpatches.Patch(color=AMBER,   label='Other HTCs'),
        mpatches.Patch(color='#484f58', label='Uranus (non-participant)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              facecolor='#161b22', edgecolor=GRID, framealpha=0.9)

    save(fig, 'fig3_residues_comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Arithmetic landscape: residues vs period
# ═══════════════════════════════════════════════════════════════════════════════

def fig6():
    P_vals   = np.linspace(20, 200, 200000)
    residues = np.array([abs((T_STAR / P - round(T_STAR / P)) * 360)
                         for P in P_vals])

    # Local minima < 5°
    minima = []
    for i in range(1, len(P_vals) - 1):
        if (residues[i] < residues[i-1] and
                residues[i] < residues[i+1] and
                residues[i] < 5.0):
            minima.append((P_vals[i], residues[i], round(T_STAR / P_vals[i])))

    # Neighbourhood [60–90 yr] for zoom panel
    P_zoom = P_vals[(P_vals >= 57) & (P_vals <= 94)]
    r_zoom = residues[(P_vals >= 57) & (P_vals <= 94)]

    # Key neighbours
    nbhd_minima = [(P, r, N) for P, r, N in minima if 57 <= P <= 94]

    halley_res  = abs((T_STAR / mean_P - round(T_STAR / mean_P)) * 360)
    other_htcs  = [
        ('12P (1385–2024)', 70.942), ('12P (1812–2024)', 70.533),
        ('55P',             33.225), ('27P',             27.558),
    ]

    fig = plt.figure(figsize=(12, 6), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, wspace=0.32,
                            top=0.88, bottom=0.12, left=0.08, right=0.97)

    # ── Left panel: full HTC range [20–200 yr] ────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(P_vals, residues, color=GREY, lw=0.6, alpha=0.7, zorder=2)

    # Mark all local minima as small dots
    for P, r, N in minima:
        ax1.plot(P, r, 'o', color=GREY, ms=3, alpha=0.5, zorder=3)

    # Mark Halley
    ax1.plot(mean_P, halley_res, '*', color=RED, ms=22, zorder=5,
             label=f'1P/Halley ($N=15$, $|\\Delta\\theta|={halley_res:.2f}°$)')
    ax1.annotate('1P/Halley', xy=(mean_P, halley_res),
                 xytext=(mean_P + 12, halley_res + 8),
                 color=RED, fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

    # Mark other HTCs
    for name, P in other_htcs:
        r = abs((T_STAR / P - round(T_STAR / P)) * 360)
        ax1.plot(P, r, 'D', color=AMBER, ms=6, zorder=4)
        ax1.annotate(name, xy=(P, r), xytext=(P + 3, r + 4),
                     fontsize=7.5, color=AMBER)

    ax1.set_xlim(18, 202)
    ax1.set_ylim(-2, 185)
    ax1.set_xlabel('Orbital period $P$ (yr)')
    ax1.set_ylabel('Angular residue $|\\Delta\\theta|$ (°)')
    ax1.set_title('Full HTC range [20–200 yr]')
    ax1.legend(loc='upper right', fontsize=9,
               facecolor='#161b22', edgecolor=GRID, framealpha=0.9)
    ax1.grid(True, zorder=1)
    ax1.tick_params(which='both', direction='in')

    # ── Right panel: zoom [60–90 yr] ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(P_zoom, r_zoom, color=GREY, lw=0.8, alpha=0.8, zorder=2)

    # Mark and label each minimum in neighbourhood
    # Stagger annotation heights to avoid overlap
    annot_heights = {}
    for i, (P, r, N) in enumerate(sorted(nbhd_minima, key=lambda x: x[0])):
        annot_heights[N] = 0.6 + (i % 2) * 1.8  # alternate high/low

    for P, r, N in nbhd_minima:
        color = RED if abs(P - mean_P) < 0.5 else BLUE
        marker = '*' if abs(P - mean_P) < 0.5 else 'o'
        ms = 18 if abs(P - mean_P) < 0.5 else 7
        ax2.plot(P, r, marker, color=color, ms=ms, zorder=5)
        y_text = annot_heights[N]
        ax2.annotate(f'$N={N}$\n{P:.2f} yr\n{r:.3f}°',
                     xy=(P, r), xytext=(P, y_text),
                     fontsize=8, color=color, ha='center',
                     arrowprops=dict(arrowstyle='->', color=color,
                                     lw=0.8, shrinkA=4))

    # Halley observed mean (vertical line)
    ax2.axvline(mean_P, color=RED, lw=1.2, ls='--', alpha=0.5,
                label=f'Halley $\\bar{{P}} = {mean_P:.3f}$ yr')

    # Shade the ±2 yr variability band
    ax2.axvspan(mean_P - 2, mean_P + 2, color=RED, alpha=0.07,
                label='±2 yr period range')

    ax2.set_xlim(56, 95)
    ax2.set_ylim(-0.5, 5.5)
    ax2.set_xlabel('Orbital period $P$ (yr)')
    ax2.set_ylabel('Angular residue $|\\Delta\\theta|$ (°)')
    ax2.set_title("Zoom: Halley's neighbourhood [60–90 yr]")
    ax2.legend(loc='upper right', fontsize=8.5,
               facecolor='#161b22', edgecolor=GRID, framealpha=0.9)
    ax2.grid(True, zorder=1)
    ax2.tick_params(which='both', direction='in')

    # Annotation box
    n_min_full = len(minima)
    ax1.text(0.03, 0.97,
             f'{n_min_full} minima $< 5°$ in [20, 200] yr\n'
             f'≈ one every {180/n_min_full:.1f} yr of period',
             transform=ax1.transAxes, fontsize=8.5, va='top',
             color=GREY, bbox=dict(boxstyle='round,pad=0.4',
                                   fc='#161b22', ec=GRID, alpha=0.85))

    fig.suptitle('Arithmetic landscape of angular residues at $T^* = 1{,}151$ yr',
                 fontsize=12, y=0.97)

    save(fig, 'fig4_arithmetic_landscape')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Rolling prediction: temporal error series
#
# x-axis : observed perihelion year as decimal (e.g. "1910.3")
# y-axis : prediction error in years (predicted - observed)
# Two series: running mean of all prior periods vs fixed T*/15
# Warm-up: first 3 apparitions (forecasts from apparition 4 onward)
# ═══════════════════════════════════════════════════════════════════════════════

def fig5():
    T15_days = P_EXACT * JD_PER_YR
    WARMUP   = 3

    def jd_to_yr(jd):
        return (jd - 1721045.0) / JD_PER_YR

    yr_float = [jd_to_yr(j) for j in HALLEY_JD]

    x_obs, err_mean_list, err_t15_list = [], [], []
    for n in range(WARMUP, len(HALLEY_JD)):
        known    = HALLEY_JD[:n]
        mean_p_d = (known[-1] - known[0]) / (n - 1)
        em = (known[-1] + mean_p_d - HALLEY_JD[n]) / JD_PER_YR
        et = (known[-1] + T15_days - HALLEY_JD[n]) / JD_PER_YR
        x_obs.append(yr_float[n])
        err_mean_list.append(em)
        err_t15_list.append(et)

    x_obs        = np.array(x_obs)
    err_mean_arr = np.array(err_mean_list)
    err_t15_arr  = np.array(err_t15_list)

    rms_m  = np.sqrt(np.mean(err_mean_arr**2))
    rms_t  = np.sqrt(np.mean(err_t15_arr**2))
    mae_m  = np.mean(np.abs(err_mean_arr))
    mae_t  = np.mean(np.abs(err_t15_arr))
    wins_t = int(np.sum(np.abs(err_t15_arr) < np.abs(err_mean_arr)))
    n_tot  = len(err_mean_arr)

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=BG)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.16)

    ax.axhline(0, color=FG, lw=0.9, alpha=0.30, zorder=2)
    ax.axhspan(-1, 1, alpha=0.04, color=BLUE, zorder=1)

    ax.plot(x_obs, err_mean_arr, 'o-',  color=BLUE, lw=1.5, ms=4.5,
            zorder=4, label='Running mean')
    ax.plot(x_obs, err_t15_arr,  's--', color=RED,  lw=1.5, ms=4.5,
            zorder=4, label=r'$T^*/15$ (no Halley data)')

    tick_pos  = x_obs[::3]
    tick_labs = [f'{v:.1f}' for v in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labs, rotation=40, ha='right', fontsize=8)

    ax.set_ylabel('Prediction error (yr)\n[predicted\u2212observed]', fontsize=10)
    ax.set_xlabel('Observed perihelion year', fontsize=10)
    ax.set_ylim(-3.8, 3.8)
    ax.grid(True, zorder=1)
    ax.tick_params(which='both', direction='in')

    stat = (f'Running mean  \u2014 RMS {rms_m:.2f} yr,  MAE {mae_m:.2f} yr\n'
            f'$T^*/15$ fixed \u2014 RMS {rms_t:.2f} yr,  MAE {mae_t:.2f} yr\n'
            f'$T^*/15$ closer: {wins_t}/{n_tot} forecasts\n'
            f'Warm-up: first {WARMUP} apparitions '
            f'(through {yr_float[WARMUP-1]:.1f})')
    ax.text(0.01, 0.97, stat, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='left', color=FG, family='monospace',
            bbox=dict(boxstyle='round,pad=0.45', fc=BG, ec=GRID, alpha=0.92))

    ax.legend(loc='upper right', fontsize=9,
              facecolor=BG, edgecolor=GRID, framealpha=0.92)

    ax.set_title(
        r'One-step-ahead perihelion predictions: error relative to observation'
        '\n'
        r'Both predictors bounded within $\pm$3 yr across 2,200 yr --- no secular drift',
        fontsize=10, pad=6)

    save(fig, 'fig5_rolling_prediction')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    tasks = [
        ('Fig 1 — Running mean convergence',     fig1),
        ('Fig 2 — Cumulative perturbation sum',  fig2),
        ('Fig 3 — Angular residues comparison',  fig4),
        ('Fig 4 — Arithmetic landscape',         fig6),
        ('Fig 5 — Rolling prediction errors',    fig5),
    ]

    print('Generating figures for Halley / 1151-yr paper')
    print('=' * 50)
    for label, fn in tasks:
        print(f'\n{label}')
        fn()

    print('\n' + '=' * 50)
    print('Done. 10 files written (5 x PDF + 5 x PNG).')
