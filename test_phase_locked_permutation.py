#!/usr/bin/env python3
"""
Phase-locked permutation test for Halley commensurability.

THE CRITICAL DIFFERENCE FROM THE PREVIOUS TEST:

Previous synthetic-clone tests asked: "could a random sequence of δP produce
the same cancellation?" They had ~10% marginal p-values because many random
sequences cancel well over 15 orbits by pure chance.

This test asks a SHARPER question: "given Halley's observed δP values
AS A SET, is their SPECIFIC PAIRING with Jupiter-Saturn phases
significantly more predictive than a random pairing?"

We hold TWO things fixed:
  1. The 29 observed δP values (so cancellation properties are identical
     in every clone — we're not testing cancellation)
  2. The 29 real Jupiter-Saturn phases at the actual perihelion dates

We then PERMUTE the δP values across the phase sequence and measure R²
of the Jupiter+Saturn sinusoidal model on each permutation.

If Halley's coupling is real, the observed pairing should produce an R²
in the extreme upper tail of the permutation distribution.
If it isn't, permutations will routinely match or beat the observed R².

This test is immune to the "cancellation by chance" critique because
both sides have identical cancellation statistics. It isolates
PAIRING-SPECIFIC information.

Author: C. Baiget Orts (2026)
Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260423)

# ============================================================
# DATA
# ============================================================
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

N_PERMUTATIONS = 1_000_000

# Chirikov & Vecheslavov (1989) effective periods
P_J_DAYS = 4332.653
P_S_DAYS = 10759.362


# ============================================================
# SETUP
# ============================================================
def compute_inputs():
    """Returns:
       dP       : 29 observed period deviations (years)
       jup_phase: Jupiter phase at start of each period (degrees, 0..360)
       sat_phase: Saturn phase at start of each period (degrees, 0..360)
    """
    jd = np.array([p[1] for p in HALLEY_PERIHELIA])
    periods_yr = np.diff(jd) / 365.25
    dP = periods_yr - periods_yr.mean()

    # Phase at the START of period i = phase of perihelion i (i = 0..28)
    jd_ref = jd[-1]
    jup_phase = (((jd[:-1] - jd_ref) / P_J_DAYS) * 360.0) % 360.0
    sat_phase = (((jd[:-1] - jd_ref) / P_S_DAYS) * 360.0) % 360.0
    return dP, jup_phase, sat_phase


# ============================================================
# METRICS ON A (dP, phases) PAIRING
# ============================================================
def R2_model(dP, jup_phase, sat_phase):
    """R² of the linear model:
       dP ~ a sin(λ_J) + b cos(λ_J) + c sin(λ_S) + d cos(λ_S)
    """
    tj = np.radians(jup_phase)
    ts = np.radians(sat_phase)
    X = np.column_stack([np.sin(tj), np.cos(tj),
                         np.sin(ts), np.cos(ts)])
    coeffs, *_ = np.linalg.lstsq(X, dP, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((dP - pred) ** 2)
    ss_tot = np.sum((dP - dP.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def R2_jupiter_only(dP, jup_phase):
    """R² of Jupiter-only sinusoidal model."""
    tj = np.radians(jup_phase)
    X = np.column_stack([np.sin(tj), np.cos(tj)])
    coeffs, *_ = np.linalg.lstsq(X, dP, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((dP - pred) ** 2)
    ss_tot = np.sum((dP - dP.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def R2_saturn_only(dP, sat_phase):
    """R² of Saturn-only sinusoidal model."""
    ts = np.radians(sat_phase)
    X = np.column_stack([np.sin(ts), np.cos(ts)])
    coeffs, *_ = np.linalg.lstsq(X, dP, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((dP - pred) ** 2)
    ss_tot = np.sum((dP - dP.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ============================================================
# MAIN TEST
# ============================================================
def run_test():
    print("=" * 72)
    print("PHASE-LOCKED PERMUTATION TEST")
    print("=" * 72)
    print(f"\nNumber of permutations: {N_PERMUTATIONS:,}")
    print("Chirikov & Vecheslavov (1989) effective periods:")
    print(f"  P_Jupiter = {P_J_DAYS} days")
    print(f"  P_Saturn  = {P_S_DAYS} days")

    dP, jup_phase, sat_phase = compute_inputs()

    n = len(dP)
    print(f"\nInputs: n = {n} period deviations, paired with n = {n} phases")
    print(f"σ(δP) = {np.std(dP, ddof=1):.4f} yr")

    # --- Observed R² values ---
    R2_obs_JS = R2_model(dP, jup_phase, sat_phase)
    R2_obs_J  = R2_jupiter_only(dP, jup_phase)
    R2_obs_S  = R2_saturn_only(dP, sat_phase)

    print(f"\nOBSERVED R²:")
    print(f"  Jupiter+Saturn (4 params) = {R2_obs_JS:.4f}")
    print(f"  Jupiter only   (2 params) = {R2_obs_J:.4f}")
    print(f"  Saturn only    (2 params) = {R2_obs_S:.4f}")

    # --- Permutation loop ---
    print(f"\nRunning {N_PERMUTATIONS:,} permutations...")

    R2_perm_JS = np.empty(N_PERMUTATIONS)
    R2_perm_J  = np.empty(N_PERMUTATIONS)
    R2_perm_S  = np.empty(N_PERMUTATIONS)

    # Vectorise: precompute phase design matrix (fixed across permutations)
    tj = np.radians(jup_phase)
    ts = np.radians(sat_phase)
    X_JS = np.column_stack([np.sin(tj), np.cos(tj), np.sin(ts), np.cos(ts)])
    X_J  = np.column_stack([np.sin(tj), np.cos(tj)])
    X_S  = np.column_stack([np.sin(ts), np.cos(ts)])

    # Use pseudoinverse once (phases are fixed)
    pinv_JS = np.linalg.pinv(X_JS)
    pinv_J  = np.linalg.pinv(X_J)
    pinv_S  = np.linalg.pinv(X_S)

    ss_tot_const = np.sum((dP - dP.mean()) ** 2)

    for k in range(N_PERMUTATIONS):
        dP_perm = RNG.permutation(dP)

        coeffs_JS = pinv_JS @ dP_perm
        pred_JS = X_JS @ coeffs_JS
        R2_perm_JS[k] = 1.0 - np.sum((dP_perm - pred_JS) ** 2) / ss_tot_const

        coeffs_J = pinv_J @ dP_perm
        pred_J = X_J @ coeffs_J
        R2_perm_J[k] = 1.0 - np.sum((dP_perm - pred_J) ** 2) / ss_tot_const

        coeffs_S = pinv_S @ dP_perm
        pred_S = X_S @ coeffs_S
        R2_perm_S[k] = 1.0 - np.sum((dP_perm - pred_S) ** 2) / ss_tot_const

        if (k + 1) % 100_000 == 0:
            print(f"  {k+1:,} / {N_PERMUTATIONS:,} done")

    # --- p-values: probability that a permutation R² ≥ observed ---
    def fmt(p):
        if p == 0:
            return f"< {1/N_PERMUTATIONS:.1e}"
        return f"{p:.5f}"

    p_JS = np.mean(R2_perm_JS >= R2_obs_JS)
    p_J  = np.mean(R2_perm_J  >= R2_obs_J)
    p_S  = np.mean(R2_perm_S  >= R2_obs_S)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"\nOne-sided p-values (P[permuted R² ≥ observed R²]):")
    print(f"  Jupiter+Saturn: {fmt(p_JS)}   (observed R² = {R2_obs_JS:.4f})")
    print(f"  Jupiter only:   {fmt(p_J)}    (observed R² = {R2_obs_J:.4f})")
    print(f"  Saturn only:    {fmt(p_S)}    (observed R² = {R2_obs_S:.4f})")

    print(f"\nPermutation R² distribution (Jupiter+Saturn):")
    print(f"  mean     = {R2_perm_JS.mean():.4f}")
    print(f"  median   = {np.median(R2_perm_JS):.4f}")
    print(f"  std      = {R2_perm_JS.std():.4f}")
    print(f"  95th pct = {np.percentile(R2_perm_JS, 95):.4f}")
    print(f"  99th pct = {np.percentile(R2_perm_JS, 99):.4f}")
    print(f"  max      = {R2_perm_JS.max():.4f}")

    print(f"\nINTERPRETATION")
    print(f"  Observed R²(J+S) sits at the {(1 - p_JS) * 100:.2f}th percentile")
    print(f"  of permutations that preserve the δP distribution but break")
    print(f"  the δP–phase pairing.")
    if p_JS < 0.05:
        print(f"  → The specific pairing of δP with J+S phases is NOT random:")
        print(f"    there is phase-dependent information in Halley's periods.")
    else:
        print(f"  → The specific pairing of δP with J+S phases is consistent")
        print(f"    with chance. No detectable phase-locked coupling.")

    # --- Plots ---
    make_plots(R2_perm_JS, R2_perm_J, R2_perm_S,
               R2_obs_JS,  R2_obs_J,  R2_obs_S)

    print(f"\nFigure saved: phase_locked_permutation.png")
    print("=" * 72)


def make_plots(R2_JS, R2_J, R2_S, obs_JS, obs_J, obs_S):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#0d1117')

    panels = [
        (axes[0], R2_JS, obs_JS, "R²  (Jupiter + Saturn)",     "4-parameter model"),
        (axes[1], R2_J,  obs_J,  "R²  (Jupiter only)",          "2-parameter model"),
        (axes[2], R2_S,  obs_S,  "R²  (Saturn only)",           "2-parameter model"),
    ]

    for ax, data, obs, xlabel, subtitle in panels:
        ax.set_facecolor('#0d1117')
        ax.hist(data, bins=100, color='#58a6ff', alpha=0.75, edgecolor='none')
        ax.axvline(obs, color='#f85149', linewidth=2.2,
                   label=f'observed = {obs:.3f}')
        # Shade the tail ≥ observed
        p = np.mean(data >= obs)
        ax.set_xlabel(xlabel, color='#c9d1d9')
        ax.set_ylabel('count', color='#c9d1d9')
        ax.set_title(f"{subtitle}\n"
                     f"observed = {obs:.3f}, p = {p:.4f}",
                     color='#e6edf3', fontsize=10)
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.legend(facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#c9d1d9', fontsize=9)

    fig.suptitle(
        f"Phase-locked permutation test: {len(R2_JS):,} permutations of δP "
        f"against fixed phases",
        color='#e6edf3', fontsize=12
    )
    plt.tight_layout()
    plt.savefig('phase_locked_permutation.png', dpi=140,
                facecolor='#0d1117', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_test()
