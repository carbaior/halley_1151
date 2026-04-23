#!/usr/bin/env python3
"""
Period-perturbation correlation test for Comet 1P/Halley.

Tests whether the deviations in Halley's orbital period correlate
with the positions of Jupiter and Saturn at each perihelion epoch.

Uses Skyfield + DE441 ephemeris for precise planetary positions.

Requires: numpy, matplotlib, skyfield
Install: pip install numpy matplotlib skyfield

On first run, Skyfield will download DE441 (~3.1 GB). 
If you prefer the smaller DE440 (~115 MB), change the ephemeris line.

Author: C. Baiget Orts (2026)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Check for Skyfield
# ============================================================
try:
    from skyfield.api import load
    from skyfield.data import mpc
    HAS_SKYFIELD = True
except ImportError:
    HAS_SKYFIELD = False
    print("WARNING: Skyfield not installed. Install with: pip install skyfield")
    print("Falling back to Chirikov & Vecheslavov (1989) phase data.\n")

# ============================================================
# DATA: Halley perihelion dates (chronological, oldest first)
# Julian Day Numbers from Chirikov & Vecheslavov (1989), Table 1
# ============================================================

HALLEY_PERIHELIA = [
    (-239, 1633907.6180),
    (-163, 1661838.0660),
    (-86,  1689863.9617),
    (-11,  1717323.3485),
    (66,   1745189.4601),
    (141,  1772638.9340),
    (218,  1800819.2235),
    (295,  1828915.8984),
    (374,  1857707.8424),
    (451,  1885963.7491),
    (530,  1914909.6300),
    (607,  1942837.9758),
    (684,  1971164.2668),
    (760,  1998788.1713),
    (837,  2026830.7700),
    (912,  2054365.1743),
    (989,  2082538.1876),
    (1066, 2110493.4340),
    (1145, 2139377.0609),
    (1222, 2167664.3229),
    (1301, 2196546.0819),
    (1378, 2224686.1872),
    (1456, 2253022.1326),
    (1531, 2280492.7385),
    (1607, 2308304.0406),
    (1682, 2335655.7807),
    (1759, 2363592.5608),
    (1835, 2391598.9387),
    (1910, 2418781.6777),
    (1986, 2446470.9518),
]

# ============================================================
# PLANETARY POSITIONS AT EACH PERIHELION
# ============================================================

def get_positions_skyfield():
    """Get Jupiter and Saturn ecliptic longitudes at each Halley perihelion
    using Skyfield + DE441 (split into two parts) or DE440."""
    
    import os
    
    print("Loading ephemeris...")
    ts = load.timescale()
    
    # DE441 is split into two files:
    #   de441_part-1.bsp covers JD 0625648.5 (-13200) to JD 2817872.5 (1969)
    #   de441_part-2.bsp covers JD 2287184.5 (1549)    to JD 7857520.5 (17191)
    # Overlap region: 1549-1969
    # Halley perihelia range: -239 (JD ~1633907) to 1986 (JD ~2446470)
    #   Part 1 covers -239 to 1969, Part 2 covers 1549 to far future
    #   So: perihelia before ~1549 need Part 1, perihelia after ~1969 need Part 2
    #   Perihelia in 1549-1969 can use either
    
    eph_part1 = None
    eph_part2 = None
    eph_single = None
    
    # Try loading split DE441 files
    import os
    
    # Search in current directory, home, and script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = ['.', script_dir, os.path.expanduser('~')]
    
    for path in search_paths:
        p1 = os.path.join(path, 'de441_part-1.bsp')
        p2 = os.path.join(path, 'de441_part-2.bsp')
        if os.path.exists(p1) and os.path.exists(p2):
            try:
                from skyfield.data.planets import _open as _
            except ImportError:
                pass
            try:
                from spktype21 import SPKType21
            except ImportError:
                pass
            try:
                from skyfield import api as sfapi
                eph_part1 = sfapi.load.open(p1)
                eph_part2 = sfapi.load.open(p2)
                # Test that subscripting works
                _ = eph_part1['sun']
                print(f"Using DE441 (two parts)")
                print(f"  Part 1: {p1}")
                print(f"  Part 2: {p2}")
                break
            except (Exception, TypeError) as e:
                # load.open returns a raw file, not a kernel
                # Try the correct method: load() with full path
                try:
                    from skyfield.api import Loader
                    loader = Loader(path)
                    eph_part1 = loader('de441_part-1.bsp')
                    eph_part2 = loader('de441_part-2.bsp')
                    _ = eph_part1['sun']
                    print(f"Using DE441 (two parts) via Loader")
                    print(f"  Part 1: {p1}")
                    print(f"  Part 2: {p2}")
                    break
                except Exception as e2:
                    print(f"  Failed method 2 from {path}: {e2}")
                    eph_part1 = None
                    eph_part2 = None
    
    # If split files not found, try single-file DE441 or DE440
    if eph_part1 is None:
        for fname in ['de441.bsp', 'de440.bsp']:
            try:
                eph_single = load(fname)
                if fname == 'de440.bsp':
                    print(f"Using {fname}")
                    print("WARNING: DE440 only covers 1549-2650 CE.")
                    print("Perihelion dates before 1549 will be skipped.")
                else:
                    print(f"Using {fname} (single file)")
                break
            except Exception:
                continue
        
        if eph_single is None and eph_part1 is None:
            print("ERROR: Could not load any ephemeris.")
            print("Place de441_part-1.bsp and de441_part-2.bsp in the current directory.")
            return None, None, None
    
    # JD boundary: Part 1 covers up to 1969-07-30 (JD ~2440431)
    #              Part 2 covers from 1549 onward
    # Halley 1910 perihelion: JD 2418781 → Part 1
    # Halley 1986 perihelion: JD 2446470 → Part 2 (after 1969)
    JD_BOUNDARY = 2440400.0  # ~1969 CE, safely before Part 1 ends
    
    def get_eph_for_jd(jd_value):
        """Return the appropriate ephemeris for a given Julian Day."""
        if eph_single is not None:
            return eph_single
        if jd_value < JD_BOUNDARY:
            return eph_part1
        else:
            return eph_part2
    
    years = []
    jup_lon = []
    sat_lon = []
    valid_indices = []
    
    for i, (year, jd) in enumerate(HALLEY_PERIHELIA):
        try:
            eph = get_eph_for_jd(jd)
            
            sun = eph['sun']
            jupiter = eph['jupiter barycenter']
            saturn = eph['saturn barycenter']
            
            t = ts.tt_jd(jd)
            
            # Heliocentric ecliptic longitude of Jupiter
            jup_pos = (jupiter - sun).at(t)
            jup_lat, jup_longitude, jup_dist = jup_pos.ecliptic_latlon()
            
            # Heliocentric ecliptic longitude of Saturn
            sat_pos = (saturn - sun).at(t)
            sat_lat, sat_longitude, sat_dist = sat_pos.ecliptic_latlon()
            
            years.append(year)
            jup_lon.append(jup_longitude.degrees)
            sat_lon.append(sat_longitude.degrees)
            valid_indices.append(i)
            
        except Exception as e:
            print(f"  Skipping year {year}: {e}")
    
    print(f"Successfully computed positions for {len(years)} perihelion dates.")
    return np.array(valid_indices), np.array(jup_lon), np.array(sat_lon)


def get_positions_chirikov():
    """Fall back to Chirikov & Vecheslavov (1989) phase data.
    X_n = t_n / P_J (Jupiter phase), Y_n = t_n / P_S (Saturn phase)
    These are orbital phases, not ecliptic longitudes, but serve
    for the correlation analysis."""
    
    # From Chirikov Table 1 (reversed to chronological order)
    # Effective periods: P_J = 4332.653 days, P_S = 10759.362 days
    P_J = 4332.653  # days
    P_S = 10759.362  # days
    
    jd_all = [p[1] for p in HALLEY_PERIHELIA]
    
    # Compute phases relative to the 1986 perihelion
    jd_ref = jd_all[-1]  # 1986
    
    jup_phase_deg = [((jd - jd_ref) / P_J * 360) % 360 for jd in jd_all]
    sat_phase_deg = [((jd - jd_ref) / P_S * 360) % 360 for jd in jd_all]
    
    valid_indices = np.arange(len(HALLEY_PERIHELIA))
    return valid_indices, np.array(jup_phase_deg), np.array(sat_phase_deg)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def circular_linear_correlation(theta_deg, y):
    """Compute circular-linear correlation between angle theta and 
    linear variable y. Returns R, optimal phase, and approximate p-value."""
    theta = np.radians(theta_deg)
    
    # Pearson correlations with sin and cos components
    y_centered = y - np.mean(y)
    
    r_sin = np.corrcoef(y, np.sin(theta))[0, 1]
    r_cos = np.corrcoef(y, np.cos(theta))[0, 1]
    
    R = np.sqrt(r_sin**2 + r_cos**2)
    phi = np.degrees(np.arctan2(r_sin, r_cos))
    
    # Approximate p-value: under H0, n*R^2 ~ chi-squared(2)
    n = len(y)
    chi2 = n * R**2
    p_value = np.exp(-chi2 / 2)
    
    return R, phi, p_value


def run_correlation_analysis(valid_indices, jup_lon, sat_lon):
    """Run the full correlation analysis."""
    
    jd_all = [p[1] for p in HALLEY_PERIHELIA]
    years_all = [p[0] for p in HALLEY_PERIHELIA]
    
    # Compute individual periods (days -> years)
    all_periods = np.array([(jd_all[i+1] - jd_all[i]) / 365.25 
                            for i in range(len(jd_all) - 1)])
    
    # Mean period
    mean_P = np.mean(all_periods)
    
    # Period deviations
    delta_P = all_periods - mean_P
    
    # For each period i (between perihelion i and i+1),
    # the relevant planetary position is at perihelion i (start of period)
    # We need indices where BOTH the start perihelion has valid positions
    # AND the period can be computed (i.e., i < len-1)
    
    # Map from perihelion index to position arrays
    idx_to_pos = {}
    for array_idx, perihelion_idx in enumerate(valid_indices):
        idx_to_pos[perihelion_idx] = array_idx
    
    # Build matched arrays
    matched_delta_P = []
    matched_jup = []
    matched_sat = []
    matched_years = []
    
    for i in range(len(all_periods)):
        if i in idx_to_pos:
            matched_delta_P.append(delta_P[i])
            pos_idx = idx_to_pos[i]
            matched_jup.append(jup_lon[pos_idx])
            matched_sat.append(sat_lon[pos_idx])
            matched_years.append(years_all[i])
    
    matched_delta_P = np.array(matched_delta_P)
    matched_jup = np.array(matched_jup)
    matched_sat = np.array(matched_sat)
    
    n = len(matched_delta_P)
    
    print(f"\n{'=' * 65}")
    print(f"PERIOD-PERTURBATION CORRELATION ANALYSIS")
    print(f"{'=' * 65}")
    print(f"\nMatched data points: {n}")
    print(f"Mean period: {mean_P:.4f} yr")
    print(f"Period std: {np.std(all_periods):.4f} yr")
    
    # Correlation with Jupiter
    print(f"\n--- JUPITER ---")
    R_j, phi_j, p_j = circular_linear_correlation(matched_jup, matched_delta_P)
    print(f"Circular-linear R: {R_j:.4f}")
    print(f"Optimal phase: {phi_j:.1f}°")
    print(f"p-value: {p_j:.6f}")
    
    # Amplitude of Jupiter's effect
    theta_j = np.radians(matched_jup)
    A_j_sin = np.corrcoef(matched_delta_P, np.sin(theta_j))[0,1] * np.std(matched_delta_P)
    A_j_cos = np.corrcoef(matched_delta_P, np.cos(theta_j))[0,1] * np.std(matched_delta_P)
    amplitude_j = 2 * np.sqrt(A_j_sin**2 + A_j_cos**2)
    print(f"Perturbation amplitude: ~{amplitude_j:.3f} yr = ~{amplitude_j*365.25:.0f} days")
    
    # Correlation with Saturn
    print(f"\n--- SATURN ---")
    R_s, phi_s, p_s = circular_linear_correlation(matched_sat, matched_delta_P)
    print(f"Circular-linear R: {R_s:.4f}")
    print(f"Optimal phase: {phi_s:.1f}°")
    print(f"p-value: {p_s:.6f}")
    
    theta_s = np.radians(matched_sat)
    A_s_sin = np.corrcoef(matched_delta_P, np.sin(theta_s))[0,1] * np.std(matched_delta_P)
    A_s_cos = np.corrcoef(matched_delta_P, np.cos(theta_s))[0,1] * np.std(matched_delta_P)
    amplitude_s = 2 * np.sqrt(A_s_sin**2 + A_s_cos**2)
    print(f"Perturbation amplitude: ~{amplitude_s:.3f} yr = ~{amplitude_s*365.25:.0f} days")
    
    if amplitude_s > 0:
        print(f"\nJupiter/Saturn amplitude ratio: {amplitude_j/amplitude_s:.1f}")
    
    # Combined model: δP ~ a·sin(λ_J) + b·cos(λ_J) + c·sin(λ_S) + d·cos(λ_S)
    print(f"\n--- COMBINED MODEL ---")
    A_mat = np.column_stack([
        np.sin(theta_j), np.cos(theta_j),
        np.sin(theta_s), np.cos(theta_s),
    ])
    
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, matched_delta_P, rcond=None)
    predicted = A_mat @ coeffs
    
    ss_res = np.sum((matched_delta_P - predicted)**2)
    ss_tot = np.sum((matched_delta_P - np.mean(matched_delta_P))**2)
    R2 = 1 - ss_res / ss_tot
    
    print(f"R² = {R2:.4f} ({R2*100:.1f}% of variance explained)")
    
    # F-test
    k = 4
    F = (R2 / k) / ((1 - R2) / (n - k - 1)) if n > k + 1 else 0
    print(f"F-statistic: {F:.2f} (df = {k}, {n-k-1})")
    
    return matched_delta_P, matched_jup, matched_sat, matched_years


def test_cancellation_in_blocks(delta_P_all):
    """Test whether perturbations cancel in blocks of 15 orbits."""
    
    print(f"\n{'=' * 65}")
    print("CANCELLATION TEST: DO PERTURBATIONS SUM TO ZERO EVERY 15 ORBITS?")
    print(f"{'=' * 65}")
    
    # Use ALL 29 periods (not just those with matched positions)
    jd_all = [p[1] for p in HALLEY_PERIHELIA]
    all_periods = np.array([(jd_all[i+1] - jd_all[i]) / 365.25 
                            for i in range(len(jd_all) - 1)])
    mean_P = np.mean(all_periods)
    delta_P = all_periods - mean_P
    
    print(f"\nAll 29 orbital periods used.")
    print(f"Mean period: {mean_P:.6f} yr")
    
    # Cumulative sum
    cumsum = np.cumsum(delta_P)
    
    print(f"\n{'n':<6} {'From':<8} {'Σ δP (yr)':<14} {'Σ δP (days)':<14} {'Σ δP/n (yr)':<14}")
    print("-" * 58)
    
    years_all = [p[0] for p in HALLEY_PERIHELIA]
    
    for i in range(len(cumsum)):
        n = i + 1
        yr = years_all[i]
        if n % 5 == 0 or n == len(cumsum) or n == 15:
            marker = "  ← 15 orbits" if n == 15 else ""
            marker = "  ← 29 orbits (ALL)" if n == 29 else marker
            print(f"{n:<6} {yr:<8} {cumsum[i]:<+14.4f} {cumsum[i]*365.25:<+14.1f} "
                  f"{cumsum[i]/n:<+14.6f}{marker}")
    
    # Test: is the sum at n=15 closer to zero than expected?
    # Random walk: expected |Σ| ~ σ·√n
    sigma = np.std(delta_P)
    expected_abs_sum_15 = sigma * np.sqrt(15)
    actual_abs_sum_15 = abs(cumsum[14]) if len(cumsum) >= 15 else None
    
    if actual_abs_sum_15 is not None:
        print(f"\nAt n = 15 orbits:")
        print(f"  Actual |Σ δP| = {actual_abs_sum_15:.4f} yr = {actual_abs_sum_15*365.25:.1f} days")
        print(f"  Expected |Σ δP| for random walk = σ·√15 = {expected_abs_sum_15:.4f} yr = {expected_abs_sum_15*365.25:.1f} days")
        
        if actual_abs_sum_15 < expected_abs_sum_15:
            print(f"  → Actual sum is SMALLER than random walk expectation")
            print(f"    ({actual_abs_sum_15/expected_abs_sum_15:.1%} of expected)")
        else:
            print(f"  → Actual sum is LARGER than random walk expectation")
    
    # Also check at n=29
    actual_abs_sum_29 = abs(cumsum[28])
    expected_abs_sum_29 = sigma * np.sqrt(29)
    
    print(f"\nAt n = 29 orbits (all data):")
    print(f"  Actual |Σ δP| = {actual_abs_sum_29:.4f} yr = {actual_abs_sum_29*365.25:.1f} days")
    print(f"  Expected |Σ δP| for random walk = σ·√29 = {expected_abs_sum_29:.4f} yr = {expected_abs_sum_29*365.25:.1f} days")
    
    return cumsum


def plot_analysis(matched_delta_P, matched_jup, matched_sat, matched_years, cumsum):
    """Generate plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')
    
    for ax in axes.flat:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
    
    # Plot 1: δP vs Jupiter longitude
    ax = axes[0, 0]
    ax.scatter(matched_jup, matched_delta_P, color='#58a6ff', s=30, alpha=0.7)
    # Fit curve
    theta = np.radians(matched_jup)
    A = np.column_stack([np.sin(theta), np.cos(theta)])
    c, _, _, _ = np.linalg.lstsq(A, matched_delta_P, rcond=None)
    x_fit = np.linspace(0, 360, 200)
    y_fit = c[0] * np.sin(np.radians(x_fit)) + c[1] * np.cos(np.radians(x_fit))
    ax.plot(x_fit, y_fit, color='#f85149', linewidth=2)
    ax.axhline(y=0, color='#8b949e', linewidth=0.5)
    ax.set_xlabel('Jupiter longitude (°)', color='#c9d1d9')
    ax.set_ylabel('δP (years)', color='#c9d1d9')
    ax.set_title('Period deviation vs Jupiter position', color='#e6edf3')
    
    # Plot 2: δP vs Saturn longitude
    ax = axes[0, 1]
    ax.scatter(matched_sat, matched_delta_P, color='#3fb950', s=30, alpha=0.7)
    theta_s = np.radians(matched_sat)
    A_s = np.column_stack([np.sin(theta_s), np.cos(theta_s)])
    c_s, _, _, _ = np.linalg.lstsq(A_s, matched_delta_P, rcond=None)
    y_fit_s = c_s[0] * np.sin(np.radians(x_fit)) + c_s[1] * np.cos(np.radians(x_fit))
    ax.plot(x_fit, y_fit_s, color='#f85149', linewidth=2)
    ax.axhline(y=0, color='#8b949e', linewidth=0.5)
    ax.set_xlabel('Saturn longitude (°)', color='#c9d1d9')
    ax.set_ylabel('δP (years)', color='#c9d1d9')
    ax.set_title('Period deviation vs Saturn position', color='#e6edf3')
    
    # Plot 3: Cumulative sum of δP
    ax = axes[1, 0]
    n_vals = np.arange(1, len(cumsum) + 1)
    ax.plot(n_vals, cumsum * 365.25, 'o-', color='#58a6ff', markersize=4)
    ax.axhline(y=0, color='#f85149', linewidth=1, linestyle='--')
    ax.axvline(x=15, color='#3fb950', linewidth=1, linestyle=':', label='n = 15')
    ax.set_xlabel('Number of orbits', color='#c9d1d9')
    ax.set_ylabel('Cumulative Σ δP (days)', color='#c9d1d9')
    ax.set_title('Cumulative perturbation sum', color='#e6edf3')
    ax.legend()
    
    # Plot 4: Running mean period vs T*/15
    ax = axes[1, 1]
    jd_all = [p[1] for p in HALLEY_PERIHELIA]
    running_means = []
    for n in range(2, len(jd_all)):
        span = jd_all[n] - jd_all[0]
        rm = span / n / 365.25
        running_means.append(rm)
    n_vals_rm = np.arange(2, len(jd_all))
    ax.plot(n_vals_rm, running_means, 'o-', color='#58a6ff', markersize=4, 
            label='Running mean')
    ax.axhline(y=1151/15, color='#f85149', linewidth=2, linestyle='--', 
               label=f'T*/15 = {1151/15:.4f} yr')
    ax.set_xlabel('Number of periods', color='#c9d1d9')
    ax.set_ylabel('Mean period (years)', color='#c9d1d9')
    ax.set_title('Convergence of mean period', color='#e6edf3')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('perturbation_correlation.png', dpi=150, facecolor='#0d1117')
    print(f"\nPlot saved: perturbation_correlation.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("HALLEY PERIOD-PERTURBATION CORRELATION TEST")
    print("=" * 65)
    
    # Get planetary positions
    if HAS_SKYFIELD:
        valid_idx, jup_lon, sat_lon = get_positions_skyfield()
        if valid_idx is None:
            print("Falling back to Chirikov phase data.")
            valid_idx, jup_lon, sat_lon = get_positions_chirikov()
            source = "Chirikov phases"
        else:
            source = "Skyfield/DE441"
    else:
        valid_idx, jup_lon, sat_lon = get_positions_chirikov()
        source = "Chirikov phases"
    
    print(f"\nPlanetary position source: {source}")
    
    # Run correlation analysis
    matched_dP, matched_j, matched_s, matched_y = run_correlation_analysis(
        valid_idx, jup_lon, sat_lon)
    
    # Run cancellation test
    cumsum = test_cancellation_in_blocks(None)
    
    # Plot
    plot_analysis(matched_dP, matched_j, matched_s, matched_y, cumsum)
    
    print(f"\n{'=' * 65}")
    print("DONE")
    print(f"{'=' * 65}")
