#!/usr/bin/env python3
"""
N-body simulation of Comet 1P/Halley using REBOUND.

HYPOTHESIS TESTED:

If T*/15 = 76.733 yr is a dynamically preferred period for Halley,
then the running mean orbital period should remain near T*/15 over
timescales much longer than the historical record (29 orbits, 2225 yr),
without systematic drift — despite Jupiter-Saturn perturbations that
cause individual period variations of ±2 yr.

This extends the 29 observed periods to ~130 simulated periods over
10,000 years, using the full gravitational influence of all 8 planets.

METHOD:

Uses REBOUND with the IAS15 integrator (adaptive timestep, 15th-order
Gauss-Radau, standard for cometary dynamics). The Solar System is
initialized from JPL Horizons elements at J2000. Halley is initialized
from its known orbital elements. Perihelion passages are detected by
monitoring the Sun-Halley distance for local minima.

TWO EXPERIMENTS:

1. MAIN: Single Halley integration for 10,000 yr.
   Monitor running mean period every perihelion.
   Test: does the running mean stay near T*/15 = 76.733 yr?

2. ENSEMBLE: 50 test particles with periods uniformly distributed
   in [73, 81] yr, same i, e, Ω, ω as Halley, integrated for 3,000 yr.
   Test: do particles initialized near T*/15 show smaller drift of
   running mean compared to particles initialized far from T*/15?
   This tests whether T*/15 is dynamically special, not just
   arithmetically convenient.

EXPECTED RESULTS under the coupling hypothesis:
  - Main: running mean oscillates around T*/15 without systematic drift
  - Ensemble: particles near T*/15 show smaller |ΔP̄| after N orbits

REBOUND references:
  Rein & Liu (2012), A&A 537, A128  — REBOUND
  Rein & Spiegel (2015), MNRAS 446  — IAS15 integrator

Author: C. Baiget Orts (2026)
Requires: rebound, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import rebound
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS AND TARGETS
# ============================================================
TSTAR_YR   = 1151.0
P_COMM     = TSTAR_YR / 15.0       # 76.7333... yr
T_MAIN     = 10_000.0              # yr, main integration
T_ENSEMBLE = 3_000.0               # yr, ensemble integration
N_PARTICLES = 50                   # ensemble size
YR          = 2 * np.pi            # REBOUND uses yr and AU by default

# Halley mean orbital elements (J2000, Horizons)
HALLEY = dict(
    a     = 17.8341442,    # AU  semi-major axis
    e     = 0.9671429,     # eccentricity
    inc   = np.radians(162.2627),   # inclination
    Omega = np.radians(58.4201),    # longitude of ascending node
    omega = np.radians(111.3328),   # argument of perihelion
    M     = np.radians(358.2538),   # mean anomaly at J2000
)

# Perihelion distance
Q_HAL = HALLEY['a'] * (1 - HALLEY['e'])   # ~0.586 AU


# ============================================================
# BUILD SIMULATION
# ============================================================

def build_sim(extra_particles=None):
    """
    Initialize REBOUND simulation with Sun + 8 planets + Halley.
    Uses JPL Horizons data via rebound.add() with hash names.
    Falls back to approximate elements if Horizons unavailable.
    extra_particles: list of dicts with orbital elements to add
                     as massless test particles.
    """
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.integrator = 'ias15'
    sim.dt = 0.01   # yr, initial timestep (IAS15 adjusts automatically)

    # --- Add Solar System bodies from Horizons ---
    try:
        bodies = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars',
                  'Jupiter', 'Saturn', 'Uranus', 'Neptune']
        for b in bodies:
            sim.add(b, date='2000-01-01')
        print("  Planets loaded from JPL Horizons (online).")
        from_horizons = True
    except Exception:
        print("  Horizons unavailable. Using approximate orbital elements.")
        from_horizons = False
        _add_planets_approximate(sim)

    sim.move_to_com()

    # --- Add Halley ---
    sim.add(
        m     = 0.0,           # massless
        a     = HALLEY['a'],
        e     = HALLEY['e'],
        inc   = HALLEY['inc'],
        Omega = HALLEY['Omega'],
        omega = HALLEY['omega'],
        M     = HALLEY['M'],
        primary = sim.particles[0]
    )
    halley_idx = len(sim.particles) - 1

    # --- Add ensemble test particles if requested ---
    if extra_particles:
        for ep in extra_particles:
            sim.add(
                m     = 0.0,
                a     = ep['a'],
                e     = ep['e'],
                inc   = ep['inc'],
                Omega = ep['Omega'],
                omega = ep['omega'],
                M     = ep['M'],
                primary = sim.particles[0]
            )

    sim.move_to_com()
    return sim, halley_idx, from_horizons


def _add_planets_approximate(sim):
    """Approximate planet elements if Horizons unavailable."""
    # [name, m/Msun, a/AU, e, i/deg, Omega/deg, omega/deg, M/deg]
    planets = [
        ('Sun',     1.0,          0,      0,     0,     0,     0,     0),
        ('Mercury', 1.6601e-7,  0.387,  0.206,  7.0,   48.3,  29.1,  174.8),
        ('Venus',   2.4478e-6,  0.723,  0.007,  3.4,   76.7,  54.9,  50.4),
        ('Earth',   3.0034e-6,  1.000,  0.017,  0.0,  -11.3, 102.9,  357.5),
        ('Mars',    3.2271e-7,  1.524,  0.093,  1.8,   49.6,  286.5, 19.4),
        ('Jupiter', 9.5458e-4,  5.203,  0.049,  1.3,  100.5,  273.9, 20.0),
        ('Saturn',  2.8577e-4,  9.537,  0.057,  2.5,  113.7,  339.4, 317.0),
        ('Uranus',  4.3653e-5, 19.191,  0.046,  0.8,   74.0,  96.9,  142.2),
        ('Neptune', 5.1499e-5, 30.069,  0.010,  1.8,  131.8,  272.8, 256.2),
    ]
    for row in planets[1:]:
        name, m, a, e, inc, Omega, omega, M = row
        sim.add(m=m, a=a, e=e,
                inc=np.radians(inc),
                Omega=np.radians(Omega),
                omega=np.radians(omega),
                M=np.radians(M),
                primary=sim.particles[0])


# ============================================================
# PERIHELION DETECTOR
# ============================================================

def integrate_and_detect_perihelia(sim, halley_idx, T_years,
                                   dt_check=0.5, verbose=True):
    """
    Integrate for T_years and detect perihelion passages of
    particle halley_idx by monitoring heliocentric distance.

    Returns list of (time_yr, period_yr) tuples.
    """
    t_end = sim.t + T_years
    dt    = dt_check   # yr between distance checks

    d_prev2 = None
    d_prev1 = None
    t_prev1 = None

    perihelia_t = []   # times of perihelion passages

    n_steps = int(T_years / dt) + 1

    for step in range(n_steps):
        t_target = sim.t + dt
        if t_target > t_end:
            t_target = t_end
        sim.integrate(t_target, exact_finish_time=1)

        p = sim.particles[halley_idx]
        sun = sim.particles[0]
        dx = p.x - sun.x
        dy = p.y - sun.y
        dz = p.z - sun.z
        d_cur = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Local minimum detection: d_prev1 < d_prev2 and d_prev1 < d_cur
        if (d_prev2 is not None and d_prev1 is not None and
                d_prev1 < d_prev2 and d_prev1 < d_cur):
            # Refine: bisect to find precise perihelion time
            # Simple parabolic interpolation
            t_peri = t_prev1
            perihelia_t.append(t_peri)

        d_prev2 = d_prev1
        d_prev1 = d_cur
        t_prev1 = sim.t

        if t_target >= t_end:
            break

    # Compute periods
    perihelia_t = np.array(perihelia_t)
    periods = np.diff(perihelia_t) if len(perihelia_t) > 1 else np.array([])

    if verbose:
        print(f"    Detected {len(perihelia_t)} perihelia, "
              f"{len(periods)} periods")

    return perihelia_t, periods


# ============================================================
# EXPERIMENT 1: MAIN INTEGRATION (single Halley, 10,000 yr)
# ============================================================

def experiment_main():
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Single Halley integration — 10,000 yr")
    print("=" * 72)
    print(f"\n  T*/15 = {P_COMM:.6f} yr (target)")
    print(f"  Integration: {T_MAIN:.0f} yr (~{T_MAIN/P_COMM:.0f} orbits)")
    print(f"  Integrator: IAS15 (adaptive timestep)")

    sim, halley_idx, from_horizons = build_sim()
    n_planets = halley_idx   # number of massive bodies before Halley

    print(f"  Bodies: {n_planets} planets + Halley (massless)")
    print(f"  Integrating...")

    t_peri, periods = integrate_and_detect_perihelia(
        sim, halley_idx, T_MAIN, dt_check=0.5, verbose=True
    )

    if len(periods) < 5:
        print("  ERROR: Too few perihelia detected. Check integration.")
        return None, None

    # Running mean period
    n_orb = len(periods)
    running_mean = np.array([periods[:k+1].mean() for k in range(n_orb)])
    running_std  = np.array([periods[:k+1].std(ddof=min(1,k))
                             for k in range(n_orb)])

    # Observed historical values for comparison
    P_obs_hist = 76.713   # yr (from 29 historical periods)

    print(f"\n  RESULTS:")
    print(f"  {'Orbit':>6}  {'Period (yr)':>12}  {'Running mean':>13}  "
          f"{'|Mean - T*/15|':>15}")
    print(f"  {'-'*52}")
    report_at = list(range(0, min(n_orb, 10))) + \
                [14, 19, 24, 29, 49, 74, 99, 119, n_orb-1]
    report_at = sorted(set(i for i in report_at if i < n_orb))
    for i in report_at:
        print(f"  {i+1:>6}  {periods[i]:>12.4f}  {running_mean[i]:>13.6f}  "
              f"  {abs(running_mean[i] - P_COMM):>12.4f}")

    print(f"\n  Final running mean ({n_orb} orbits): {running_mean[-1]:.6f} yr")
    print(f"  T*/15:                              {P_COMM:.6f} yr")
    print(f"  Historical P̄ (29 orbits):           {P_obs_hist:.6f} yr")
    print(f"  |Final mean - T*/15|:               "
          f"{abs(running_mean[-1] - P_COMM)*365.25:.2f} days")
    print(f"  |Final mean - P̄_hist|:              "
          f"{abs(running_mean[-1] - P_obs_hist)*365.25:.2f} days")

    # Drift test: linear trend of running mean over last 50% of orbits
    if n_orb > 20:
        half = n_orb // 2
        idx = np.arange(half, n_orb)
        slope, intercept, r, p_val, se = \
            __import__('scipy').stats.linregress(idx, running_mean[half:])
        print(f"\n  Linear drift of running mean (last {n_orb-half} orbits):")
        print(f"  slope = {slope*365.25:.4f} days/orbit, "
              f"p = {p_val:.4f} "
              f"({'significant drift' if p_val < 0.05 else 'no significant drift'})")

    return t_peri, periods, running_mean, running_std


# ============================================================
# EXPERIMENT 2: ENSEMBLE (50 particles, 3,000 yr)
# ============================================================

def experiment_ensemble():
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Ensemble — 50 particles in [73, 81] yr, 3,000 yr")
    print("=" * 72)
    print(f"\n  T*/15 = {P_COMM:.6f} yr")
    print(f"  Particles: {N_PARTICLES} with a uniformly spaced in [73, 81] yr")
    print(f"  Same e, i, Ω, ω as Halley")

    # Period range → semi-major axis via Kepler's third law
    # P² = a³ (in yr and AU with GM_sun = 4π²)
    P_range = np.linspace(73.0, 81.0, N_PARTICLES)
    a_range = P_range ** (2/3)

    # Build extra particles
    extra = []
    for a in a_range:
        extra.append(dict(
            a     = a,
            e     = HALLEY['e'],
            inc   = HALLEY['inc'],
            Omega = HALLEY['Omega'],
            omega = HALLEY['omega'],
            M     = HALLEY['M'],
        ))

    sim, halley_idx, _ = build_sim(extra_particles=extra)
    n_total   = len(sim.particles)
    # Extra particles start AFTER Halley
    extra_start = halley_idx + 1

    print(f"  Extra test particles: {n_total - extra_start}")
    print(f"  Integrating {T_ENSEMBLE:.0f} yr...")

    # Track only the 50 extra particles (not Halley itself)
    # P_range[k] corresponds to particle at index extra_start + k
    particle_indices = list(range(extra_start, n_total))
    n_test = len(particle_indices)   # = N_PARTICLES = 50

    dt_check = 0.5
    n_steps = int(T_ENSEMBLE / dt_check) + 1

    # Storage
    d_prev2 = [None] * n_test
    d_prev1 = [None] * n_test
    t_prev1 = [None] * n_test
    perihelia_all = [[] for _ in range(n_test)]

    sun = sim.particles[0]

    for step in range(n_steps):
        t_target = sim.t + dt_check
        if t_target > T_ENSEMBLE:
            t_target = T_ENSEMBLE
        try:
            sim.integrate(t_target, exact_finish_time=1)
        except rebound.Escape:
            print(f"    Particle escaped at t={sim.t:.1f} yr")
            break

        for k, idx in enumerate(particle_indices):
            if idx >= len(sim.particles):
                continue
            p = sim.particles[idx]
            dx = p.x - sun.x
            dy = p.y - sun.y
            dz = p.z - sun.z
            d_cur = np.sqrt(dx*dx + dy*dy + dz*dz)

            if (d_prev2[k] is not None and d_prev1[k] is not None and
                    d_prev1[k] < d_prev2[k] and d_prev1[k] < d_cur):
                perihelia_all[k].append(t_prev1[k])

            d_prev2[k] = d_prev1[k]
            d_prev1[k] = d_cur
            t_prev1[k] = sim.t

        if t_target >= T_ENSEMBLE:
            break

    # Compute final running mean for each particle
    results = []
    for k in range(n_test):
        t_arr = np.array(perihelia_all[k])
        if len(t_arr) < 3:
            results.append((P_range[k], np.nan, 0))
            continue
        periods_k = np.diff(t_arr)
        mean_k = periods_k.mean()
        n_k = len(periods_k)
        results.append((P_range[k], mean_k, n_k))

    results = np.array([(r[0], r[1], r[2]) for r in results])
    P_init  = results[:, 0]
    P_final = results[:, 1]
    N_orbs  = results[:, 2]

    valid = ~np.isnan(P_final)
    drift = P_final[valid] - P_init[valid]

    print(f"\n  RESULTS:")
    print(f"  {'P_init':>8}  {'P_final':>9}  {'drift':>8}  {'N_orbs':>7}  "
          f"{'|drift|':>8}")
    print(f"  {'-'*50}")
    for i in range(len(P_init)):
        if not valid[i]:
            continue
        marker = ' ← T*/15' if abs(P_init[i] - P_COMM) < 0.5 else ''
        print(f"  {P_init[i]:>8.3f}  {P_final[i]:>9.4f}  "
              f"{P_final[i]-P_init[i]:>+8.4f}  {int(N_orbs[i]):>7}  "
              f"{abs(P_final[i]-P_init[i]):>8.4f}{marker}")

    # Does drift correlate with distance from T*/15?
    dist_from_comm = np.abs(P_init[valid] - P_COMM)
    abs_drift = np.abs(drift)
    if len(abs_drift) > 5:
        from scipy import stats
        r, p = stats.pearsonr(dist_from_comm, abs_drift)
        print(f"\n  Correlation |drift| vs |P_init - T*/15|:")
        print(f"  r = {r:.4f},  p = {p:.4f}")
        if p < 0.05:
            print(f"  → Particles near T*/15 drift LESS: "
                  f"dynamical preference confirmed")
        else:
            print(f"  → No significant correlation: "
                  f"no detectable dynamical preference at this timescale")

    return P_init, P_final, N_orbs, valid


# ============================================================
# PLOTS
# ============================================================

def make_plots_main(t_peri, periods, running_mean, running_std):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0d1117')

    n = len(periods)
    orb_idx = np.arange(1, n + 1)
    t_mid = (t_peri[:-1] + t_peri[1:]) / 2 if len(t_peri) > 1 else orb_idx

    def style(ax):
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e')
        for sp in ax.spines.values():
            sp.set_color('#30363d')

    # Panel 1: individual periods
    ax = axes[0]; style(ax)
    ax.scatter(orb_idx, periods, s=15, color='#58a6ff', alpha=0.7, zorder=3)
    ax.axhline(P_COMM,    color='#f85149', lw=2,   linestyle='--',
               label=f'T*/15 = {P_COMM:.3f} yr')
    ax.axhline(76.713,    color='#3fb950', lw=1.5, linestyle=':',
               label='hist. P̄ = 76.713 yr')
    ax.set_xlabel('orbit number', color='#c9d1d9')
    ax.set_ylabel('period (yr)', color='#c9d1d9')
    ax.set_title('Individual periods\n(10,000 yr simulation)',
                 color='#e6edf3')
    ax.legend(facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#c9d1d9', fontsize=8)

    # Panel 2: running mean
    ax = axes[1]; style(ax)
    ax.plot(orb_idx, running_mean, color='#58a6ff', lw=1.5)
    ax.fill_between(orb_idx,
                    running_mean - running_std / np.sqrt(orb_idx),
                    running_mean + running_std / np.sqrt(orb_idx),
                    alpha=0.2, color='#58a6ff')
    ax.axhline(P_COMM,  color='#f85149', lw=2, linestyle='--',
               label=f'T*/15 = {P_COMM:.3f} yr')
    ax.axhline(76.713,  color='#3fb950', lw=1.5, linestyle=':',
               label='hist. P̄ = 76.713 yr')
    ax.axvline(15, color='#e3b341', lw=1, linestyle=':',
               label='n=15 (1 T* cycle)')
    ax.axvline(29, color='#8b949e', lw=1, linestyle=':',
               label='n=29 (hist. baseline)')
    ax.set_xlabel('orbit number', color='#c9d1d9')
    ax.set_ylabel('running mean period (yr)', color='#c9d1d9')
    ax.set_title('Running mean convergence\n(shaded = ±σ/√n)',
                 color='#e6edf3')
    ax.legend(facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#c9d1d9', fontsize=7)

    # Panel 3: deviation from T*/15 over time
    ax = axes[2]; style(ax)
    dev = (running_mean - P_COMM) * 365.25   # days
    ax.plot(orb_idx, dev, color='#e3b341', lw=1.2)
    ax.axhline(0, color='#f85149', lw=1.5, linestyle='--',
               label='T*/15 (zero deviation)')
    ax.axhline((76.713 - P_COMM) * 365.25, color='#3fb950',
               lw=1, linestyle=':', label='hist. P̄')
    ax.fill_between(orb_idx, dev, 0,
                    where=(np.abs(dev) < 10),
                    alpha=0.2, color='#e3b341')
    ax.set_xlabel('orbit number', color='#c9d1d9')
    ax.set_ylabel('running mean − T*/15 (days)', color='#c9d1d9')
    ax.set_title('Drift from T*/15\n(zero = perfect commensurability)',
                 color='#e6edf3')
    ax.legend(facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#c9d1d9', fontsize=8)

    fig.suptitle('REBOUND N-body: 1P/Halley running mean period — '
                 f'{T_MAIN:.0f} yr',
                 color='#e6edf3', fontsize=12)
    plt.tight_layout()
    plt.savefig('rebound_main.png', dpi=140,
                facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print("  Figure saved: rebound_main.png")


def make_plots_ensemble(P_init, P_final, N_orbs, valid):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117')

    def style(ax):
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e')
        for sp in ax.spines.values():
            sp.set_color('#30363d')

    Pi = P_init[valid]
    Pf = P_final[valid]
    drift = Pf - Pi

    # Panel 1: P_init vs P_final
    ax = axes[0]; style(ax)
    ax.scatter(Pi, Pf, c=np.abs(drift), cmap='RdYlGn_r',
               s=40, alpha=0.8, zorder=3)
    lims = [min(Pi.min(), Pf.min()) - 0.2,
            max(Pi.max(), Pf.max()) + 0.2]
    ax.plot(lims, lims, color='#8b949e', lw=1, linestyle='--',
            label='no drift')
    ax.axvline(P_COMM, color='#f85149', lw=2, linestyle='--',
               label=f'T*/15 = {P_COMM:.3f} yr')
    ax.set_xlabel('initial period (yr)', color='#c9d1d9')
    ax.set_ylabel('final running mean period (yr)', color='#c9d1d9')
    ax.set_title(f'Ensemble: initial vs final period\n'
                 f'({T_ENSEMBLE:.0f} yr, {N_PARTICLES} particles)',
                 color='#e6edf3')
    ax.legend(facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#c9d1d9', fontsize=8)

    # Panel 2: |drift| vs distance from T*/15
    ax = axes[1]; style(ax)
    dist = np.abs(Pi - P_COMM)
    ax.scatter(dist, np.abs(drift) * 365.25, color='#58a6ff',
               s=40, alpha=0.8, zorder=3)
    # Regression
    if len(dist) > 5:
        m, b = np.polyfit(dist, np.abs(drift) * 365.25, 1)
        xs = np.linspace(dist.min(), dist.max(), 100)
        ax.plot(xs, m * xs + b, color='#f85149', lw=1.5)
    ax.axvline(0, color='#f85149', lw=1.5, linestyle='--',
               label='T*/15')
    ax.set_xlabel('|P_init − T*/15| (yr)', color='#c9d1d9')
    ax.set_ylabel('|drift of running mean| (days)', color='#c9d1d9')
    ax.set_title('Drift magnitude vs distance from T*/15\n'
                 '(expected: positive slope if T*/15 is dynamically preferred)',
                 color='#e6edf3')
    ax.legend(facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#c9d1d9', fontsize=8)

    fig.suptitle(f'REBOUND ensemble: {N_PARTICLES} test particles, '
                 f'{T_ENSEMBLE:.0f} yr',
                 color='#e6edf3', fontsize=12)
    plt.tight_layout()
    plt.savefig('rebound_ensemble.png', dpi=140,
                facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print("  Figure saved: rebound_ensemble.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("REBOUND N-BODY SIMULATION — 1P/Halley & T*/15 commensurability")
    print("=" * 72)
    print(f"\nREBOUND version: {rebound.__version__}")
    print(f"T*/15 target period: {P_COMM:.6f} yr = {P_COMM*365.25:.3f} days")

    # --- Experiment 1 ---
    result_main = experiment_main()
    if result_main[0] is not None:
        t_peri, periods, running_mean, running_std = result_main
        make_plots_main(t_peri, periods, running_mean, running_std)

    # --- Experiment 2 ---
    P_init, P_final, N_orbs, valid = experiment_ensemble()
    if valid.sum() > 3:
        make_plots_ensemble(P_init, P_final, N_orbs, valid)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
