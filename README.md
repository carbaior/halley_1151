# Comet 1P/Halley Completes 15 Orbits in 1,151 Years

**Replication package for:**

> Baiget Orts, C. (2026). *Comet 1P/Halley Completes 15 Orbits in 1,151 Years:
> Commensurability with the Solar System Quasi-Period and Evidence for
> Jupiter–Saturn Dynamical Coupling.* arXiv: [in preparation]

**Companion paper (planetary cycle):**

> Baiget Orts, C. (2026). *A 1151-Year Quasi-Commensurability of the Solar System:
> Empirical Detection, Statistical Characterization, and the Anomalous Exclusion
> of Uranus.* arXiv:2604.03049

---

## Overview

This repository contains the complete replication code for all statistical
results reported in the Halley paper. The central finding is that comet
1P/Halley's mean orbital period (76.713 yr, from 29 observed orbital periods
spanning 2,225 years) satisfies:

```
T* / P̄ = 1151 / 76.713 = 15.004
```

with an angular residue of +1.43° — the smallest of any Solar System body
analyzed with this method, smaller than those of all seven participating planets.

Four independent statistical tests establish that Jupiter and Saturn couple to
Halley's orbital period through distinct mechanisms:
- **Jupiter** via phase-dependent modulation (p = 0.027–0.04, three tests)
- **Saturn** via distance-amplitude modulation (p = 0.007, confirmed by
  random-phase control p = 0.133)

---

## Repository structure

```
halley_1151_replication.py      Main replication battery (all paper results)
test_phase_locked_permutation.py Phase-locked permutation test (p=0.035)
test_gravitational_impulse.py   Direct gravitational impulse test
test_saturn_proximity_permutation.py Saturn proximity permutation (extended)
test_perturbation_correlation.py Chirikov-phase correlation + cancellation
rebound_halley.py               N-body ensemble (requires REBOUND)
rebound_stabilization.py        Long-term stabilization test (requires REBOUND)
tstar_population.py             T* divisor analysis — HTC population
tstar_asteroids.py              T* divisor analysis — asteroid belt
README.md                       This file
```

---

## Requirements

### Core replication (no special dependencies)
```bash
pip install numpy matplotlib scipy
```

### N-body scripts only
```bash
pip install rebound
```

REBOUND will attempt to download planet positions from JPL Horizons on first
run (requires internet). The scripts fall back to approximate elements if
Horizons is unavailable.

### Population analysis scripts
```bash
pip install requests   # for JPL SBDB download
```

---

## Quick start

Run the complete replication battery:

```bash
python3 halley_1151_replication.py
```

Expected runtime: ~15 minutes on a modern laptop (dominated by Monte Carlo
tests with 10⁵–10⁶ iterations). All results print to stdout with the
paper value noted for comparison.

---

## Test inventory

### `halley_1151_replication.py` — 14 tests covering all paper sections

| # | Function | Paper section | Key result |
|---|---|---|---|
| 1 | `test_basic` | §3.1 | P̄ = 76.713 yr, residue +1.43°, deviation 7.4 days |
| 2 | `test_comparison` | Table 1 | Residues for all Solar System bodies |
| 3 | `test_htc_survey` | §3.4, Table 2 | Survey of HTCs; residues 80°–130° |
| 4 | `test_bootstrap` | §3.2 | T*/15 inside 95% CI; residue robust |
| 5 | `test_surrogates` | §3.3 | p = 0.036 (Gaussian), p = 0.007 (uniform) |
| 6 | `test_period_scan` | §3.3 | T* rank 1/1901 joint, 16/1901 Halley alone |
| 7 | `test_coincidence_mc` | §3.3 | p = 0.009 (look-elsewhere corrected) |
| 8 | `test_sensitivity` | §3.2 | Robust to ±180 day historical uncertainties |
| 9 | `test_rolling_prediction` | §4.6 | RMS: T*/15 = 472 d, running mean = 494 d |
| 10 | `test_cancellation` | §4.4 | Cancellation at n=15: 9.4% of random-walk σ√15 |
| 11 | `test_arithmetic_landscape` | Fig. 4 | 52 minima below 5°; Halley's convergence is dynamical |
| 12 | `test_jupiter_phase_correlation` | §4.2 | R = 0.47, p = 0.04; R²(J+S) = 0.234 |
| 13 | `test_gravitational_impulse` | §4.2–4.3 | r_J = −0.41, p = 0.027; r_dist_S = −0.496, p = 0.007 |
| 14 | `test_saturn_proximity` | §4.3 | p_perm = 0.007, p_rphase = 0.133 (ratio 20×) |

### Additional scripts (extended analysis)

| Script | Paper section | Key result | Runtime |
|---|---|---|---|
| `test_phase_locked_permutation.py` | §4.2 | p = 0.035 (Jupiter, 10⁶ permutations) | ~3 min |
| `test_gravitational_impulse.py` | §4.2–4.3 | Full impulse analysis with plots | ~5 min |
| `test_saturn_proximity_permutation.py` | §4.3 | Extended Saturn tests with plots | ~5 min |
| `test_perturbation_correlation.py` | §4.2, §4.4 | Chirikov phases + cancellation figure | ~2 min |
| `rebound_halley.py` | §4.5 | N-body ensemble, r = 0.44, p = 0.001 | ~30 min |
| `rebound_stabilization.py` | §4.5 | 15,000 yr ensemble; T*/16 migration | ~20 min |
| `tstar_population.py` | — | HTC population survey (JPL SBDB) | ~2 min |
| `tstar_asteroids.py` | — | Asteroid belt survey (JPL SBDB) | ~5 min |

---

## Results not in the main battery

Two results in the paper come from scripts not included in the main battery,
with notes on their interpretation:

**Phase-locked permutation test (p = 0.035):**
Run `test_phase_locked_permutation.py`. Uses 10⁶ permutations; excluded from
main battery for runtime. Replicates the third independent Jupiter test.

**Synthetic-clone joint test (p = 0.012):**
Reported in §4.4. A subsequent analysis showed the three metrics are not fully
independent (the ratio joint/product ≈ 1), so the joint p-value should be
interpreted with caution. The marginal results (cancellation, R²) are the
primary evidence; the joint test is supplementary.

**N-body results (§4.5):**
Run `rebound_halley.py` for the 3,000-yr ensemble (r = 0.44, p = 0.001) and
`rebound_stabilization.py` for the 15,000-yr run showing migration toward
T*/16 = 71.94 yr. Requires REBOUND.

---

## Data

All perihelion dates are from:
> Yeomans, D. K., Rahe, J., & Freitag, R. S. (1986). The History of Comet
> Halley. *Journal of the Royal Astronomical Society of Canada*, 80, 62.

Julian Day Numbers are from:
> Chirikov, B. V., & Vecheslavov, V. V. (1989). Chaotic dynamics of Comet
> Halley. *A&A*, 221, 146.

The 1986 perihelion (JD 2446470.9518) is from direct observation.

Planetary periods and ephemeris positions use the DE441 ephemeris via
Skyfield (Rhodes 2019) where applicable; simplified Chirikov effective
periods are used in the main battery to avoid external dependencies.

---

## Expected output (key values)

Running `halley_1151_replication.py` should reproduce:

```
Mean period:           76.713006 yr
T*/P̄:                  15.003975
Angular residue:       +1.43°
Deviation from T*/15:  −7.4 days
Monte Carlo p-value:   ~0.009

Jupiter R (circ-lin):  0.47,  p = 0.04
Jupiter r (impulse):   −0.41, p = 0.027 (permutation, n=100,000)
Saturn dist r:         −0.496, p = 0.007 (permutation)
Saturn random-phase:   p = 0.133

Cancellation n=15:     9.4% of σ√15
```

Small variations in Monte Carlo results are expected due to random seed
differences; p-values should agree to 2 significant figures.

---

## Citation

If you use this code, please cite:

```bibtex
@article{BaigetOrts2026halley,
  author  = {Baiget Orts, Carlos},
  title   = {Comet 1P/Halley Completes 15 Orbits in 1,151 Years:
             Commensurability with the Solar System Quasi-Period
             and Evidence for Jupiter--Saturn Dynamical Coupling},
  year    = {2026},
  note    = {arXiv: in preparation}
}

@article{BaigetOrts2026cycle,
  author  = {Baiget Orts, Carlos},
  title   = {A 1151-Year Quasi-Commensurability of the Solar System:
             Empirical Detection, Statistical Characterization,
             and the Anomalous Exclusion of Uranus},
  year    = {2026},
  journal = {arXiv:2604.03049}
}
```

---

## Contact

Carlos Baiget Orts  
Independent researcher, Valencia, Spain  
asinfreedom@gmail.com  
ORCID: [0009-0000-6725-5188](https://orcid.org/0009-0000-6725-5188)

Issues and questions welcome via GitHub Issues.
