# hPSC Biophysical Parameters — Literature Review

Parameters used to calibrate the active Brownian particle simulation for
human Pluripotent Stem Cells (hPSCs, including iPSCs and ESCs).

---

## Cell Radius

**Used in simulation: R = 6.5 µm**

| Source | Value | Notes |
|--------|-------|-------|
| ResearchGate / Ali Babaie (Mayo Clinic), Luna II counter, n=3 on hiPSCs | diameter **11–14 µm**, mean ~12.6 µm | Detached/suspended single cells |
| General mammalian stem cell literature | diameter **10–15 µm** | High nuclear-to-cytoplasm ratio |
| Wadkin et al. 2017 (PMC5428844) | ~10–15 µm (implied from displacement data) | hESC on Matrigel |

**Conclusion:** dissociated individual hPSC diameter ≈ 10–15 µm → radius **5–8 µm**.
We use **R = 6.5 µm** as a representative mid-range value.

---

## Migration Speed

**Used in simulation: v = 0.004 µm/s ≈ 0.24 µm/min ≈ 14.4 µm/h**

| Source | Cell type | Speed |
|--------|-----------|-------|
| Wadkin et al. 2017, *Sci. Reports* ([PMC5428844](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5428844/)) | hESC (H9), individual cells on Matrigel | **16.25 µm/h** median (range ~12–25 µm/h) |
| Wadkin et al. 2017 (Cell Tracer-labeled) | hESC, individual cells | **11.51 µm/h** median |
| Hamada et al. 2018, *PLOS ONE* ([PMC6130871](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6130871/)) | hiPSC undifferentiated, individual cells | ~**5–10 µm/h** (slowest group; speed increases with differentiation) |
| Hadjiantoniou et al. 2024, *Life* ([PMC11595361](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11595361/)) | hPSC colonies (H9, AD3, CaSR hiPSC) | v_th = **7.9–34 µm/h** across lines and phenotypes |
| Wadkin et al. 2018, arXiv:1803.00063 | hESC, single isolated cells | Correlated random walk; consistent with 10–20 µm/h range |

**Conclusion:** individual hPSC migration speed ≈ **5–20 µm/h** (0.08–0.33 µm/min),
central value ~10–16 µm/h. Undifferentiated cells tend toward the lower end.
We use **v = 0.004 µm/s (= 14.4 µm/h)** as a representative value.

---

## Derived Simulation Parameters (Stokes-Einstein-Debye, 37 °C, water-like medium)

| Quantity | Value |
|----------|-------|
| R | 6.5 µm |
| T | 310 K (37 °C) |
| η | 1×10⁻³ Pa·s |
| D_T | ~0.035 µm²/s |
| D_R | ~6.2×10⁻⁴ rad²/s |
| τ_R = 1/D_R | ~1613 s (~27 min) |
| v | 0.004 µm/s (14.4 µm/h) |
| l_p = v/D_R | ~6.5 µm |

---

## Caveats

- Cell size on substrate (spread/adherent) is larger than in suspension; the 11–14 µm figure is for rounded suspended cells.
- Speed depends strongly on substrate (Matrigel vs. fibronectin), medium (mTeSR1 vs. E8), and ROCK inhibitor use.
- D_T and D_R are computed from Stokes-Einstein-Debye assuming a sphere in water — an approximation for cells crawling on a 2D substrate.
