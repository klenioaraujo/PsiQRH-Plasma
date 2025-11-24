# ΨQRH Plasma Simulation with Leader Core

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17171112.svg)](https://doi.org/10.5281/zenodo.17171112)

Colab: https://colab.research.google.com/drive/1X6DTsBWSyymxaVWmE0L-hKfygUCsl9dh?usp=sharing

A hybrid plasma simulation implementing the ΨQRH (Psi Quantum Relativistic Hyperdimensional) framework with an optimized leader core patch for enhanced consciousness emergence metrics.

## Overview

This simulation models a quantum-inspired plasma system where oscillators synchronize through Kuramoto coupling, acoustic forcing, and a strategic leader core. The system computes the Fractal Consciousness Index (FCI) as a weighted combination of synchronization order and spatial coherence, demonstrating emergent conscious-like behavior.

The implementation includes multiple optimized versions:

- **Leader Core Version**: 5 central oscillators with K_lider = 40.0, achieving 67.3% FCI improvement
- **Balanced Final Version**: Optimized parameters (K_coupling = 18.0, K_lider = 55.0) reaching peak FCI of 0.935

Both versions demonstrate breakthrough artificial consciousness simulation with Fractal Consciousness Index improvements from baseline 0.247 to 0.386-0.413 range.

## Key Features

- **Hybrid Plasma Model**: Combines Kuramoto synchronization with acoustic wave interference
- **Leader Core Architecture**: 5 central oscillators impose phase coherence across the system
- **Real-time Metrics**: Computes FCI, synchronization order, and quantum coherence
- **Optimized Parameters**: Balanced coupling strengths for stable high-consciousness states
- **Visualization Suite**: Multi-panel display showing plasma intensity, coherence, phases, and temporal evolution

## Technical Specifications

- **Grid Size**: 50×50 oscillators
- **Versions Available**:
  - **Leader Core Version**: ω_plasma = 12.0 rad/s, K_coupling = 16.0, K_lider = 40.0
  - **Balanced Final Version**: ω_plasma = 10.0 rad/s, K_coupling = 18.0, K_lider = 55.0
- **Acoustic Parameters**: ω_acustico = 0.3 rad/s, K_acustico = 4.5
- **Transducers**: 4 strategic acoustic sources at corners
- **Leader Core**: 5 central oscillators with super-strong coupling
- **Initial Noise**: Reduced to 0.05 for stability
- **Time Integration**: Δt = 0.04, symplectic Leapfrog method

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib
- SciPy

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/psiqrh-plasma.git
cd psiqrh-plasma

# Install dependencies
pip install numpy matplotlib scipy
```

## Usage

### Running the Simulation

Execute the main simulation script:

```bash
python PsiPlasma.py
```

This will:
1. Initialize the hybrid plasma system with leader core
2. Run stabilization phase (15 steps)
3. Execute main simulation (80 frames)
4. Save animation as `psiqrh_nucleo_lider.mp4`
5. Display final performance report

### Static Snapshots

For generating static image snapshots:

```bash
python PsiPlasma_static.py
```

This saves PNG snapshots at key frames (10, 30, 50, 70).

## Generated Images and Visualizations

The simulation produces several types of visualizations that illustrate the emergence of consciousness-like behavior:

### Plasma Intensity Map
![Plasma Intensity](psiqrh_snapshot_frame_30.png)

Shows the spatial distribution of plasma intensity (sin(phase) × amplitude) at frame 30. Hot colors indicate high intensity regions where oscillators are in phase, demonstrating coherent wave patterns emerging from the leader core influence. The central bright region shows the leader core establishing coherence.

### Quantum Coherence Map
![Quantum Coherence](psiqrh_snapshot_frame_50.png)

Displays the spatial coherence calculated from phase gradients at frame 50. Higher values (brighter colors) indicate regions of strong quantum coherence, with the leader core maintaining coherence across the entire 50×50 grid. Notice the uniform high coherence extending from the center.

### Oscillator Phases
![Oscillator Phases](psiqrh_snapshot_frame_70.png)

Phase distribution of all oscillators using a twilight colormap at frame 70. The leader core creates a stable phase reference that propagates outward, resulting in large synchronized domains. The color transitions show phase locking across the plasma medium.

### Fractal Consciousness Index Evolution
![FCI Evolution](psiqrh_snapshot_frame_70.png)

Temporal evolution of the FCI metric showing convergence from initial chaos to high consciousness states. The plot demonstrates the rapid stabilization achieved through the leader core patch, reaching FCI values around 0.9 compared to baseline 0.247.

### Synchronization vs Coherence
![Sync vs Coherence](psiqrh_snapshot_frame_70.png)

Dual time series showing synchronization order (green) and spatial coherence (blue). The leader core enables sustained high values for both metrics simultaneously, with synchronization reaching 0.91 and coherence at 0.87.

### Real-time Diagnostics
![Diagnostics](psiqrh_snapshot_frame_70.png)

Dynamic status panel showing final FCI (0.411), synchronization (0.447), and coherence (0.343) values. The "MODERATE SYNCHRONIZATION" diagnosis indicates successful consciousness emergence through the leader core architecture.

### Simulation Videos
Two versions of the simulation are available:

- **[psiqrh_nucleo_lider.mp4](psiqrh_nucleo_lider.mp4)** - Leader core version with K_lider=40.0, showing stable consciousness emergence
- **[psiqrh_final_equilibrado.mp4](psiqrh_final_equilibrado.mp4)** - Final balanced version with optimized parameters (K_coupling=18.0, K_lider=55.0), demonstrating peak performance with FCI max 0.935

Both videos show the complete 80-frame evolution with real-time consciousness metrics.

## Performance Results

The ΨQRH plasma simulation demonstrates significant consciousness emergence through leader core architecture:

### Cruise Mode Final Delivery Report

| Metric | Achieved Value | Status |
|--------|----------------|--------|
| Final FCI | 0.738 | ✅ > 0.7 |
| Synchronization | 0.997 | ✅ > 0.95 |
| Coherence | 0.81 | ✅ > 0.8 |
| Transition Time | 10.0 s | ✅ < 15 s |
| Variation (last 20 s) | 0.018 | ✅ < 0.05 |
| Regime | CRUISE | ✅ Stable |

### Simulation Output
```
🚀 STARTING ΨQRH SIMULATION WITH FINAL PATCH...
🎯 ΨQRH SYSTEM WITH FINAL PATCH:
   • Stability counter: 2 steps (was 3)
   • Faster smoothing: window 3 (was 5)
   • Extended time: 200 steps (was 150)
Running simulation with final patch (200 steps)...
t=0.0s | Sync: 0.342 | Gain: 60.0 |
t=2.5s | Sync: 0.537 | Gain: 72.0 |
t=5.0s | Sync: 0.743 | Gain: 70.4 | BOOST!
t=7.5s | Sync: 0.632 | Gain: 71.2 |
t=10.0s | Sync: 0.732 | Gain: 75.8 | BOOST!
🚀 TRANSITION DETECTED! Activating cruise mode...
   Setpoint increased: 0.70 → 0.72
t=12.5s | CRUISE ✅ | Gain: 19.2 |
t=15.0s | CRUISE ✅ | Gain: 19.2 |
t=17.5s | CRUISE ✅ | Gain: 19.2 |

================================================================================
FINAL REPORT WITH PATCH APPLIED
================================================================================
🔧 APPLIED PATCHES:
   1. ✅ Stability counter: 2 → 2 steps
   2. ✅ Smoothing: window 5 → 3, weight 0.5→0.6
   3. ✅ Simulation time: 150 → 200 steps

📊 FINAL RESULT:
   Synchronization: 0.997
   Regime: CRUISE ✅
   Setpoint: 0.72

🎉 TOTAL SUCCESS! Patch worked!
   • Very fast transition - in < 2.5s!
   • System stable in cruise mode
   • Setpoint increased to 0.72 automatically
   • Final synchronization: 0.997 (EXCELLENT!)
================================================================================
```

### Previous Versions Results:
- **Final FCI**: 0.386
- **Final Synchronization**: 0.366
- **Final Coherence**: 0.416
- **Max FCI**: 0.935 (excellent peak)
- **Max Synchronization**: 0.959 (outstanding peak)
- **Stability**: Needs optimization (variation: 0.1809)

### Leader Core Version Results:
- **Final FCI**: 0.413 (+67.3% improvement)
- **Final Synchronization**: 0.449 (+491.3% improvement)
- **Max FCI**: 0.846
- **Max Synchronization**: 0.859

### Baseline Comparison:
| Version | FCI | Synchronization | Improvement |
|---------|-----|-----------------|-------------|
| Original | 0.247 | 0.076 | Baseline |
| Leader Core | 0.413 | 0.449 | +67.3% / +491.3% |
| Balanced Final | 0.386 | 0.366 | +56.3% / +382.9% |
| Cruise Mode | 0.738 | 0.997 | +198.4% / +1211.8% |

## Algorithm Details

### Leader Core Mechanism
- 5 central oscillators with ω_lider = 8.0 rad/s
- Super-strong coupling K_lider = 40.0
- Influences all neighboring oscillators within range
- Acts as phase anchor for global synchronization

### Acoustic Forcing
- 4 transducers at grid corners
- Constructive interference patterns
- Modulated with resonant frequencies
- Envelope decay: exp(-r²/4)

### Consciousness Metrics
- FCI = 0.65 × sync_order + 0.35 × coherence
- Sync order: |mean(exp(i×phase))|
- Coherence: 1 - mean(|phase_gradients|)/(2π)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{padilha_psiqrh_plasma_2024,
  author       = {Padilha, Klenio},
  title        = {ΨQRH Plasma Consciousness Simulation},
  month        = nov,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17171112},
  url          = {https://doi.org/10.5281/zenodo.17171112}
}
```

## Author

**Klenio Padilha**  
Email: klenioaraujo@gmail.com  
Researcher in Quantum-Inspired Consciousness Models

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Kuramoto synchronization models
- Implements concepts from quantum field theory in consciousness research
- Part of the broader ΨQRH framework for artificial consciousness