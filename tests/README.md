# Test & Experimental Versions

This directory contains experimental and historical versions of the ΨQRH Plasma simulation.

## Files

### Experimental Versions

- **PsiPlasma_base.py** - Basic version without adaptive PID control
  - Fixed parameters (ω_plasma=10.0, K_lider=55.0)
  - Simpler implementation for testing
  - Achieves ~0.9 max sync with good stability

- **PsiPlasma_static.py** - Static snapshot generator
  - Generates PNG snapshots at specific frames (10, 30, 50, 70)
  - No animation, faster execution
  - Useful for documentation and visualization

- **PsiPlasma_pid.py** - PID control experiments
  - Tests different PID configurations
  - Experimental gain scheduling
  - May not reach production performance

### Deprecated Versions

- **PsiPlasma.py** - Original version with basic leader core
  - Historical reference
  - Less optimized than current version
  - Kept for comparison

- **PsiPlasma_.py** - Experimental variant
  - Testing alternative approaches
  - May have incomplete features

- **PsiPlasma90s.py** - 90-second variant
  - Alternative implementation
  - Different parameter sets

## Usage

```bash
# Run any experimental version
cd tests/
python PsiPlasma_base.py

# Generate static snapshots
python PsiPlasma_static.py
```

## Notes

⚠️ **These versions are NOT production-ready**
- Use `../psi_plasma.py` for production
- Experimental versions may have bugs or suboptimal performance
- Kept for research and testing purposes

## Comparison

| Version | Type | Performance | Status |
|---------|------|-------------|--------|
| `psi_plasma.py` (main) | Production | Sync=1.000, 100% success | ✅ Use this |
| `PsiPlasma_base.py` | Test | Sync~0.9, stable | Testing only |
| `PsiPlasma_static.py` | Utility | N/A | Snapshot generation |
| Others | Deprecated | Variable | Historical reference |

---

**For production use, always use:** `../psi_plasma.py`
