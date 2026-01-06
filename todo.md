# PDE Status Summary

## Basic PDEs (7/7)
All working correctly. No fixes needed.

## Physics PDEs (16/16)
All working correctly after fixes. One optional enhancement:
- `swift-hohenberg` - could add quintic term option (low priority)

## Biology PDEs (10/18 working, 8 need fixes)

**Working (10):**
| Status | PDEs |
|--------|------|
| EXCELLENT | bacteria-flow, brusselator, cyclic-competition, fisher-kpp, schnakenberg |
| GOOD | allen-cahn-standard, harsh-environment, heterogeneous, keller-segel, topography |

**Need Fixes (8):**

| PDE | Issue | Priority |
|-----|-------|----------|
| fitzhugh-nagumo | Remove /3 from u³ term | High |
| gierer-meinhardt | Use standard (a,b,c) params | High |
| vegetation | Add water diffusion ∇²w + fix Dn validation | High |
| keller-segel | Add logistic growth | High |
| cyclic-competition | Update to asymmetric Lotka-Volterra | Medium |
| cross-diffusion | Use Schnakenberg kinetics | Medium |
| allen-cahn | Noise diffuses too fast | - |
| sir | No epidemic wave, needs localized IC | - |
| turing-conditions | Not in Turing regime | - |
| immunotherapy | Optional - valid 2-species variant | Low |

**Totals: 33/41 PDEs working (80%)**
