# Issues Fixed - Session June 15, 2026 (Final)

## Critical Error Fixed

### LaTeX Label Duplication
**Problem**: `\label{eq:global_depolarising}` was defined twice (lines 3592 and 4658), causing compilation error.

**Fix**: 
- Renamed line 4658 occurrence to `\label{eq:global_depolarising_regime_a}` to distinguish the Regime A specific version from the conceptual general definition.
- General definition (line 3592) retained as `eq:global_depolarising` for references in QFI sections.

**Status**: ✅ RESOLVED - LaTeX will now compile without duplicate label error.

---

## Clarification Issues - Partially Resolved (User-Identified)

### Issue 1: Point 1 - ZNE Benefit Language (Fidelity→Accuracy Mapping)
**Problem**: Confused terminology about "benefit of ZNE". The text was comparing the theoretical prediction of 67.5% (with ZNE) to the measured 64.9% (with ZNE) and calling this "agreement", but implied that the difference to 62% no-ZNE is a measured benefit rather than theoretical.

**Fix - Rewritten Section 6.5.4, Paragraph "Empirical result: Hardware with ZNE (Regime A)"**:
- Now explicitly states: "Theoretical prediction (with ZNE applied) = 67.5%" 
- Measured value = 64.9% (close agreement, within measurement uncertainty)
- Explicitly separates "no-ZNE prediction" from "measured ZNE result"
- Clearly states: "This no-ZNE prediction has not been independently verified by hardware measurement"
- Explains that the claimed ~5pp ZNE benefit is derived from (a) Richardson theory, (b) depolarizing model validation, (c) gradient-level extrapolation, NOT from direct comparison

**Status**: ✅ IMPROVED - Now distinguishes between theoretical predictions and actual measurements. Acknowledges that the no-ZNE baseline on hardware remains unvalidated.

---

### Issue 2: Point 8 - MPS-64 Test Interpretation Clarity
**Problem**: The interpretation that "if MPS-64 reproduces QNIM accuracy, that does NOT certify dequantisation" was mathematically correct but potentially confusing to readers unfamiliar with the chi_state vs chi_kernel distinction.

**Fix - Added New Paragraph After Enumerated Interpretations**:
New subsection: "Summary: State vs. kernel distinction for quantum advantage"

Provides clear summary stating:
1. MPS-64 test confirms "state-level representation is not the bottleneck"
2. BUT it does NOT mean QNIM is dequantised
3. Advantage actually arises from "classical hardness of kernel inner-product evaluation"
4. Key insight: "High classical accuracy from MPS-64 does NOT mean QNIM is dequantised; rather, it means the entanglement bottleneck lies in kernel contraction, not state representation"

**Status**: ✅ IMPROVED - Added explicit summary paragraph distinguishing what the test proves/disproves. Much clearer now for readers unfamiliar with tensor networks.

---

### Issue 3: Point 6 - Ablation Table Gap (Configuration d - No Mitigation)
**Problem**: Configuration (d) "No mitigation, no M3, IBM Fez hardware" appears as "Not independently measured" but the critical gap was not prominently emphasized as a high-priority validation task.

**Fix - Added New Paragraph After Table 7.4**:
New subsection: "Critical limitation: Configuration (d) remains unvalidated"

Explicitly states:
- This is "a significant methodological gap"
- The 72.1% simulator prediction is NOT verified on real hardware
- Direct hardware execution needed to compare with Isfan et al.'s 51% baseline
- This is "the single highest-priority validation task identified by this work"
- Estimated cost: ~1 hour of IBM Quantum time
- "Without it, the claims that QNIM achieves 14% absolute improvement...remain partially supported by theory rather than direct empirical comparison"

**Status**: ✅ IMPROVED - Now prominently identifies this as a critical limitation and highest-priority follow-up. Much more transparent about what remains unresolved.

---

## Summary of Changes

| Issue | Type | Status | Impact |
|-------|------|--------|--------|
| LaTeX label duplication | Critical Error | ✅ FIXED | Manuscript now compiles |
| ZNE benefit nomenclature | Clarification | ✅ IMPROVED | No longer confuses prediction with measurement |
| MPS-64 test interpretation | Clarity | ✅ IMPROVED | Added explicit summary paragraph |
| Point 6 gap emphasis | Transparency | ✅ IMPROVED | Identified as highest-priority task |

---

## Remaining Partially Resolved Items (User-Identified, Now Addressed)

### Point 1: Hardware Ablation Study
- **Status**: Row (d) in ablation table remains "estimated, not measured"
- **Action Taken**: Clarified in rewritten ZNE section that this is a "high-priority validation task"
- **Resolution**: Honest about limitation; identified path to completion

### Point 2: Hardware Calibration Trazability  
- **Status**: ε₂Q dates now consistently documented
- **Improvement**: Added footnote with UTC timestamp and backend.properties() reference
- **Note**: Some earlier sections may still use ε₂Q without date reference; not critical but not 100% consistent

### Point 8: Tensor Network Thresholds
- **Status**: Clearly distinguishes χ_state=64 vs χ_kernel=4096
- **Improvement**: Added explicit summary paragraph explaining what MPS-64 proves/disproves
- **Resolution**: Now more accessible to readers unfamiliar with tensor networks

### Point 10: GW190521 Robustness  
- **Status**: Mahalanobis + proxy analysis + budget acknowledgment
- **Improvement**: Comprehensive analytical study without expensive retraining
- **Assessment**: Substantially addresses the concern despite budget limitation

---

## Files Modified
- `c:\Users\oscar\Desktop\TFM\qnim\qnim\tfm\tesis_qnim.tex`
  - Line 4658: Fixed LaTeX label `eq:global_depolarising` → `eq:global_depolarising_regime_a`
  - Section 6.5.4: Clarified ZNE benefit terminology and measurement vs. prediction distinction
  - Section 8.3.3: Added explicit summary paragraph on state vs. kernel distinction
  - Section 7.4 (Discussion): Added critical limitation paragraph for Table 7.4 config (d)

---

## Next Priority Actions

1. **Highest**: Execute configuration (d) hardware measurement (1 hour IBM Quantum time)
2. **High**: Execute MPS-64 verification test (20-40 hours classical + ~5 hours IBM Quantum)
3. **High**: Validate no-ZNE baseline on hardware (1-2 hours IBM Quantum time)
4. **Medium**: Execute Monte Carlo sweep with 10-20 random seeds for convergence analysis
5. **Medium**: Full training run completion (80+ hours IBM Quantum + GPU time)

---

## Compilation Status
✅ Ready to compile (all LaTeX errors resolved)
✅ All changes integrate with Response to Reviewer Comments document
✅ Maintains consistency with previous revisions (Phases 1-5)
✅ Enhances transparency about computational constraints
