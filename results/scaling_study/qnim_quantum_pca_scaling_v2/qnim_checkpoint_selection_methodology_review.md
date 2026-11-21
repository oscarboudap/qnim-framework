# QNIM checkpoint-selection methodology review

## 1. Executive recommendation

**Recommendation: REDESIGN THE STATISTICAL SELECTION METHOD.**

Keep the current delta-method score as an exploratory **screening heuristic only**. Make the official checkpoint decision using measurements that are independent of all screening measurements, with a predeclared bounded-loss test and a fixed confirmation budget. This addresses the selected-winner bias that the current method does not, and permits a transparent finite-sample error guarantee when its independence assumptions hold.

The guarantee would be conditional on the fixed 130-example training-reference set and on finite-shot measurement randomness. It would **not** establish generalization to a population of examples, quantum advantage, or superiority to other models. A distribution-free bound at the frozen \(5\times10^{-4}\) improvement threshold is likely computationally prohibitive; a new version must either accept an often-inconclusive decision or predeclare a scientifically justified minimum effect and resource cap. Do not tune either using validation.

This is a design proposal only. No sampler, circuit, test, or training execution was performed for this review. The frozen V2 protocol and all source code remain unchanged.

## 2. Current method: mathematical review

### 2.1 Multinomial shots and observable covariance

For one fixed input and parameter vector, the ideal circuit defines a bitstring distribution \(P(B=b)\). The current estimator assumes that the \(S=512\) measured bitstrings are independent draws from this distribution, so the count vector is multinomial:

\[
(N_b)_b\sim\operatorname{Multinomial}(S,(P(B=b))_b).
\]

This is a reasonable simulator/shot model when shots are independently sampled from a stable circuit distribution. It is not established by code alone for a real backend: within-job correlations, drift, calibration changes, and pseudorandom-seed behavior are outside the model.

Each bitstring is decoded to \(f(B)\in\mathbb R^{31}\): 16 single-qubit \(P(q_i=0)\) indicators and 15 adjacent \(Z_iZ_{i+1}\) products. For \(\hat\mu=S^{-1}\sum_s f(B_s)\), the code computes the weighted sample covariance of the joint 31-vector and divides it by \(S\):

\[
\widehat{\operatorname{Cov}}(\hat\mu)=
\frac{1}{S}\frac{1}{S-1}
\sum_s(f(B_s)-\hat\mu)(f(B_s)-\hat\mu)^T.
\]

The covariance treatment is correct for observable correlations arising from the same bitstrings, subject to the multinomial sampling assumptions. It is implemented in [`measurement_moments`](../../../src/infrastructure/qnim_v2_support.py#L126), with bit order in [`bitstring_features`](../../../src/infrastructure/qnim_v2_support.py#L111).

### 2.2 Delta-method loss variance

For fixed MLP parameters, let \(g=\nabla_f\ell(\hat\mu)\), where the loss is the clipped cross-entropy
\[
\ell(f)=-\log(\max(p_y(f),10^{-10})).
\]
The plug-in first-order approximation is:
\[
\widehat{\operatorname{Var}}(\ell(\hat\mu))
\approx g^T\widehat{\operatorname{Cov}}(\hat\mu)g.
\]
The implementation evaluates \(g\) at the observed feature mean, through the current ReLU/softmax head, and averages per-example variances assuming independent circuit outcomes across examples. It is a standard local delta approximation, not an exact variance for this nonlinear statistic. It is least trustworthy near a ReLU boundary, where the derivative changes discontinuously, or near the probability floor, where the code sets the feature gradient to zero. Cross-entropy curvature and finite-shot bias are not captured.

The exact implementation is [`mlp_loss_and_feature_gradient`](../../../src/infrastructure/qnim_v2_support.py#L154) and [`candidate_statistics`](../../../src/infrastructure/qnim_v2_support.py#L206).

### 2.3 Paired candidate difference and the reported upper bound

The point estimate pairs candidate and incumbent losses on the same ordered 130 examples:
\[
\hat\Delta=\frac1{130}\sum_i(\hat\ell_{c,i}-\hat\ell_{b,i}).
\]
Pairing controls for fixed differences between examples. It does **not** pair shot outcomes: the implemented variance is
\[
\widehat{\operatorname{SE}}(\hat\Delta)=
\frac{\sqrt{\sum_i(\hat v_{c,i}+\hat v_{b,i})}}{130},
\]
which assumes independent candidate/incumbent shot noise. If their measurement outcomes are correlated, omitting the covariance term can either overstate or understate uncertainty.

The rule forms a one-sided normal upper limit
\[
U=\hat\Delta+z_{1-\alpha/22}\widehat{\operatorname{SE}}(\hat\Delta)
\]
and replaces the incumbent only when \(U<-0.0005\). The code is [`compare_candidates`](../../../src/infrastructure/qnim_v2_support.py#L265); the frozen policy is [protocol.json](./protocol.json).

The \(0.05/22\) Bonferroni allocation would control familywise error at 0.05 **if** every per-comparison test were valid at its allocated level and there were at most 22 comparisons. Bonferroni does not make an approximate delta/normal interval valid. Nor does it fix adaptive reuse of the incumbent's original noisy measurement. The current runner retains those counts and compares future candidates against them; the incumbent was selected partly because its earlier estimate was favorable. Later comparisons therefore are not conditionally centered on a freshly measured incumbent. Candidate existence and early stopping may also depend on earlier selection outcomes. The fixed number “22” matches the intended 20 pre-refit candidates plus two scheduled post-refit candidates; it is not itself a calibration argument. The runner's stored incumbent flow is [`checkpoint_candidate`](../../../scripts/run_qnim_quantum_pca_scaling_v2.py#L468).

The method is valid as a transparent **heuristic** conditional on C. It is not a calibrated confidence bound and should not be described as statistical significance.

## 3. Alternative approaches

| Approach | Assumptions and possible guarantee | Added circuit cost | Adaptivity and architecture fit | Main limitation |
|---|---|---:|---|---|
| **A. Keep delta method as heuristic** | Multinomial shots, independent examples/evaluations, local smoothness, and a useful normal approximation. No guaranteed error rate without evidence that these approximations calibrate in the operating regime. | None beyond current selection: 23 states × 130 examples = 2,990 circuit evaluations and 1,530,880 shots (checkpoint-selection only). | Fits current q16 architecture exactly. Screening all states can remain useful for ranking and debugging. | Reuses noisy incumbent results; synthetic coverage does not establish real-shot calibration. |
| **B. Independent confirmation** | Freeze candidate parameters after screening, then use fresh independent measurement streams for candidate and incumbent. A paired t/delta interval remains approximate; a valid bounded-loss interval can provide a conditional guarantee. | For one comparison and \(R\) independent rounds: \(2·130·R=260R\) circuit evaluations, \(133,120R\) shots. For \(K\) comparisons, multiply by \(K\). | Fresh confirmation makes inference conditional on the screening history and removes the old incumbent's selected measurement noise. It is compatible with the same circuit, labels and C rows. | It does not repair an invalid interval by itself. A shortlist can omit the true best candidate; repeated comparisons still need multiplicity control. It says nothing about examples outside C. |
| **C. Bounded repeated-batch decision rule** | Treat each confirmed per-example candidate/incumbent loss difference as bounded and independently sampled. A fixed-horizon Hoeffding bound is distribution-free under those conditions. A validated empirical-Bernstein method may be tighter but must be specified and audited before use. | Same order as B, with \(R\) selected by a predeclared precision/resource rule. Worst-case Hoeffding cost can be enormous, as quantified below. | Fresh data for each comparison plus preallocated error budgets gives conditional repeated-comparison control. A confidence sequence can permit optional stopping, but only if a time-uniform method is formally specified. | A conservative bound may leave every candidate inconclusive. The target remains the finite C set and a declared finite-shot objective. |

### A — improved diagnostics, but not official selection

If retained, delta-method decisions should be labelled exploratory; record per-example counts and losses, compare predicted variance with independent synthetic Monte Carlo over smooth, high-entropy, low-probability, and ReLU-boundary cases, and document failures rather than tuning the interval until it passes. Fresh independent measurements of both states would also remove stale-incumbent reuse, but would not make a normal approximation exact. Approach A is useful for screening, not sufficient for the final checkpoint claim.

### B — confirmation after adaptive screening

Use the existing C examples without changing their membership. Let screening pick a provisional challenger and incumbent using any method; freeze both complete parameter vectors before confirmation. Then collect new 512-shot measurements for **both** states on all 130 examples. Do not reuse the incumbent's earlier counts. If several challengers are confirmed, either use simultaneous bounds across the frozen shortlist or do a final independent confirmation of the single selected winner. Choosing the lowest noisy confirmation score among several candidates without correcting for that choice recreates selection bias.

Fresh confirmation data are independent of the earlier adaptive path, conditional on the frozen states. This remains true if the shortlist itself was adaptively chosen, but only for claims about the frozen shortlist; it cannot certify that an omitted checkpoint was inferior.

### C — bounded inference and cost feasibility

The per-example clipped loss lies in \([0,L]\), where \(L=-\ln(10^{-10})\approx23.026\). For each frozen candidate/incumbent pair and each independent confirmation replicate \(r\), define
\[
D_{ir}=\ell_{c,ir}-\ell_{b,ir}\in[-L,L],
\]
where each loss is computed from its own fresh 512-shot batch. The target must be stated explicitly as the mean **expected 512-shot loss difference** over the fixed C examples; it is not the infinite-shot loss because applying a nonlinear MLP loss to finite-shot means can be biased.

If the \(130R\) paired differences are independent and bounded, a one-sided Hoeffding upper bound for their average is
\[
U_R=\bar D+
L\sqrt{\frac{2\ln(1/\alpha_j)}{130R}}.
\]
For at most 22 confirmation comparisons, set \(\alpha_j=0.05/22\). This gives a familywise upper error bound of 0.05 by a union bound, conditional on the independence/boundedness assumptions and on each candidate being fixed before its confirmation data. The rule must not reuse observations from earlier comparisons. If independence across PUBs cannot be justified, a more conservative round-level bound is required.

This guarantee is deliberately conservative. Retaining the frozen \(\delta=0.0005\), the Hoeffding radius alone would require roughly
\[
R\ge\frac{2L^2\ln(22/0.05)}{130\delta^2}\approx2.0\times10^8
\]
independent paired batches per reference example to make the worst-case radius that small. That is not a practical budget. Empirical-Bernstein bounds may exploit lower observed variance, but their exact finite-sample theorem, assumptions, and fixed-horizon implementation must be chosen and frozen before execution. If a valid upper bound cannot resolve the comparison within the declared budget, the correct outcome is **inconclusive; retain the incumbent**, not “no difference” or “the candidate is worse.”

No method using repeated shots on C converts the 130 fixed training examples into a population-level validation set. It only quantifies measurement uncertainty for the declared C-set objective.

## 4. Recommended methodology

Use a two-stage **screen-and-confirm** policy in a separately versioned protocol:

1. **Exploration:** retain the current delta score only for screening and telemetry. Save every complete candidate vector, phase, parameters, counts, and screen score. Do not call these scores confidence bounds.
2. **Freeze:** after the optimizer stops, freeze a deterministic candidate order/shortlist and the complete vectors before collecting any confirmation measurements. The shortlist rule and size must be in the protocol. If a shortlist rather than all candidates is confirmed, explicitly limit the claim to that shortlist.
3. **Independent confirmation:** for each challenger in that frozen order, remeasure both it and the current confirmed incumbent on the same 130 C examples, at 512 shots per circuit, using fresh independent streams. Remeasure the incumbent each time; never carry its old screen counts into a confirmatory comparison. Persist seeds/stream policy, per-example outcome counts, actual shots, backend, and timestamps.
4. **Error control:** preallocate total one-sided \(\alpha=0.05\) across the declared maximum 22 comparisons (or the smaller, frozen shortlist size). Use a fixed-horizon bounded-loss procedure whose exact formula is specified in the new protocol. A sequential stopping rule requires a confidence sequence or other time-uniform procedure; do not repeatedly inspect ordinary fixed-time intervals.
5. **Decision:** preserve the existing \(\delta=0.0005\) only if it is retained explicitly in the new protocol. Replace the confirmed incumbent only if the valid upper bound for candidate-minus-incumbent is below \(-\delta\). Otherwise retain the incumbent and record “not established”/“inconclusive”; do not use validation to break ties. If a budget cap is reached while inconclusive, stop confirmation and report that outcome.
6. **Reporting:** keep validation metrics reporting-only. Report the confirmed objective and interval conditional on C and finite-shot sampling; do not describe it as population generalization, quantum advantage, or evidence of useful quantum representations by itself.

The seed policy must create independent confirmation streams, not reuse one cached incumbent estimate. For local simulation, derive and record distinct reproducible child seeds. On hardware, randomize or interleave candidate/incumbent execution order to reduce time-order confounding, while acknowledging that backend drift remains a separate uncertainty source.

## 5. Cost estimate

The present selection path evaluates the initial state, 20 post-blocking pre-refit states, and two post-refit states: 23 complete candidates, or 2,990 C-example circuit evaluations and 1,530,880 shots at 512 shots each. This excludes optimizer minibatches, warm-start/refits, and validation reporting.

Independent confirmation adds:

- One candidate/incumbent comparison: **260R** circuit evaluations and **133,120R** shots.
- \(K\) comparisons: **260KR** evaluations and **133,120KR** shots.
- Illustrative only: \(K=3, R=10\) costs 7,800 evaluations and 3,993,600 shots.
- Confirming all 22 possible comparisons at \(R=10\) costs 57,200 evaluations and 29,286,400 shots.

These are formulas, not proposed authorized budgets. The Hoeffding calculation shows that a worst-case formal guarantee at \(5\times10^{-4}\) is unlikely to be feasible. A new protocol must set a hard resource cap and accept inconclusive outcomes; any larger practical effect threshold needs a scientific rationale and advance declaration, not adjustment after seeing validation.

## 6. Assumptions and limitations

- The formal bounded rule assumes independent measurement batches and independence of paired differences across reference examples. A sampler's fixed seed is reproducible bookkeeping, not by itself evidence of statistical independence.
- It covers shot-sampling error only. It does not include uncertainty from sampling a broader event population, optimizer randomness, model misspecification, hardware drift, or PCA/dataset shift.
- Confirmation on C is permissible for training-only checkpoint selection because C is within the original training split; it does not make C an external test set.
- The confirmation target is the mean expected loss of a **512-shot batch**. It must not be conflated with the ideal infinite-shot model objective.
- The current \(10^{-10}\) loss floor makes losses bounded but introduces a clipping kink; the proposed bound remains valid for the implemented clipped loss because it is bounded, but it does not make the loss an unbiased estimator of an unclipped objective.
- If candidate ranking is based on the same confirmation outcomes used to report the winner, multiplicity correction or another independent final confirmation remains necessary.
- No probability, coverage, or runtime value from a real V2 sampler run is available or reconstructed in this review.

## 7. Protocol version and exact next implementation steps

A **new protocol version is required**. The V2 protocol is explicitly frozen and must remain byte-for-byte unchanged. The selection estimand, confirmation schedule, independent seed policy, statistical bound, alpha allocation, candidate shortlist rule, resource cap, and inconclusive outcome all affect the scientific method. Create a separately identified protocol (for example, QNIM_QUANTUM_PCA_SCALING_V3); do not silently amend V2.

Before any code or execution:

1. Freeze the estimand: expected clipped 512-shot loss difference over the unchanged C indices.
2. Decide whether all candidate states or a fixed-size shortlist are formally covered; state the limits of any shortlist.
3. Select and independently review the exact fixed-horizon bounded confidence formula, its independent-unit definition, and total alpha allocation. Keep Hoeffding as a conservative reference; do not invent a variance-adjusted interval.
4. Set a maximum confirmation budget and a rule for unresolved comparisons. Preserve A=2,275, B=195, C=130, validation isolation, architecture, optimizer updates, and dataset hashes.
5. Implement screening and confirmation as separate records/state machines. Persist fresh candidate and fresh incumbent outcome counts, full vectors/hashes, per-example losses, seed policy, actual shots, interval inputs, bounds, and decision.
6. Add deterministic NumPy-only synthetic tests for null rejection, adaptive incumbent chains with fresh data, repeated looks/alpha allocation, bounded-loss edge cases, duplicate parameter states, and “inconclusive at budget.” Validate coverage against the selected theorem's assumptions. Do not use Qiskit in these tests.
7. Conduct a static implementation audit first. Any later sampler smoke or training requires separate explicit authorization; this review authorizes neither.

## 8. Final recommendation

**REDESIGN THE STATISTICAL SELECTION METHOD.** Retain the current delta method only as screening telemetry, and use independent confirmation with a predeclared bounded-loss decision procedure for official checkpoint selection. The finite-sample guarantee is conditional and may be too costly at the current \(0.0005\) threshold; if the bound is unresolved within a frozen budget, report inconclusive and keep the incumbent. Keep V2 untouched and version these changes separately.
