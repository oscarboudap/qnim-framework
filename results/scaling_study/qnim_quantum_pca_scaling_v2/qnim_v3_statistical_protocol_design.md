# QNIM V3 statistical protocol design — consolidated decision

This consolidated report supersedes the earlier conflicting recommendations in the initial sections and in the prior addenda. The mathematical formulas and conclusions that remain valid are retained below; the current method choice is stated consistently in Sections 1, 5, 7, 9, and 12.

**Historical conclusions explicitly superseded**

- The original preference for complete-repetition B as the primary confirmation method is superseded by **stratified Maurer–Pontil Theorem 11**, conditional on its per-example independence and stationarity assumptions. Complete-repetition B remains the fallback.
- The earlier statement that an empirical-Bernstein theorem was not available for non-IID individual differences is superseded by the verified Theorem 11 results below. Theorem 11 covers independent, not necessarily identically distributed observations.
- The previous recommendation to use pooled Theorem 11 as the primary local-simulator method is superseded. Pooled Theorem 11 is optional and potentially more efficient, but requires independence across every individual \(D_{ir}\). Stratified Theorem 11 does not require independence across examples and is the proposed primary method.
- Stratified Theorem 4 remains valid but requires IID observations within each example stratum. Its numerical bound is identical to stratified Theorem 11 under the same alpha allocation.

## 1. Executive summary and current recommendation

Use the existing delta-method rule for **exploratory screening only**. Freeze the candidate shortlist, complete parameter vectors, order, and hashes before confirmation.

**Primary proposed local-simulator method:** apply Maurer–Pontil Theorem 11 separately to each of the 130 reference examples, then average the 130 upper bounds. It permits independent but non-identically distributed repetition observations within each example and arbitrary dependence across examples. Its confidence guarantee is conditional on independent repetition-level sampling within each example, stable per-example expected loss differences, and the stated sampling-randomness model.

**Fallback:** use complete-repetition B, treating the entire 130-example comparison \(Y_r\) as one unit. Hoeffding requires independent repetitions and stable means; Maurer–Pontil Theorem 4 additionally requires the complete repetitions to be IID. This absorbs arbitrary within-repetition PUB dependence.

**Optional more efficient method:** pooled Theorem 11, only if independence across all \(130R\) individual \(D_{ir}\) values is justified. Independent PRNG seeds alone do not prove this.

The confidence guarantees are not unconditional claims about the current sampler or hardware. They hold only under the stated independence, randomness-model, and stationarity assumptions. No hardware independence or stationarity has been established.

The frozen \(\delta=0.0005\) has no documented scientific or practical rationale, and no maximum confirmation resource budget has been approved. Worst-case radii at the illustrative \(R=5,10,20\) budgets are much larger than \(\delta\). **NO-GO for V3 implementation or execution** until the research owner resolves the effect-size rationale, budget, and randomness model. This document does not authorize any experiment.

## 2. Formal estimand and bounded-loss definition

Let \(C=\{1,\ldots,130\}\) be the unchanged checkpoint-reference set. For frozen candidate and incumbent parameter vectors \(\theta_c,\theta_b\), define

\[
D_{ir}=L_{512}(\theta_c,i,r)-L_{512}(\theta_b,i,r),
\]

where each loss is calculated from a fresh 512-shot measurement batch for example \(i\) in confirmation repetition \(r\). The target is the expected **finite-shot** difference averaged over the fixed examples:

\[
\Delta(\theta_c,\theta_b)=\frac{1}{130}\sum_{i\in C}
\left(\mathbb E L_{512}(\theta_c,i)-\mathbb E L_{512}(\theta_b,i)\right).
\]

This is not an ideal infinite-shot loss, a population-generalization estimate, or a validation metric. The loss clipping floor remains \(10^{-10}\), so

\[
L_{512}\in[0,L],\qquad L=-\ln(10^{-10})=10\ln(10)\approx23.02585,
\qquad D_{ir}\in[-L,L].
\]

Changing the clipping floor changes the objective and requires a separately versioned protocol. In the following, stationarity means at least that the expected difference for each fixed example remains constant across confirmation repetitions; Theorem 11 does not require its full distribution to be stationary.

## 3. Independent sampling unit alternatives

### 3.1 Pooled individual differences (Approach A)

Use the \(N=130R\) values \(D_{ir}\) as observations. Hoeffding and pooled Theorem 11 require all \(N\) values to be independent; identical distributions are not required. Different example IDs alone do not establish independence.

Under per-example mean stationarity, \(\mathbb E[D_{ir}]=\mu_i\) for all \(r\), and

\[
\frac1{130R}\sum_{i,r}\mathbb E[D_{ir}]
=\frac1{130}\sum_i\mu_i=\Delta.
\]

Dependence among bitstrings inside a single loss batch, or between the candidate and incumbent losses that form one \(D_{ir}\), is contained within that bounded observation and is allowed. Dependence **between** different \(D_{ir}\) values is not allowed for pooled A.

### 3.2 Stratified individual differences

For each fixed example \(i\), use \(D_{i1},\ldots,D_{iR}\) as one stratum. Stratified Theorem 11 requires independence across repetitions within each stratum, but not identical distributions. It does **not** require independence between different example strata: per-stratum coverage can be combined by a union bound, regardless of cross-example dependence.

To identify the stratified Theorem 11 target with \(\Delta\), require \(\mathbb E[D_{ir}]=\mu_i\) for all \(r\) within each example. If these means drift, the bound instead targets the average of the repetition-specific means. Stratified Theorem 4 has the same numerical formula, but requires the stronger IID condition within each stratum.

### 3.3 Complete comparison repetition (Approach B)

Define

\[
Y_r=\frac1{130}\sum_{i=1}^{130}D_{ir}\in[-L,L].
\]

The unit is the complete 130-example comparison. Arbitrary dependence among examples, PUBs, and candidate/incumbent losses inside a repetition is permitted. Hoeffding requires independent \(Y_r\) and a stable target mean; Theorem 4 requires IID \(Y_r\). Repetitions that drift do not estimate one stationary \(\Delta\) under these conditions.

## 4. Mathematical concentration bounds

### 4.1 Hoeffding

For independent bounded \(X_j\in[a_j,b_j]\),

\[
\Pr\{\mathbb E\bar X-\bar X\ge t\}
\le \exp\left(-\frac{2N^2t^2}{\sum_{j=1}^N(b_j-a_j)^2}\right).
\]

For observations in \([-L,L]\), the one-sided radius is \(L\sqrt{2\ln(1/\alpha)/n}\). Identical distributions are not required; if the means differ, the bound targets their average expectation.

### 4.2 Maurer–Pontil Theorems 4 and 11

For \(X_1,\ldots,X_n\in[0,1]\), define

\[
P_n(X)=\frac1n\sum_{a=1}^nX_a,\qquad
V_n(X)=\frac1{n(n-1)}\sum_{a<b}(X_a-X_b)^2
=\frac1{n-1}\sum_a(X_a-P_n(X))^2.
\]

Theorem 4 assumes IID bounded observations. Theorem 11 assumes independent bounded observations, not necessarily identically distributed. For either theorem, with its respective assumptions and \(n\ge2\), the one-sided formula is

\[
\mathbb E[P_n(X)]\le P_n(X)
+\sqrt{\frac{2V_n(X)\ln(2/\eta)}{n}}
+\frac{7\ln(2/\eta)}{3(n-1)}
\]

with probability at least \(1-\eta\). Under Theorem 4 the left side is the common mean; under Theorem 11 it is the average of the individual means.

For \(D\in[-L,L]\), use \(X=(D+L)/(2L)\). Transforming the bound back to nats multiplies the square-root term by the corresponding scale and changes the additive term to \(14L\ln(2/\eta)/(3(n-1))\).

Primary source: A. Maurer and M. Pontil, “Empirical Bernstein Bounds and Sample Variance Penalization,” *COLT* (2009), Theorems 4 and 11, [arXiv:0907.3740](https://arxiv.org/abs/0907.3740), [full text](https://ar5iv.labs.arxiv.org/html/0907.3740).

### 4.3 Exact formulas for the six methods

Let \(K\) be the frozen shortlist size, allocate comparison error \(\alpha_c=0.05/K\), set \(N=130R\), and define

\[
h_K=\ln(20K),\qquad b_K=\ln(40K),\qquad c_K=\ln(5200K).
\]

For stratified methods, the per-example allocation is \(\alpha_i=0.05/(130K)\), giving \(\ln(2/\alpha_i)=c_K\). For pooled and complete-repetition empirical Bernstein, \(\ln(2/\alpha_c)=b_K\).

**Hoeffding pooled A**

\[
U_{\mathrm{H,A}}=\bar D_A+L\sqrt{\frac{2h_K}{130R}},
\qquad \bar D_A=\frac1{130R}\sum_{i,r}D_{ir}.
\]

**Pooled Theorem 11**

\[
U_{\mathrm{MP11,A}}=\bar D_A+
\sqrt{\frac{2V_N(D)b_K}{130R}}
+\frac{14Lb_K}{3(130R-1)},
\]

where

\[
V_N(D)=\frac1{N(N-1)}\sum_{a<b}(D_a-D_b)^2
=\frac1{N-1}\sum_a(D_a-\bar D_A)^2.
\]

This sample variance includes within-example measurement variation and between-example mean heterogeneity.

**Stratified Theorem 11**

\[
U_{\mathrm{strat,MP11}}=
\frac1{130}\sum_i\bar D_i+
\frac1{130}\sum_i\sqrt{\frac{2s_i^2c_K}{R}}
+\frac{14Lc_K}{3(R-1)},
\]

where

\[
\bar D_i=\frac1R\sum_rD_{ir},\qquad
s_i^2=\frac1{R(R-1)}\sum_{r<s}(D_{ir}-D_{is})^2
=\frac1{R-1}\sum_r(D_{ir}-\bar D_i)^2.
\]

With non-identical observations, \(s_i^2\) is the sample-variance statistic in the theorem; it need not estimate a common variance.

**Stratified Theorem 4** has exactly the same \(U\) and \(s_i^2\) formula as stratified Theorem 11, but requires \(D_{i1},\ldots,D_{iR}\) to be IID within each \(i\).

**Hoeffding complete-repetition B**

\[
U_{\mathrm{H,B}}=\bar Y+L\sqrt{\frac{2h_K}{R}},
\qquad \bar Y=\frac1R\sum_rY_r.
\]

**Complete-repetition Theorem 4**

\[
U_{\mathrm{MP4,B}}=\bar Y+
\sqrt{\frac{2s_Y^2b_K}{R}}
+\frac{14Lb_K}{3(R-1)},
\qquad
s_Y^2=\frac1{R-1}\sum_r(Y_r-\bar Y)^2.
\]

This requires IID complete repetitions. No fixed-horizon theorem above permits unadjusted repeated looks or optional stopping.

## 5. Comparison of all six methods

The variance-dependent and worst-case radii are:

| Method | Independent sampling unit / assumptions | Variance-dependent radius (nats) | Worst-case radius (nats) |
|---|---|---|---|
| Hoeffding pooled A | Each \(D_{ir}\); all \(130R\) independent | None | \(L\sqrt{2h_K/(130R)}\) |
| Pooled MP11 | Each \(D_{ir}\); all \(130R\) independent, not necessarily IID | \(\sqrt{2V_N(D)b_K/(130R)}+14Lb_K/[3(130R-1)]\) | \(L\sqrt{2b_K/(130R-1)}+14Lb_K/[3(130R-1)]\) |
| Stratified MP11 | For each \(i\), \(D_{ir}\) independent over \(r\); no cross-\(i\) independence | \(130^{-1}\sum_i\sqrt{2s_i^2c_K/R}+14Lc_K/[3(R-1)]\) | \(L\sqrt{2c_K/(R-1)}+14Lc_K/[3(R-1)]\) |
| Stratified MP4 | For each \(i\), \(D_{ir}\) IID over \(r\); no cross-\(i\) independence | Same as stratified MP11 | Same as stratified MP11 |
| Hoeffding complete-repetition B | \(Y_r\) independent; arbitrary dependence within repetition | None | \(L\sqrt{2h_K/R}\) |
| Complete-repetition MP4 | \(Y_r\) IID; arbitrary dependence within repetition | \(\sqrt{2s_Y^2b_K/R}+14Lb_K/[3(R-1)]\) | \(L\sqrt{2b_K/(R-1)}+14Lb_K/[3(R-1)]\) |

For \(m\) observations in \([-L,L]\), the sample-variance statistic is at most \(mL^2/(m-1)\). This gives the worst-case columns. Stratified MP11 and MP4 have the same formula and radius under the same data and alpha; they differ only in assumptions. Empirical-Bernstein methods may improve when observed variance is small, but they do not uniformly dominate Hoeffding. Pooled variance can be inflated by between-example heterogeneity; stratification avoids that contribution at the cost of the \(130\)-stratum alpha allocation and only \(R\) observations per stratum.

The formulas provide confidence bounds only conditional on the assumptions stated for each row and on the randomization model in Section 9. They do not prove that the local sampler or hardware satisfies those assumptions.

## 6. Shortlist and familywise error control

The 23 screened states comprise the initial state plus up to 22 non-initial candidate states. Let \(K\in\{1,3,5,22\}\) be the number of unique challenger vectors in the frozen shortlist; remove exact full-vector duplicates first. Set \(\alpha_c=0.05/K\). For stratified methods, further allocate \(\alpha_i=\alpha_c/130\).

| \(K\) | Comparison alpha | Defensible claim | Omission limitation |
|---:|---:|---|---|
| 1 | 0.050000 | The frozen challenger is supported as improving on its confirmed incumbent if its bound passes. | Better unshortlisted candidates may be missed; probability unknown. |
| 3 | 0.016667 | Up to three fixed, familywise-controlled improvement comparisons. | No numeric omission probability; no global-optimum claim. |
| 5 | 0.010000 | Up to five fixed, familywise-controlled comparisons. | Omission probability remains unknown. |
| 22 | 0.002273 | All 22 non-initial states are tested in the frozen order. | This still does not prove a global best; non-acceptance can reflect low power. |

Freeze shortlist, vectors/hashes, order, \(K\), and tie-breaking before confirmation. The screen-noise and candidate-effect distributions are unknown, so no numerical probability of omitting a better candidate can be stated.

For adaptive sequential comparisons, conditional on prior outcomes the candidate/current-incumbent pair must be fixed before fresh confirmation data are collected. If each comparison's guarantee is conditionally valid at \(0.05/K\), the union bound controls the probability of one or more false accepted improvements by \(0.05\), despite dependent comparison decisions. This does not claim global optimality.

## 7. Confirmation algorithm

1. **Screen:** use the existing delta-method rule for up to 23 states. Label all results exploratory; do not make confirmatory claims.
2. **Freeze:** select and deduplicate \(K\in\{1,3,5,22\}\) challengers by a predeclared rule. Freeze full vectors, hashes, order, C indices, shot count, alpha, and fixed \(R\).
3. **Initialize:** start from the predeclared initial incumbent. Screening counts do not count as confirmation.
4. **Compare:** in the frozen order, compare each challenger to the current incumbent. Measure both afresh using the same 130 examples and 512 shots per loss batch. Never reuse historical incumbent counts.
5. **Primary calculation:** use stratified Theorem 11, with \(R\ge2\) independent observations within each example and the \(\alpha_c/130\) allocation. Use it only under its assumptions and the stated randomness model.
6. **Fallback calculation:** if the defensible unit is a complete repetition, use Hoeffding B for independent repetitions with stable means; use Theorem 4 B only if complete repetitions are IID. Use a single method/alpha plan frozen before outcomes; do not choose a smaller bound after observing data without accounting for that choice.
7. **Accept:** replace the incumbent only if the predeclared one-sided upper bound satisfies \(U_j<-\delta\), with unchanged \(\delta=0.0005\).
8. **Otherwise:** retain the incumbent and report “not accepted” or “inconclusive.” Without a separately allocated lower-bound test, do not claim that the challenger is worse or equivalent.
9. **Finalize:** report the final state as the endpoint of confirmed improvements among the frozen shortlist, not the global optimum among all 23 screened states. Validation remains reporting-only.

Use one fixed confirmation budget per comparison and no unadjusted interim looks. Any optional stopping would require a separately specified time-uniform method and a new protocol review.

## 8. Resource and run/job cost tables

One confirmation repetition for one candidate/incumbent comparison uses 130 examples × two states = 260 circuit evaluations, each at 512 shots. Thus

\[
N_{\mathrm{circuits}}=260KR,\qquad
N_{\mathrm{shots}}=133{,}120KR.
\]

The screening cost is separate: up to \(23\times130=2{,}990\) circuit evaluations and 1,530,880 shots. These counts exclude optimizer minibatches, warm-start/refits, and validation reporting.

The table gives planning scenarios, not confirmed backend capacity. Pooled A assumes one isolated one-PUB run/job per loss batch. The batched designs assume 130 PUBs in each candidate/incumbent job for a single repetition. These are run/job counts under the proposed grouping, not proof of independence.

| \(K\) | \(R\) | Circuit evaluations | Shots | Pooled-A runs/jobs | Stratified/B runs/jobs |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 1,300 | 665,600 | 1,300 | 10 |
| 1 | 10 | 2,600 | 1,331,200 | 2,600 | 20 |
| 1 | 20 | 5,200 | 2,662,400 | 5,200 | 40 |
| 3 | 5 | 3,900 | 1,996,800 | 3,900 | 30 |
| 3 | 10 | 7,800 | 3,993,600 | 7,800 | 60 |
| 3 | 20 | 15,600 | 7,987,200 | 15,600 | 120 |
| 5 | 5 | 6,500 | 3,328,000 | 6,500 | 50 |
| 5 | 10 | 13,000 | 6,656,000 | 13,000 | 100 |
| 5 | 20 | 26,000 | 13,312,000 | 26,000 | 200 |
| 22 | 5 | 28,600 | 14,643,200 | 28,600 | 220 |
| 22 | 10 | 57,200 | 29,286,400 | 57,200 | 440 |
| 22 | 20 | 114,400 | 58,572,800 | 114,400 | 880 |

The radii below are worst-case, in nats, computed from the exact formulas and \(L=23.02585\). The two stratified columns are numerically identical, as explained in Section 5.

| \(K\) | \(R\) | Hoeffding A | Pooled MP11 | Stratified MP11 | Stratified MP4 | Hoeffding B | Complete MP4 B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 2.21 | 3.07 | 277.48 | 277.48 | 25.21 | 130.37 |
| 1 | 10 | 1.56 | 2.04 | 133.91 | 133.91 | 17.82 | 64.89 |
| 1 | 20 | 1.11 | 1.38 | 70.24 | 70.24 | 12.60 | 35.21 |
| 3 | 5 | 2.58 | 3.59 | 309.96 | 309.96 | 29.47 | 164.23 |
| 3 | 10 | 1.83 | 2.37 | 149.00 | 149.00 | 20.84 | 80.91 |
| 3 | 20 | 1.29 | 1.60 | 77.82 | 77.82 | 14.73 | 43.42 |
| 5 | 5 | 2.74 | 3.82 | 325.00 | 325.00 | 31.25 | 179.81 |
| 5 | 10 | 1.94 | 2.52 | 155.98 | 155.98 | 22.10 | 88.24 |
| 5 | 20 | 1.37 | 1.69 | 81.31 | 81.31 | 15.63 | 47.16 |
| 22 | 5 | 3.15 | 4.45 | 368.46 | 368.46 | 35.93 | 224.53 |
| 22 | 10 | 2.23 | 2.91 | 176.11 | 176.11 | 25.41 | 109.21 |
| 22 | 20 | 1.58 | 1.94 | 91.37 | 91.37 | 17.96 | 57.80 |

At \(R=5,10,20\), the worst-case radii are far above \(\delta=0.0005\). A much larger true improvement could still pass; the table does not establish actual power or observed variance. For Hoeffding, resolving an effect only \(\delta\) requires approximately

\[
R_{\mathrm A}\ge\frac{2L^2\ln(20K)}{130\delta^2},\qquad
R_{\mathrm B}\ge\frac{2L^2\ln(20K)}{\delta^2}.
\]

For \(K=1,3,5,22\), the A values are approximately \(9.8\times10^7,\ 1.34\times10^8,\ 1.50\times10^8,\ 1.99\times10^8\) repetitions per example; B requires 130 times more. The empirical-Bernstein additive terms alone require approximately 0.79–1.46 million repetitions for complete-repetition MP4 B, 6,100–11,200 for pooled MP11 A after accounting for \(N=130R\), and 1.84–2.50 million for stratified MP11/MP4. These are additive-term thresholds only; the variance terms still apply. They are theoretical calculations, not recommended budgets.

## 9. Independence engineering and PRNG assumptions

All confidence guarantees are conditional on the stated sampling-randomness and stationarity assumptions.

- **Stratified MP11 primary design:** one candidate and one incumbent job/run per repetition, each containing the 130 example PUBs. Dependence among examples within a repetition is allowed. For every fixed example, its \(D_{ir}\) values must be independent across repetitions; their distributions may differ if their means remain \(\mu_i\).
- **Complete-repetition fallback:** a fresh whole repetition \(Y_r\) must be independent of other repetitions; MP4 additionally needs identical distributions. Within-repetition PUB dependence is allowed.
- **Pooled A option:** each of the \(130R\) paired differences must be independent of every other one. An isolated one-PUB run for each candidate and incumbent loss batch is the proposed architecture, but job separation alone does not prove independence.
- A fixed PRNG seed makes the output deterministic conditional on that seed. Distinct seeds or child streams are an engineering technique, not proof of mathematical independence. The theorem's probability is with respect to an idealized source of fresh independent sampler randomness; treating a PRNG as approximating that source is an explicit modeling assumption. Reproducibility logs should record seed generation, stream-to-example mapping, parameters, and order, but logging does not establish independence.
- No sampler-specific cross-PUB independence has been verified here. No hardware independence or stationarity has been established; separate jobs and randomized order do not prove either and cannot rule out drift.

If these assumptions cannot be defended, the resulting bounds must not be described as calibrated confidence guarantees for the actual execution.

## 10. Unresolved scientific decisions

1. **Practical effect size:** the frozen V2 protocol uses \(\delta=0.0005\), but supplies no scientific or decision-theoretic rationale for a 0.0005-nat change. The value is not changed here. The research owner must independently justify it or retain it while accepting that outcomes may remain inconclusive. Do not select \(\delta\) from validation metrics, observed results, or affordability.
2. **Resource cap:** no maximum confirmation budget, job quota, wall-clock allowance, or acceptable inconclusive rate has been approved. The worst-case calculations at \(R=5,10,20\) do not resolve \(\delta\); an affordable budget must not be used to weaken the statistical standard.
3. **Randomness model:** local simulator streams and repetition-level independence have not been verified. A PRNG-seed plan is not a mathematical independence proof.
4. **Hardware behavior:** stationarity, drift, and job/PUB dependence on physical hardware remain unverified. No hardware confidence claim is authorized.
5. **Shortlist coverage:** the probability that screening omits a better checkpoint is unknown. Confirmation only supports statements about the frozen shortlist and its sequence of tested improvements.

## 11. Future V3 implementation requirements

Any future implementation needs a distinct identity such as `QNIM_QUANTUM_PCA_SCALING_V3`. V1 and V2 must remain unchanged, including V2 artifacts and hashes. Carry forward without change:

- A=2,275 optimizer examples; B=195 readout examples; C=130 checkpoint-reference examples.
- 650 validation examples, reporting-only.
- q16/PCA16, circuit/readout architecture, optimizer equations, initialization and optimizer configuration.
- 512 shots per measurement batch and the existing class order.
- V2 dataset file hashes and manifest hashes from [protocol.json](./protocol.json).
- The clipped loss floor \(10^{-10}\), unless an independently justified objective change is separately versioned.

Before implementation:

1. Resolve the scientific rationale for \(\delta\), the hard resource cap, the shortlist size, and fixed \(R\); explicitly allow an inconclusive outcome.
2. Freeze the primary stratified Theorem 11 method and its alpha allocation. Define the fallback and the exact conditions under which it is used. Do not silently switch methods after observing outcomes.
3. Specify the randomization model and independent repetition execution. Do not claim that unique PRNG seeds prove independence. For hardware, specify an execution-order/stationarity rationale without representing it as verified evidence.
4. Preserve full vector hashes, use fresh candidate and incumbent measurements on every comparison, verify exact shots and PUB-to-example association, and separate screening from confirmation artifacts.
5. Implement the fixed-horizon equations verbatim with \(R\ge2\), correct empirical-variance definitions, alpha allocation, and explicit `accepted`, `not accepted`, or `inconclusive` outcomes.
6. Add non-quantum synthetic tests for coverage under the stated models, non-IID strata, permitted cross-example dependence, adaptive incumbent chains with fresh data, alpha allocation, clipping, and budget exhaustion. Such tests cannot establish live sampler compatibility.
7. Perform a new static review and obtain separate authorization before any sampler smoke test or training. Do not access the test split during implementation or confirmation.

## 12. Final GO/NO-GO

- **Mathematical design:** complete as a conditional protocol design. The six methods, bounds, multiplicity, resources, and assumptions are specified; this does not establish that the current simulator or hardware meets them.
- **Primary proposed method:** stratified Maurer–Pontil Theorem 11, conditional on independent repetitions within each fixed example, stationary per-example means, and the stated idealized randomization model.
- **Fallback:** complete-repetition Hoeffding for independent complete repetitions and stable means; use complete-repetition Maurer–Pontil Theorem 4 only if those repetitions can credibly be treated as IID.
- **Optional method:** pooled Theorem 11 only if independence across every individual \(D_{ir}\) is justified.
- **Unverified assumptions:** local sampler independence across repetition blocks, any PRNG-to-independent-randomness approximation, and physical-hardware independence/stationarity or absence of drift.
- **Unresolved decisions:** \(\delta=0.0005\) lacks a scientific rationale, and no maximum confirmation budget has been approved. Neither may be chosen from observed validation results or merely to fit available compute.
- **Authorization:** **NO-GO for V3 implementation and execution** until the research owner resolves the effect-size rationale, confirmation budget, and randomness model and freezes them in a new protocol. V1/V2, their hashes, data, and protocols remain untouched. No experiment is authorized by this report.
