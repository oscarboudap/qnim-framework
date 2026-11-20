# QNSPSA-EML-Feynman implementation note

This note describes the implemented calculations in `src/infrastructure/qnspsa_eml_feynman.py` and the Qiskit loss function used by `QNIM_QUANTUM_PCA_SCALING_V1`. The experiment's optimizer mathematics is unchanged by the corrective engineering work; execution counters and trace reporting are instrumentation.

## Implemented calculations

### SPSA gradient estimates

For a Rademacher vector `Delta` and perturbation `c_t`, the optimizer evaluates the same minibatch at `theta + c_t Delta` and `theta - c_t Delta`, then computes

`g_hat = (f(theta + c_t Delta) - f(theta - c_t Delta)) / (2 c_t Delta)`.

It draws a second independent Rademacher vector and repeats the two evaluations to obtain `g_hat2`. Thus the iteration uses four loss evaluations for these two SPSA estimates. The fixed batch is refreshed once per outer iteration and is shared by the optimizer's gradient, quadrature, blocking, and current-loss evaluations.

### `F_hat` and preconditioner

The matrix named `F_hat` is constructed as

`F_hat = 0.5 * (outer(g_hat, g_hat) + outer(g_hat2, g_hat2))`.

Its vectors are stochastic gradients of the classical loss. This construction does not use state derivatives, quantum-state overlaps, measurement derivatives, or a quantum Fisher-information estimator. Consequently `F_hat` is a loss-gradient-based, rank-at-most-two proxy; its name does not make it a physical QFI or Fubini–Study quantum geometric tensor.

The implemented preconditioner is a dense matrix `H_inv`, initialized as `0.1 * learning_rate * I`. Each iteration applies the code's Sherman–Morrison-style rank-one update using the mean of `F_hat`'s columns, then adds `hessian_regularization * I`. The step is `theta_new = theta - a_t * (H_inv @ g_total)`. This is the matrix actually used; the implementation does not calculate or invert a physical QGT.

### EML term

The code computes the smallest eigenvalue of `F_hat + 1e-8 I`, floors it at `1e-10`, sets `eml_scale = min(lambda_eml / lambda_min, eml_max_scale)`, and adds `eml_grad = -eml_scale * g_hat` to the update direction. This is an implemented loss-gradient heuristic. The experiment has no calculation demonstrating that it improves gradient concentration or mitigates barren plateaus, so no such scientific claim is made here.

### Gauss–Legendre calculation and parameter selection

The implementation evaluates the baseline loss and eight Gauss–Legendre-shifted loss values per treated coordinate. Nodes on `[-1, 1]` are scaled by `pi/2`; the result is `(integral - 2 * baseline) * pi/2`. This calculation replaces the SPSA entries for the first `n_feynman_params=4` gradient coordinates; the remaining coordinates retain the first SPSA estimate. In this experiment the parameter vector is ordered as ansatz weights followed by classical MLP weights, so the four treated coordinates are the first four **EfficientSU2 ansatz parameters**, not the ZZFeatureMap data parameters or classical readout parameters.

The historical implementation's stated motivation concerns a Chebyshev feature map and a smooth single-frequency cost form. This experiment instead encodes inputs with `ZZFeatureMap` and applies the quadrature to EfficientSU2 ansatz angles. A finite parameterized circuit's expectation can be a trigonometric polynomial, but its frequency content and the cross-entropy after a nonlinear classical readout do not establish the specific integral identity or the claimed eight-point accuracy used here. No architecture-specific derivation or convergence validation establishes that this Feynman component is mathematically appropriate for this ZZFeatureMap/EfficientSU2/readout combination. It remains an unvalidated optimizer heuristic for this experiment.

### Noise handling

Zero-noise extrapolation is not performed: the runner uses a local Aer `matrix_product_state` sampler at a single noise level, does not fold circuits or scale noise, and does not fit an observable-versus-noise curve. `noise_aware_training=False`; EML is not ZNE.

## Interpretation limits

Implemented facts above describe code behavior only. “Natural gradient,” “quantum Fisher information,” “Feynman exact gradient,” “zero-noise extrapolation,” and “barren-plateau mitigation” must not be used as experimentally established properties of this run. The optimizer's historical class name is retained for compatibility, not as evidence for those theoretical interpretations.
