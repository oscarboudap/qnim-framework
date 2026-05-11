import type { GWEvent, TheoryResult, QFIResult, TrainingEpoch, ConfusionEntry, HardwareResult } from '../types/qnim';

// ─── GW Events Catalog ───────────────────────────────────────────────────────
export const GW_EVENTS: GWEvent[] = [
  { id: 'GW150914', name: 'GW150914', type: 'BBH', m1: 35.6, m2: 30.6, dL: 440, snr: 24.0, grConsistency: 'High', chirpMass: 28.3, spinEff: -0.04, finalMass: 63.5, finalSpin: 0.672 },
  { id: 'GW170817', name: 'GW170817', type: 'BNS', m1: 1.46, m2: 1.27, dL: 40, snr: 32.4, grConsistency: 'Exceptional', chirpMass: 1.19, spinEff: 0.0 },
  { id: 'GW190521', name: 'GW190521', type: 'IMBH', m1: 85.1, m2: 66.0, dL: 5300, snr: 14.7, grConsistency: 'Ringdown limited', chirpMass: 64.5, spinEff: 0.08 },
  { id: 'GW190814', name: 'GW190814', type: 'Mixed', m1: 23.2, m2: 2.59, dL: 240, snr: 25.0, grConsistency: 'Mass-gap probe', chirpMass: 6.09, spinEff: -0.002 },
  { id: 'GW190412', name: 'GW190412', type: 'BBH', m1: 30.1, m2: 8.3, dL: 730, snr: 19.0, grConsistency: 'Higher Harmonics', chirpMass: 13.3, spinEff: 0.28 },
  { id: 'GW200105', name: 'GW200105', type: 'NSBH', m1: 8.9, m2: 1.9, dL: 280, snr: 13.9, grConsistency: 'Spin-orbit coupling', chirpMass: 3.22, spinEff: -0.01 },
  { id: 'GW170104', name: 'GW170104', type: 'BBH', m1: 31.2, m2: 19.4, dL: 880, snr: 13.0, grConsistency: 'Graviton mass bound', chirpMass: 21.4, spinEff: -0.12 },
  { id: 'GW151226', name: 'GW151226', type: 'BBH', m1: 13.7, m2: 7.7, dL: 440, snr: 13.0, grConsistency: 'Spin measurement', chirpMass: 8.9, spinEff: 0.21 },
  { id: 'GW200115', name: 'GW200115', type: 'NSBH', m1: 5.9, m2: 1.5, dL: 300, snr: 11.6, grConsistency: 'Parity tests', chirpMass: 2.59, spinEff: -0.01 },
  { id: 'GW190425', name: 'GW190425', type: 'BNS', m1: 2.0, m2: 1.4, dL: 160, snr: 12.9, grConsistency: 'High-mass BNS', chirpMass: 1.44, spinEff: 0.0 },
];

// ─── Bayes Factors (GW150914) ─────────────────────────────────────────────────
// backend keys map to TheoryFamily enum in src/domain/astrophysics/value_objects.py
export const THEORY_RESULTS: TheoryResult[] = [
  { theory: 'Brans-Dicke', layer: 5, log10B: -0.32, uncertainty: 0.18, interpretation: 'Anecdotal for GR', color: '#00d4ff' },
  { theory: 'Massive Graviton', layer: 5, log10B: -0.28, uncertainty: 0.15, interpretation: 'Anecdotal for GR', color: '#00d4ff' },
  { theory: 'Extra Dimensions (ADD/KK)', layer: 5, log10B: -0.21, uncertainty: 0.19, interpretation: 'Anecdotal for GR', color: '#00d4ff' },
  { theory: 'LQG Echoes', layer: 6, log10B: 0.41, uncertainty: 0.22, interpretation: 'Anecdotal for LQG (1.4σ)', color: '#7c3aed' },
  { theory: 'String Fuzzballs', layer: 6, log10B: -0.18, uncertainty: 0.19, interpretation: 'Anecdotal for GR', color: '#7c3aed' },
  { theory: 'Spacetime Foam', layer: 7, log10B: -0.15, uncertainty: 0.12, interpretation: 'Inconclusive', color: '#ff4466' },
  { theory: 'f(R) Gravity', layer: 5, log10B: -0.35, uncertainty: 0.20, interpretation: 'Anecdotal for GR', color: '#00d4ff' },
  { theory: 'Axion Superradiance', layer: 5, log10B: -0.09, uncertainty: 0.14, interpretation: 'Inconclusive', color: '#ff4466' },
  { theory: 'Randall-Sundrum', layer: 5, log10B: -0.11, uncertainty: 0.16, interpretation: 'Inconclusive', color: '#ff4466' },
  { theory: 'General Relativity', layer: 0, log10B: 0, uncertainty: 0, interpretation: 'Reference model', color: '#00ff88' },
];

// ─── QFI vs CFI ───────────────────────────────────────────────────────────────
// QFI values clipped to [10, 40]; advantage = f_quantum / f_classical >= 1.8 (backend minimum)
export const QFI_RESULTS: QFIResult[] = [
  { parameter: 'δQ (quadrupole)', fq: 24.3, fc: 11.8, advantage: 2.06, sigma: '>3σ' },
  { parameter: 'mg (graviton mass)', fq: 18.7, fc: 9.2, advantage: 2.03, sigma: '>2σ' },
  { parameter: '|R| (LQG echoes)', fq: 31.5, fc: 14.1, advantage: 2.23, sigma: '>4σ' },
  { parameter: 'Δs (scalar-tensor)', fq: 15.2, fc: 8.3, advantage: 1.83, sigma: '>2σ' },
  { parameter: 'α (quantum foam)', fq: 22.8, fc: 10.3, advantage: 2.21, sigma: '>4σ' },
];

// ─── Training history ─────────────────────────────────────────────────────────
export const TRAINING_HISTORY: TrainingEpoch[] = [
  { epoch: 1, loss_train: 0.891, loss_val: 0.902, acc_train: 48.2, acc_val: 46.8 },
  { epoch: 5, loss_train: 0.612, loss_val: 0.638, acc_train: 67.1, acc_val: 63.9 },
  { epoch: 10, loss_train: 0.421, loss_val: 0.455, acc_train: 79.3, acc_val: 75.4 },
  { epoch: 15, loss_train: 0.342, loss_val: 0.380, acc_train: 83.8, acc_val: 80.1 },
  { epoch: 20, loss_train: 0.285, loss_val: 0.311, acc_train: 87.6, acc_val: 84.2 },
  { epoch: 25, loss_train: 0.231, loss_val: 0.258, acc_train: 90.2, acc_val: 87.8 },
  { epoch: 30, loss_train: 0.198, loss_val: 0.227, acc_train: 91.8, acc_val: 89.5 },
  { epoch: 34, loss_train: 0.183, loss_val: 0.219, acc_train: 92.4, acc_val: 91.0 },
];

// ─── Confusion Matrix ─────────────────────────────────────────────────────────
// Labels map to TheoryFamily: GR=KERR_VACUUM, BD=BRANS_DICKE, ADD=KALUZA_KLEIN,
// dRGT=MASSIVE_GRAVITY, LQG=PLANCK_STAR, FZ=STRING_FUZZBALL,
// MEM=BMS_SOFT_HAIR, RD=RANDALL_SUNDRUM, HK=HORNDESKI, QF=QUANTUM_FOAM
export const CONFUSION_LABELS = ['GR', 'BD', 'ADD', 'dRGT', 'LQG', 'FZ', 'MEM', 'RD', 'HK', 'QF'];
export const CONFUSION_DATA: ConfusionEntry[] = [
  { theory: 'GR',   values: [0.96, 0.01, 0.00, 0.01, 0.00, 0.00, 0.01, 0.01, 0.00, 0.00] },
  { theory: 'BD',   values: [0.02, 0.92, 0.01, 0.02, 0.00, 0.01, 0.00, 0.01, 0.01, 0.00] },
  { theory: 'ADD',  values: [0.00, 0.02, 0.94, 0.01, 0.00, 0.01, 0.01, 0.00, 0.01, 0.00] },
  { theory: 'dRGT', values: [0.01, 0.02, 0.01, 0.93, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01] },
  { theory: 'LQG',  values: [0.00, 0.00, 0.01, 0.00, 0.95, 0.03, 0.01, 0.00, 0.00, 0.00] },
  { theory: 'FZ',   values: [0.00, 0.01, 0.01, 0.00, 0.04, 0.91, 0.00, 0.02, 0.01, 0.00] },
  { theory: 'MEM',  values: [0.01, 0.00, 0.01, 0.00, 0.00, 0.01, 0.97, 0.00, 0.00, 0.00] },
  { theory: 'RD',   values: [0.01, 0.01, 0.00, 0.01, 0.01, 0.02, 0.00, 0.89, 0.02, 0.03] },
  { theory: 'HK',   values: [0.00, 0.01, 0.01, 0.01, 0.00, 0.01, 0.00, 0.02, 0.85, 0.09] },
  { theory: 'QF',   values: [0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.00, 0.02, 0.09, 0.86] },
];

// ─── Hardware comparison ──────────────────────────────────────────────────────
export const HARDWARE_RESULTS: HardwareResult[] = [
  { backend: 'Aer Simulator', accuracy: 92.4, error: 1.8, timePerEvent: 2, shots: 2048 },
  { backend: 'IBM Kingston (No ZNE)', accuracy: 74.3, error: 3.2, timePerEvent: 45, shots: 2048 },
  { backend: 'IBM Kingston (ZNE)', accuracy: 86.1, error: 2.4, timePerEvent: 180, shots: 6144 },
];

// ─── SNR accuracy comparison ──────────────────────────────────────────────────
export const SNR_ACCURACY = [
  { snr: 8, vqc12: 68, resnet: 64, vqc27: 71 },
  { snr: 12, vqc12: 79, resnet: 71, vqc27: 82 },
  { snr: 20, vqc12: 88, resnet: 79, vqc27: 91 },
  { snr: 30, vqc12: 91, resnet: 84, vqc27: 94 },
  { snr: 50, vqc12: 95, resnet: 89, vqc27: 97 },
];

// ─── Detection efficiency by model and SNR ────────────────────────────────────
export const DETECTION_EFFICIENCY = [
  { model: 'Brans-Dicke', snr8: 12, snr12: 28, snr20: 55, snr30: 78, snr50: 92 },
  { model: 'LQG Echoes', snr8: 18, snr12: 35, snr20: 68, snr30: 85, snr50: 97 },
  { model: 'Massive Graviton', snr8: 9, snr12: 22, snr20: 48, snr30: 71, snr50: 89 },
  { model: 'Extra Dim. ADD', snr8: 14, snr12: 31, snr20: 60, snr30: 81, snr50: 95 },
  { model: 'Quantum Foam', snr8: 5, snr12: 12, snr20: 28, snr30: 49, snr50: 74 },
];

// ─── GW150914 parameters ─────────────────────────────────────────────────────
export const GW150914_PARAMS = [
  { param: 'm₁ [M☉]', qnim: 35.2, qnimErr: 1.8, gwtc: 35.6, gwtcLow: 30.8, gwtcHigh: 40.4 },
  { param: 'm₂ [M☉]', qnim: 30.1, qnimErr: 1.5, gwtc: 30.6, gwtcLow: 26.2, gwtcHigh: 33.6 },
  { param: 'χ_eff', qnim: -0.04, qnimErr: 0.08, gwtc: -0.07, gwtcLow: -0.21, gwtcHigh: 0.10 },
  { param: 'd_L [Mpc]', qnim: 418, qnimErr: 52, gwtc: 410, gwtcLow: 230, gwtcHigh: 570 },
  { param: 'M_f [M☉]', qnim: 63.5, qnimErr: 1.2, gwtc: 63.1, gwtcLow: 60.1, gwtcHigh: 66.5 },
  { param: 'χ_f', qnim: 0.672, qnimErr: 0.035, gwtc: 0.67, gwtcLow: 0.60, gwtcHigh: 0.72 },
];

// ─── Barren plateau gradient variance ────────────────────────────────────────
export const BARREN_PLATEAUS = [
  { qubits: 8, variance: 0.062, trainable: true },
  { qubits: 12, variance: 0.016, trainable: true },
  { qubits: 16, variance: 0.0039, trainable: 'marginal' },
  { qubits: 20, variance: 0.00095, trainable: 'marginal' },
  { qubits: 27, variance: 0.00017, trainable: true },
];
