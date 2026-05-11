// Core data types for QNIM frontend

export interface GWEvent {
  id: string;
  name: string;
  type: 'BBH' | 'BNS' | 'NSBH' | 'IMBH' | 'Mixed';
  m1: number;
  m2: number;
  dL: number;
  snr: number;
  grConsistency: string;
  chirpMass?: number;
  spinEff?: number;
  finalMass?: number;
  finalSpin?: number;
}

export interface TheoryResult {
  theory: string;
  layer: number;
  log10B: number;
  uncertainty: number;
  interpretation: string;
  color: string;
}

export interface QFIResult {
  parameter: string;
  fq: number;
  fc: number;
  advantage: number;
  sigma: string;
}

export interface TrainingEpoch {
  epoch: number;
  loss_train: number;
  loss_val: number;
  acc_train: number;
  acc_val: number;
}

export interface ConfusionEntry {
  theory: string;
  values: number[];
}

export interface HardwareResult {
  backend: string;
  accuracy: number;
  error: number;
  timePerEvent: number;
  shots: number;
}

export interface WaveformPoint {
  time: number;
  strain: number;
  strainGR?: number;
  strainBSM?: number;
}

export interface SimulationConfig {
  m1: number;
  m2: number;
  spin1: number;
  spin2: number;
  distance: number;
  theory: string;
  snr: number;
  injectionLayer: number;
}

export interface MetricCard {
  label: string;
  value: string;
  unit?: string;
  delta?: string;
  deltaPositive?: boolean;
  color: string;
}
