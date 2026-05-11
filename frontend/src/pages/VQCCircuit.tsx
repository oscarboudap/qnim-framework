import { useState } from 'react';
import { motion } from 'framer-motion';
import SectionHeader from '../components/SectionHeader';

interface Gate {
  type: string;
  qubit: number;
  qubit2?: number;
  color: string;
  label: string;
}

const CIRCUIT_LAYERS = [
  // Feature map layer
  [
    { type: 'H', qubit: 0, color: '#00d4ff', label: 'H' },
    { type: 'H', qubit: 1, color: '#00d4ff', label: 'H' },
    { type: 'H', qubit: 2, color: '#00d4ff', label: 'H' },
    { type: 'H', qubit: 3, color: '#00d4ff', label: 'H' },
  ],
  // Rx rotation (feature encoding)
  [
    { type: 'Rx', qubit: 0, color: '#7c3aed', label: 'Rx(x₀)' },
    { type: 'Rx', qubit: 1, color: '#7c3aed', label: 'Rx(x₁)' },
    { type: 'Rx', qubit: 2, color: '#7c3aed', label: 'Rx(x₂)' },
    { type: 'Rx', qubit: 3, color: '#7c3aed', label: 'Rx(x₃)' },
  ],
  // CNOT entanglement
  [
    { type: 'CNOT', qubit: 0, qubit2: 1, color: '#00ff88', label: '●' },
    { type: 'CNOT', qubit: 2, qubit2: 3, color: '#00ff88', label: '●' },
  ],
  // Ry rotation (variational)
  [
    { type: 'Ry', qubit: 0, color: '#ffd700', label: 'Ry(θ₀)' },
    { type: 'Ry', qubit: 1, color: '#ffd700', label: 'Ry(θ₁)' },
    { type: 'Ry', qubit: 2, color: '#ffd700', label: 'Ry(θ₂)' },
    { type: 'Ry', qubit: 3, color: '#ffd700', label: 'Ry(θ₃)' },
  ],
  // Second CNOT layer
  [
    { type: 'CNOT', qubit: 1, qubit2: 2, color: '#00ff88', label: '●' },
    { type: 'CNOT', qubit: 0, qubit2: 3, color: '#00ff88', label: '●' },
  ],
  // Rz rotation (variational)
  [
    { type: 'Rz', qubit: 0, color: '#ff4466', label: 'Rz(θ₄)' },
    { type: 'Rz', qubit: 1, color: '#ff4466', label: 'Rz(θ₅)' },
    { type: 'Rz', qubit: 2, color: '#ff4466', label: 'Rz(θ₆)' },
    { type: 'Rz', qubit: 3, color: '#ff4466', label: 'Rz(θ₇)' },
  ],
  // Measure
  [
    { type: 'M', qubit: 0, color: '#9ca3af', label: '⊗' },
    { type: 'M', qubit: 1, color: '#9ca3af', label: '⊗' },
    { type: 'M', qubit: 2, color: '#9ca3af', label: '⊗' },
    { type: 'M', qubit: 3, color: '#9ca3af', label: '⊗' },
  ],
];

const NQUBITS = 4; // Showing first 4 qubits for clarity
const QUBIT_SPACING = 60;
const LAYER_SPACING = 90;
const MARGIN_LEFT = 70;
const MARGIN_TOP = 40;

export default function VQCCircuit() {
  const [hoveredGate, setHoveredGate] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  const circuitWidth = MARGIN_LEFT + CIRCUIT_LAYERS.length * LAYER_SPACING + 60;
  const circuitHeight = MARGIN_TOP + NQUBITS * QUBIT_SPACING + 30;

  return (
    <div className="space-y-6">
      <SectionHeader
        title="VQC Circuit Architecture"
        subtitle="EfficientSU2 ansatz · 12 qubits · reps=2 · ChebyshevFeatureMap encoding"
        badge="CIRCUIT"
      />

      {/* Circuit info cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Total Qubits', value: '12', color: '#00d4ff' },
          { label: 'Circuit Depth', value: '~24', color: '#7c3aed' },
          { label: 'CNOT gates', value: '22', color: '#00ff88' },
          { label: 'Parameters', value: '288', color: '#ffd700' },
        ].map((item, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}
            className="card p-4 text-center">
            <div className="text-xl font-bold mb-0.5" style={{ color: item.color }}>{item.value}</div>
            <div className="text-xs text-slate-500">{item.label}</div>
          </motion.div>
        ))}
      </div>

      {/* SVG Circuit diagram */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="card p-5"
      >
        <SectionHeader title="Simplified Circuit (4-qubit view)" subtitle="Showing first 4 qubits for clarity · full 12-qubit EfficientSU2 shown below" />
        <div className="overflow-x-auto">
          <svg width={circuitWidth} height={circuitHeight} className="text-xs font-mono">
            {/* Qubit wires */}
            {Array.from({ length: NQUBITS }, (_, q) => (
              <g key={q}>
                <line
                  x1={MARGIN_LEFT - 30}
                  y1={MARGIN_TOP + q * QUBIT_SPACING}
                  x2={circuitWidth - 20}
                  y2={MARGIN_TOP + q * QUBIT_SPACING}
                  stroke="rgba(255,255,255,0.15)"
                  strokeWidth={1}
                />
                <text x={MARGIN_LEFT - 35} y={MARGIN_TOP + q * QUBIT_SPACING + 4} fill="#6b7280" textAnchor="end" fontSize={11}>
                  q{q}
                </text>
                <text x={10} y={MARGIN_TOP + q * QUBIT_SPACING + 4} fill="#9ca3af" textAnchor="start" fontSize={10}>
                  |0⟩
                </text>
              </g>
            ))}

            {/* Gates */}
            {CIRCUIT_LAYERS.map((layer, li) => {
              const x = MARGIN_LEFT + li * LAYER_SPACING;
              return (
                <g key={li}>
                  {layer.map((gate, gi) => {
                    if (gate.qubit >= NQUBITS) return null;
                    const y = MARGIN_TOP + gate.qubit * QUBIT_SPACING;
                    const gateKey = `${li}-${gi}`;
                    const isHovered = hoveredGate === gateKey;

                    if (gate.type === 'CNOT' && gate.qubit2 !== undefined && gate.qubit2 < NQUBITS) {
                      const y2 = MARGIN_TOP + gate.qubit2 * QUBIT_SPACING;
                      return (
                        <g key={gi} onMouseEnter={() => setHoveredGate(gateKey)} onMouseLeave={() => setHoveredGate(null)} cursor="pointer">
                          <line x1={x} y1={y} x2={x} y2={y2} stroke={gate.color} strokeWidth={isHovered ? 2.5 : 1.5} />
                          <circle cx={x} cy={y} r={6} fill={gate.color} opacity={isHovered ? 1 : 0.8} />
                          <circle cx={x} cy={y2} r={12} fill="none" stroke={gate.color} strokeWidth={isHovered ? 2 : 1.5} opacity={isHovered ? 1 : 0.8} />
                          <line x1={x - 8} y1={y2} x2={x + 8} y2={y2} stroke={gate.color} strokeWidth={1.5} opacity={0.8} />
                          <line x1={x} y1={y2 - 8} x2={x} y2={y2 + 8} stroke={gate.color} strokeWidth={1.5} opacity={0.8} />
                        </g>
                      );
                    }

                    return (
                      <g key={gi} onMouseEnter={() => setHoveredGate(gateKey)} onMouseLeave={() => setHoveredGate(null)} cursor="pointer">
                        <rect
                          x={x - 22} y={y - 16} width={44} height={32}
                          rx={4}
                          fill={`${gate.color}20`}
                          stroke={gate.color}
                          strokeWidth={isHovered ? 2 : 1}
                          opacity={isHovered ? 1 : 0.8}
                        />
                        <text x={x} y={y + 4} fill={gate.color} textAnchor="middle" fontSize={gate.type === 'Rx' || gate.type === 'Ry' || gate.type === 'Rz' ? 9 : 11} fontWeight="bold">
                          {gate.label}
                        </text>
                        {isHovered && (
                          <rect x={x - 30} y={y - 35} width={60} height={16} rx={3} fill="#0a0f1e" stroke="rgba(0,212,255,0.4)" />
                        )}
                        {isHovered && (
                          <text x={x} y={y - 24} fill="#00d4ff" textAnchor="middle" fontSize={9}>{gate.type}</text>
                        )}
                      </g>
                    );
                  })}
                </g>
              );
            })}
          </svg>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-4 mt-4 text-xs">
          {[
            { label: 'Hadamard (H) — Superposition', color: '#00d4ff' },
            { label: 'Rotation (Rx/Ry/Rz) — Feature/Variational', color: '#7c3aed' },
            { label: 'CNOT — Entanglement', color: '#00ff88' },
            { label: 'Measure (⊗)', color: '#9ca3af' },
          ].map(item => (
            <div key={item.label} className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded" style={{ background: `${item.color}40`, border: `1px solid ${item.color}` }} />
              <span className="text-slate-400">{item.label}</span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Architecture description */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card p-5"
        >
          <SectionHeader title="Feature Map: ChebyshevFeatureMap" subtitle="Maps 12 PCA components to SU(2¹²) Hilbert space" />
          <div className="space-y-3 text-xs text-slate-400">
            <div className="p-3 rounded-lg font-mono" style={{ background: '#020510', border: '1px solid rgba(0,212,255,0.15)' }}>
              <p className="text-cyan-400">U(x_i) = exp(-i·x_i·Ĥ)</p>
              <p className="text-slate-600 mt-1"># Ĥ = Hermitian spin operator</p>
              <p className="text-slate-600"># x_i = i-th PCA component of h(t)</p>
              <p className="text-purple-400 mt-2">|ψ(x)⟩ ∈ ℂ^4096  [12 qubits]</p>
            </div>
            <div className="leading-relaxed">
              By encoding GW features as unitary evolution parameters, QNIM maps the signal into a 4096-dimensional Hilbert space where non-linear Planckian anomalies become <span className="text-cyan-400">linearly separable</span> — impossible in classical Euclidean space.
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card p-5"
        >
          <SectionHeader title="Variational Ansatz: EfficientSU2" subtitle="reps=2 · linear entanglement · 288 trainable parameters" />
          <div className="space-y-3 text-xs text-slate-400">
            <div className="p-3 rounded-lg font-mono" style={{ background: '#020510', border: '1px solid rgba(124,58,237,0.15)' }}>
              <p className="text-purple-400">V(θ) = ∏ [Ry(θ_k)·Rz(φ_k) ⊗ CNOT]</p>
              <p className="text-slate-600 mt-1"># 12 qubits × 2 reps × 12 params</p>
              <p className="text-fuchsia-400 mt-1">Expressibility E = 0.043 nats</p>
              <p className="text-green-400">Von Neumann S̄ = 3.82 bits</p>
            </div>
            <div className="leading-relaxed">
              The EfficientSU2 ansatz achieves <span className="text-purple-400">96% coverage</span> of the SU(2¹²) state space with minimal circuit depth, balancing expressibility against NISQ-era decoherence constraints.
            </div>
          </div>
        </motion.div>
      </div>

      {/* QNSPSA pseudocode */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="card p-5"
      >
        <SectionHeader title="QNSPSA-EML-Feynman Optimizer" subtitle="~124 evaluations/iteration · QGT + EML + Feynman path regularization" />
        <div className="font-mono text-xs p-4 rounded-lg leading-relaxed" style={{ background: '#020510', border: '1px solid rgba(0,255,136,0.15)' }}>
          <p className="text-green-400 font-bold">Algorithm: QNSPSA-EML-Feynman</p>
          <p className="text-slate-600">Ĥ_inv ← a₀·I  (Quantum Geometric Tensor init)</p>
          <p className="text-cyan-400 mt-2">for t = 1 to T do:</p>
          <p className="text-slate-400 ml-4">Δ ← Rademacher(p)  <span className="text-slate-600"># SPSA gradient (2 evals)</span></p>
          <p className="text-slate-400 ml-4">ĝ ← [f(θ+cΔ) - f(θ-cΔ)] / 2cΔ</p>
          <p className="text-purple-400 ml-4">F̂ ← QGT estimate  <span className="text-slate-600"># Fubini-Study metric (2 evals)</span></p>
          <p className="text-yellow-400 ml-4">g_feynman ← FeynmanGL(f, θ, k=0..11)  <span className="text-slate-600"># 12×n_pts evals</span></p>
          <p className="text-red-400 ml-4">g_eml ← g_feynman + g_spsa + λ·∇R_curv  <span className="text-slate-600"># Anti-barren-plateau</span></p>
          <p className="text-slate-400 ml-4">θ ← θ - a_t·Ĥ_inv·g_eml  <span className="text-slate-600"># Natural gradient step</span></p>
          <p className="text-green-400">return θ_best, loss_best</p>
        </div>
      </motion.div>
    </div>
  );
}
