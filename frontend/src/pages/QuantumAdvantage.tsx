import { motion } from 'framer-motion';
import SectionHeader from '../components/SectionHeader';
import StatCard from '../components/StatCard';
import { QFI_RESULTS, BARREN_PLATEAUS, HARDWARE_RESULTS } from '../data/mockData';
import { Zap, Shield, Cpu } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  Legend, LineChart, Line, ReferenceLine, ScatterChart, Scatter, ZAxis
} from 'recharts';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="card p-2 text-xs" style={{ border: '1px solid rgba(0,212,255,0.3)' }}>
        <p className="text-cyan-400 mb-1">{label || payload[0]?.payload?.parameter}</p>
        {payload.map((p: any) => (
          <p key={p.name} style={{ color: p.color }}>{p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</p>
        ))}
      </div>
    );
  }
  return null;
};

export default function QuantumAdvantage() {
  const qfiChartData = QFI_RESULTS.map(r => ({
    parameter: r.parameter,
    FQ: r.fq,
    FC: r.fc,
    advantage: r.advantage,
  }));

  const hardwareData = HARDWARE_RESULTS.map(r => ({
    backend: r.backend.replace('IBM Kingston', 'IBM').replace('(', '\n('),
    accuracy: r.accuracy,
    error: r.error,
  }));

  const gradientData = BARREN_PLATEAUS.map(bp => ({
    qubits: bp.qubits,
    variance: bp.variance,
    logVariance: Math.log10(bp.variance),
  }));

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Quantum Advantage Analysis"
        subtitle="Formal demonstration of QFI superiority over classical Fisher Information"
        badge="FORMAL PROOF"
      />

      {/* Key metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Max QFI/CFI Ratio" value="2.23×" delta="|R| LQG echoes" deltaPositive icon={<Zap size={16} />} color="#00d4ff" delay={0.1} />
        <StatCard label="Min QFI/CFI Ratio" value="1.75×" delta="Brans-Dicke Δs" deltaPositive icon={<Zap size={16} />} color="#7c3aed" delay={0.2} />
        <StatCard label="Expressibility" value="96%" delta="E = 0.043 nats" deltaPositive icon={<Cpu size={16} />} color="#00ff88" delay={0.3} />
        <StatCard label="Von Neumann Entropy" value="3.82 bits" delta="Max: 6 bits" deltaPositive icon={<Shield size={16} />} color="#ffd700" delay={0.4} />
      </div>

      {/* QFI vs CFI chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="card p-5"
      >
        <SectionHeader
          title="Quantum Fisher Information vs Classical Fisher Information"
          subtitle="FQ ≥ FC guaranteed by quantum Cramér-Rao bound · Var(θ̂) ≥ 1/FQ(θ)"
        />
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={qfiChartData} margin={{ bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="parameter" stroke="#ffffff20" tick={{ fontSize: 10, fill: '#9ca3af' }} angle={-20} textAnchor="end" interval={0} />
            <YAxis stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: '11px' }} />
            <Bar dataKey="FQ" name="QFI (FQ)" fill="#00d4ff" radius={[4, 4, 0, 0]} />
            <Bar dataKey="FC" name="CFI (FC)" fill="#7c3aed" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Advantage per parameter */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="card p-5"
        >
          <SectionHeader title="FQ/FC Advantage Ratio" subtitle="All parameters exceed the QFI > 1.5 criterion (O4 objective)" />
          <div className="space-y-3 mt-2">
            {QFI_RESULTS.map((r, i) => {
              const pct = ((r.advantage - 1) / 1.5) * 100;
              const colors = ['#00d4ff', '#7c3aed', '#00ff88', '#ffd700', '#ff4466'];
              return (
                <div key={r.parameter}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">{r.parameter}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-slate-500">{r.sigma}</span>
                      <span className="font-mono font-bold" style={{ color: colors[i] }}>{r.advantage.toFixed(2)}×</span>
                    </div>
                  </div>
                  <div className="progress-bar">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(pct, 100)}%` }}
                      transition={{ duration: 1.2, delay: i * 0.1 }}
                      className="progress-fill"
                      style={{ background: `linear-gradient(90deg, ${colors[i]}80, ${colors[i]})` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </motion.div>

        {/* Barren plateau analysis */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="card p-5"
        >
          <SectionHeader title="Barren Plateau Analysis" subtitle="Gradient variance vs qubit count · QNSPSA-EML mitigation" />
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={gradientData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="qubits" stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }}
                label={{ value: 'Qubits (n)', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 10 }} />
              <YAxis stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }}
                label={{ value: 'log₁₀(Var[∂L/∂θ])', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={-4} stroke="#ff446640" strokeDasharray="3 3" label={{ value: 'Plateau risk', fill: '#ff446670', fontSize: 9 }} />
              <Line type="monotone" dataKey="logVariance" name="log₁₀(Variance)" stroke="#00d4ff" strokeWidth={2} dot={{ fill: '#00d4ff', r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-3 space-y-1">
            {BARREN_PLATEAUS.map((bp) => (
              <div key={bp.qubits} className="flex items-center gap-2 text-xs">
                <span className="text-slate-500 w-16">n={bp.qubits} qubits</span>
                <div className="flex-1 progress-bar">
                  <div className="progress-fill" style={{ width: `${Math.abs(Math.log10(bp.variance) / 4) * 100}%`, background: bp.trainable === true ? '#00ff88' : '#ffd700' }} />
                </div>
                <span className="w-20 text-right font-mono" style={{ color: bp.trainable === true ? '#00ff88' : '#ffd700' }}>
                  {bp.trainable === true ? '✓ Trainable' : '△ Marginal'}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Hardware comparison */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
        className="card p-5"
      >
        <SectionHeader title="Real Hardware vs Simulation" subtitle="ZNE recovers 12pp of accuracy on ibm_kingston · p=0.003 (Student's t-test)" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={HARDWARE_RESULTS} margin={{ bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="backend" stroke="#ffffff20" tick={{ fontSize: 9, fill: '#9ca3af' }} interval={0} angle={-10} textAnchor="end" />
              <YAxis stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }} domain={[65, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={85} stroke="#00d4ff40" strokeDasharray="3 3" label={{ value: 'O3 target', fill: '#00d4ff80', fontSize: 9 }} />
              <Bar dataKey="accuracy" name="Accuracy (%)" radius={[4, 4, 0, 0]}>
                {HARDWARE_RESULTS.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? '#00ff88' : i === 1 ? '#ff4466' : '#00d4ff'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="space-y-3">
            <p className="text-xs text-slate-500 uppercase tracking-wider">Backend Details</p>
            {HARDWARE_RESULTS.map((hw, i) => {
              const colors = ['#00ff88', '#ff4466', '#00d4ff'];
              return (
                <div key={hw.backend} className="p-3 rounded-lg" style={{ background: `${colors[i]}08`, border: `1px solid ${colors[i]}20` }}>
                  <div className="flex justify-between mb-1">
                    <span className="text-xs font-medium" style={{ color: colors[i] }}>{hw.backend}</span>
                    <span className="text-xs font-mono text-white">{hw.accuracy}% ± {hw.error}%</span>
                  </div>
                  <div className="flex gap-4 text-xs text-slate-500">
                    <span>Time: {hw.timePerEvent}s/event</span>
                    <span>Shots: {hw.shots.toLocaleString()}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </motion.div>

      {/* Formulae */}
      <div className="card p-5 font-mono text-xs space-y-3">
        <div className="text-slate-500 uppercase tracking-wider mb-3 text-xs">Key Mathematical Formulas</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { label: 'Quantum Cramér-Rao Bound', formula: 'Var(θ̂) ≥ 1 / FQ(θ)', color: '#00d4ff' },
            { label: 'QFI Definition', formula: 'FQ(θk) = 4[⟨∂kψ|∂kψ⟩ - |⟨ψ|∂kψ⟩|²]', color: '#7c3aed' },
            { label: 'Parameter-Shift Rule', formula: '∂|ψ⟩/∂θk ≈ [|ψ(θ+π/2·ek)⟩ - |ψ(θ-π/2·ek)⟩] / 2', color: '#00ff88' },
            { label: 'Bonferroni Correction', formula: 'αBonferroni = αglobal / k = 2.87×10⁻⁷ / 6', color: '#ffd700' },
          ].map(item => (
            <div key={item.label} className="p-3 rounded-lg" style={{ background: `${item.color}08`, border: `1px solid ${item.color}20` }}>
              <div className="text-xs text-slate-500 mb-1">{item.label}</div>
              <div style={{ color: item.color }}>{item.formula}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
