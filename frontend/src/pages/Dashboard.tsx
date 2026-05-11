import { motion } from 'framer-motion';
import { Atom, Zap, BarChart3, Brain, Activity, TrendingUp, Cpu, Globe } from 'lucide-react';
import StatCard from '../components/StatCard';
import SectionHeader from '../components/SectionHeader';
import { TRAINING_HISTORY, SNR_ACCURACY, DETECTION_EFFICIENCY } from '../data/mockData';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="card p-3 text-xs" style={{ border: '1px solid rgba(0,212,255,0.3)' }}>
        <p className="text-cyan-400 font-mono mb-1">{label}</p>
        {payload.map((p: any) => (
          <p key={p.name} style={{ color: p.color }}>{p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</p>
        ))}
      </div>
    );
  }
  return null;
};

const radarData = [
  { subject: 'GR vs BSM', QNIM: 91, Classical: 85 },
  { subject: 'LQG vs Fuzzballs', QNIM: 89, Classical: 52 },
  { subject: 'Param. Precision', QNIM: 94, Classical: 76 },
  { subject: 'Low SNR Detect.', QNIM: 68, Classical: 64 },
  { subject: 'High SNR Detect.', QNIM: 97, Classical: 89 },
  { subject: 'Multi-Theory', QNIM: 86, Classical: 48 },
];

export default function Dashboard() {
  return (
    <div className="space-y-8">
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative overflow-hidden rounded-2xl p-8"
        style={{
          background: 'linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(124,58,237,0.08) 50%, rgba(0,255,136,0.05) 100%)',
          border: '1px solid rgba(0,212,255,0.2)',
        }}
      >
        <div className="absolute top-0 right-0 w-64 h-64 opacity-10"
          style={{ background: 'radial-gradient(circle, #00d4ff 0%, transparent 70%)', transform: 'translate(30%, -30%)' }} />
        <div className="relative z-10">
          <div className="flex items-center gap-2 mb-3">
            <span className="px-2 py-1 rounded text-xs font-mono"
              style={{ background: 'rgba(0,255,136,0.1)', color: '#00ff88', border: '1px solid rgba(0,255,136,0.3)' }}>
              v2.0 · NISQ ERA
            </span>
            <span className="px-2 py-1 rounded text-xs font-mono"
              style={{ background: 'rgba(0,212,255,0.1)', color: '#00d4ff', border: '1px solid rgba(0,212,255,0.3)' }}>
              IBM KINGSTON VALIDATED
            </span>
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">
            <span className="gradient-text-blue">QNIM</span> — Quantum NeuroInspired Manifold
          </h1>
          <p className="text-slate-400 max-w-2xl text-sm leading-relaxed">
            Framework cuántico para el análisis de ondas gravitacionales y detección de física más allá del Modelo Estándar.
            Variational Quantum Circuits (VQC) sobre IBM Quantum + optimización QUBO via D-Wave Neal.
          </p>
          <div className="flex flex-wrap gap-4 mt-4">
            {[
              { label: 'QFI/CFI Ratio', value: '1.75–2.23', color: '#00d4ff' },
              { label: 'Accuracy', value: '91 ± 2%', color: '#00ff88' },
              { label: 'Significance', value: '5.3σ', color: '#ffd700' },
              { label: 'Theories', value: '13 BSM', color: '#7c3aed' },
            ].map((item) => (
              <div key={item.label} className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full" style={{ background: item.color }} />
                <span className="text-xs text-slate-500">{item.label}:</span>
                <span className="text-xs font-bold" style={{ color: item.color }}>{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Validation Accuracy" value="91.0" unit="%" delta="vs 85% classical" deltaPositive icon={<BarChart3 size={16} />} color="#00ff88" delay={0.1} />
        <StatCard label="QFI Advantage" value="2.23×" delta="LQG echoes param." deltaPositive icon={<Zap size={16} />} color="#00d4ff" delay={0.2} />
        <StatCard label="Statistical Significance" value="5.3σ" delta="Bonferroni corrected" deltaPositive icon={<TrendingUp size={16} />} color="#ffd700" delay={0.3} />
        <StatCard label="Training Epochs" value="34" unit="epochs" delta="Early stopping" deltaPositive={false} icon={<Cpu size={16} />} color="#7c3aed" delay={0.4} />
      </div>

      {/* Training & SNR comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training curves */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="card p-5"
        >
          <SectionHeader title="Training Convergence" subtitle="QNSPSA-EML-Feynman optimizer · 5-run average" />
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={TRAINING_HISTORY}>
              <defs>
                <linearGradient id="accTrain" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00ff88" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00ff88" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="accVal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="epoch" stroke="#ffffff30" tick={{ fontSize: 11, fill: '#6b7280' }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }} />
              <YAxis stroke="#ffffff30" tick={{ fontSize: 11, fill: '#6b7280' }} domain={[40, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Area type="monotone" dataKey="acc_train" name="Train Acc %" stroke="#00ff88" fill="url(#accTrain)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="acc_val" name="Val Acc %" stroke="#00d4ff" fill="url(#accVal)" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Radar chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="card p-5"
        >
          <SectionHeader title="QNIM vs Classical ML" subtitle="Performance radar across key tasks" />
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.08)" />
              <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9, fill: '#6b7280' }} />
              <Radar name="QNIM" dataKey="QNIM" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.2} strokeWidth={2} />
              <Radar name="Classical" dataKey="Classical" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.15} strokeWidth={2} />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Tooltip content={<CustomTooltip />} />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Detection efficiency */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
        className="card p-5"
      >
        <SectionHeader title="Detection Efficiency by Theory & SNR" subtitle="Bonferroni-corrected · k=6 independent tests" badge="5σ PIPELINE" />
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={[
            { snr: 8, ...Object.fromEntries(DETECTION_EFFICIENCY.map(d => [d.model, d.snr8])) },
            { snr: 12, ...Object.fromEntries(DETECTION_EFFICIENCY.map(d => [d.model, d.snr12])) },
            { snr: 20, ...Object.fromEntries(DETECTION_EFFICIENCY.map(d => [d.model, d.snr20])) },
            { snr: 30, ...Object.fromEntries(DETECTION_EFFICIENCY.map(d => [d.model, d.snr30])) },
            { snr: 50, ...Object.fromEntries(DETECTION_EFFICIENCY.map(d => [d.model, d.snr50])) },
          ]}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="snr" stroke="#ffffff30" tick={{ fontSize: 11, fill: '#6b7280' }} label={{ value: 'SNR', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }} />
            <YAxis stroke="#ffffff30" tick={{ fontSize: 11, fill: '#6b7280' }} label={{ value: 'Efficiency (%)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: '11px' }} />
            {DETECTION_EFFICIENCY.map((d, i) => {
              const colors = ['#00d4ff', '#7c3aed', '#00ff88', '#ffd700', '#ff4466'];
              return <Line key={d.model} type="monotone" dataKey={d.model} stroke={colors[i]} strokeWidth={2} dot={{ r: 3 }} />;
            })}
          </LineChart>
        </ResponsiveContainer>
      </motion.div>

      {/* SNR accuracy comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="card p-5"
        >
          <SectionHeader title="VQC-QNIM vs ResNet-18" subtitle="Accuracy comparison at different SNR levels" />
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={SNR_ACCURACY}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="snr" stroke="#ffffff30" tick={{ fontSize: 11, fill: '#6b7280' }} />
              <YAxis stroke="#ffffff30" tick={{ fontSize: 11, fill: '#6b7280' }} domain={[60, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Line type="monotone" dataKey="vqc12" name="VQC-QNIM (12q)" stroke="#00d4ff" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="vqc27" name="VQC-QNIM (27q)" stroke="#00ff88" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="resnet" name="ResNet-18" stroke="#7c3aed" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Key milestones */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.7 }}
          className="card p-5"
        >
          <SectionHeader title="Project Objectives Status" subtitle="TFM validation criteria" />
          <div className="space-y-3">
            {[
              { label: 'O1 – DDD Architecture (156 modules)', done: true, value: '100%' },
              { label: 'O2 – SSTG Engine (KL-div > 0.3 nats)', done: true, value: '0.41 nats' },
              { label: 'O3 – IBM Quantum VQC ≥ 85%', done: true, value: '86.1% (ZNE)' },
              { label: 'O4 – QFI/CFI Advantage > 1.5', done: true, value: '1.75–2.23×' },
              { label: 'O5 – 5σ Bonferroni Pipeline', done: true, value: '85% @ SNR=30' },
              { label: 'O6 – GW150914 Reanalysis', done: true, value: '|ΔΩ| < 1σ' },
            ].map((item, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className={`w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0 text-xs ${item.done ? 'bg-green-500' : 'bg-slate-600'}`}>
                  {item.done ? '✓' : '○'}
                </div>
                <span className="text-xs text-slate-400 flex-1">{item.label}</span>
                <span className="text-xs font-mono" style={{ color: item.done ? '#00ff88' : '#6b7280' }}>{item.value}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Layers of physics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.8 }}
        className="card p-5"
      >
        <SectionHeader title="QNIM Physical Layer Hierarchy" subtitle="7 layers from Classical GR to Deep Quantum Corrections" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { layer: 'L5', name: 'Beyond-GR Physics', desc: 'Brans-Dicke, Massive Graviton, Extra Dimensions, LIV, k-essence', snr: 12, color: '#00d4ff', theories: 5 },
            { layer: 'L6', name: 'Horizon Quantum Topology', desc: 'LQG Echoes, Fuzzballs, Discrete Area Spectrum, Gravitational Memory', snr: 25, color: '#7c3aed', theories: 4 },
            { layer: 'L7', name: 'Deep Quantum Corrections', desc: 'Hawking Radiation, Spacetime Foam, Zeta Regularization', snr: 40, color: '#ff4466', theories: 3 },
          ].map((item) => (
            <div key={item.layer} className="rounded-xl p-4" style={{ background: `${item.color}08`, border: `1px solid ${item.color}25` }}>
              <div className="flex items-center gap-2 mb-2">
                <span className="font-mono text-sm font-bold" style={{ color: item.color }}>{item.layer}</span>
                <span className="text-xs text-slate-400">{item.name}</span>
              </div>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">{item.desc}</p>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Min SNR: <span style={{ color: item.color }} className="font-mono">{item.snr}</span></span>
                <span className="text-slate-500">Theories: <span style={{ color: item.color }} className="font-mono">{item.theories}</span></span>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
