import { useState } from 'react';
import { motion } from 'framer-motion';
import SectionHeader from '../components/SectionHeader';
import { CONFUSION_DATA, CONFUSION_LABELS, THEORY_RESULTS, TRAINING_HISTORY } from '../data/mockData';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, LineChart, Line, Legend, ReferenceLine
} from 'recharts';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="card p-2 text-xs" style={{ border: '1px solid rgba(0,212,255,0.3)' }}>
        <p className="text-cyan-400 mb-1">{label}</p>
        {payload.map((p: any) => (
          <p key={p.name} style={{ color: p.color }}>{p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</p>
        ))}
      </div>
    );
  }
  return null;
};

function getColor(value: number) {
  if (value > 0.9) return '#00ff88';
  if (value > 0.8) return '#7ec88a';
  if (value > 0.5) return '#ffd700';
  if (value > 0.1) return '#ff8833';
  return '#1e3a5f';
}

export default function TheoryClassifier() {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [selectedTheory, setSelectedTheory] = useState<string>('LQG Echoes');

  const bayesData = THEORY_RESULTS.map(t => ({
    name: t.theory,
    log10B: t.log10B,
    uncertainty: t.uncertainty,
    color: t.color,
  }));

  const selectedResult = THEORY_RESULTS.find(t => t.theory === selectedTheory);

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Theory Classifier"
        subtitle="QNIM Hilbert-space multiclass classifier · 13 competing theories"
        badge="91% ACCURACY"
      />

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Training samples', value: '800', sub: '10 classes × 80' },
          { label: 'Validation samples', value: '200', sub: 'Stratified' },
          { label: 'Best accuracy', value: '92.4%', sub: 'Aer simulator' },
          { label: 'Convergence', value: '34 epochs', sub: 'Early stopping' },
        ].map((item, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }} className="card p-4 text-center">
            <div className="text-lg font-bold text-cyan-400 mb-0.5">{item.value}</div>
            <div className="text-xs text-slate-400">{item.label}</div>
            <div className="text-xs text-slate-600">{item.sub}</div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion matrix */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="card p-5"
        >
          <SectionHeader title="Confusion Matrix" subtitle="Normalized · Hilbert space classifier output" />
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr>
                  <th className="w-8 text-slate-600 text-left pb-1 pr-1">↓Pred</th>
                  {CONFUSION_LABELS.map(l => (
                    <th key={l} className="text-slate-500 pb-1 font-mono w-8 text-center">{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {CONFUSION_DATA.map((row, ri) => (
                  <tr key={row.theory}>
                    <td className="text-slate-500 font-mono pr-2 py-0.5">{row.theory}</td>
                    {row.values.map((val, ci) => (
                      <td
                        key={ci}
                        onMouseEnter={() => setHoveredCell({ row: ri, col: ci })}
                        onMouseLeave={() => setHoveredCell(null)}
                        className="text-center py-0.5 w-8 cursor-default transition-all rounded"
                        style={{
                          background: getColor(val),
                          color: val > 0.4 ? '#000' : val > 0.05 ? '#fff' : '#6b7280',
                          fontWeight: val > 0.8 ? 'bold' : 'normal',
                          opacity: hoveredCell && (hoveredCell.row === ri || hoveredCell.col === ci) ? 1 : hoveredCell ? 0.6 : 1,
                        }}
                      >
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {hoveredCell && (
            <div className="mt-2 text-xs text-slate-500">
              Predicted: <span className="text-cyan-400">{CONFUSION_LABELS[hoveredCell.col]}</span> · True: <span className="text-purple-400">{CONFUSION_DATA[hoveredCell.row].theory}</span> · Value: <span className="text-white">{CONFUSION_DATA[hoveredCell.row].values[hoveredCell.col].toFixed(2)}</span>
            </div>
          )}
        </motion.div>

        {/* Training loss curves */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="card p-5"
        >
          <SectionHeader title="Training Dynamics" subtitle="Loss convergence over epochs" />
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={TRAINING_HISTORY}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="epoch" stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <YAxis stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }} domain={[0.1, 1]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <ReferenceLine y={0.219} stroke="#00d4ff40" strokeDasharray="3 3" label={{ value: 'Best Val Loss', fill: '#00d4ff80', fontSize: 9 }} />
              <Line type="monotone" dataKey="loss_train" name="Train Loss" stroke="#ff4466" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="loss_val" name="Val Loss" stroke="#00d4ff" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Bayes factors */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="card p-5"
      >
        <SectionHeader title="Bayes Factors — GW150914" subtitle="log₁₀(B) relative to GR null hypothesis · all within anecdotal range" />
        <div className="flex gap-4">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={bayesData} layout="vertical" margin={{ left: 80, right: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis type="number" stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }} domain={[-0.7, 0.7]} />
              <YAxis type="category" dataKey="name" stroke="#ffffff20" tick={{ fontSize: 10, fill: '#9ca3af' }} width={80} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine x={0} stroke="#ffffff30" />
              <ReferenceLine x={0.5} stroke="#ffd70040" strokeDasharray="3 3" label={{ value: 'Substantial', fill: '#ffd70060', fontSize: 9 }} />
              <ReferenceLine x={-0.5} stroke="#ffd70040" strokeDasharray="3 3" />
              <Bar dataKey="log10B" name="log₁₀(B)" radius={[0, 4, 4, 0]}>
                {bayesData.map((entry, index) => (
                  <Cell key={index} fill={entry.log10B > 0 ? '#7c3aed' : entry.log10B === 0 ? '#00ff88' : '#00d4ff'}
                    opacity={selectedTheory === entry.name ? 1 : 0.6}
                    cursor="pointer"
                    onClick={() => setSelectedTheory(entry.name)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Detail panel */}
          <div className="w-56 flex-shrink-0">
            <div className="card p-4 h-full">
              <p className="text-xs text-slate-500 mb-3">Selected Theory</p>
              {selectedResult && (
                <div className="space-y-3">
                  <div>
                    <div className="text-sm font-semibold text-white">{selectedResult.theory}</div>
                    <div className="text-xs mt-0.5 px-2 py-0.5 rounded inline-block" style={{ background: `${selectedResult.color}20`, color: selectedResult.color }}>
                      Layer {selectedResult.layer}
                    </div>
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-500">log₁₀(B)</span>
                      <span className="font-mono" style={{ color: selectedResult.log10B > 0 ? '#7c3aed' : selectedResult.log10B === 0 ? '#00ff88' : '#00d4ff' }}>
                        {selectedResult.log10B >= 0 ? '+' : ''}{selectedResult.log10B.toFixed(2)} ± {selectedResult.uncertainty.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Verdict</span>
                      <span className="text-slate-300">{selectedResult.interpretation}</span>
                    </div>
                  </div>
                  <div className="text-xs text-slate-500 leading-relaxed pt-2 border-t" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
                    {selectedResult.log10B > 0
                      ? 'Weak evidence for this theory. More events needed to reach decisive threshold (log₁₀B > 2).'
                      : selectedResult.log10B === 0
                        ? 'General Relativity is the reference model (log₁₀B = 0).'
                        : 'Evidence favors GR. Theory not excluded — future detectors required.'}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Architecture info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { title: 'Feature Map', desc: 'ChebyshevFeatureMap (reps=1) → 12 PCA components → SU(2¹²) Hilbert space', color: '#00d4ff' },
          { title: 'Ansatz', desc: 'EfficientSU2 (reps=2, linear entanglement) · Expressibility E = 0.043 nats', color: '#7c3aed' },
          { title: 'Optimizer', desc: 'QNSPSA + EML + Feynman path constraints · ~124 evaluations/iteration', color: '#00ff88' },
        ].map(item => (
          <div key={item.title} className="card p-4">
            <div className="text-sm font-semibold mb-1" style={{ color: item.color }}>{item.title}</div>
            <div className="text-xs text-slate-500 leading-relaxed">{item.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
