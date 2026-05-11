import { motion } from 'framer-motion';
import SectionHeader from '../components/SectionHeader';
import { SNR_ACCURACY, HARDWARE_RESULTS, DETECTION_EFFICIENCY, BARREN_PLATEAUS } from '../data/mockData';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine,
  BarChart, Bar, Cell
} from 'recharts';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div className="card p-2 text-xs" style={{ border: '1px solid rgba(0,212,255,0.3)' }}>
        <p className="text-cyan-400 mb-1">{label}</p>
        {payload.map((p: any) => <p key={p.name} style={{ color: p.color }}>{p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</p>)}
      </div>
    );
  }
  return null;
};

export default function MetricsHardware() {
  return (
    <div className="space-y-6">
      <SectionHeader
        title="Metrics & Hardware Performance"
        subtitle="Full benchmarking suite: accuracy, scalability, and real-hardware validation"
        badge="BENCHMARKS"
      />

      {/* Hardware cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { backend: 'ibm_kingston', qubits: 27, topology: 'Heavy-Hex', t2: '100–150 µs', ecx: '0.001', status: '✓ Used in TFM', color: '#00d4ff' },
          { backend: 'ibm_brisbane', qubits: 127, topology: 'Heavy-Hex', t2: '80–120 µs', ecx: '0.002', status: '✓ Eagle r3', color: '#7c3aed' },
          { backend: 'ibm_torino', qubits: 133, topology: 'Heron', t2: '200–300 µs', ecx: '0.0005', status: '✓ Best (2026)', color: '#00ff88' },
        ].map((hw) => (
          <motion.div
            key={hw.backend}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-4"
            style={{ border: `1px solid ${hw.color}25` }}
          >
            <div className="flex justify-between items-start mb-3">
              <div>
                <div className="font-mono text-sm font-medium" style={{ color: hw.color }}>{hw.backend}</div>
                <div className="text-xs text-slate-500">{hw.topology}</div>
              </div>
              <div className="text-right">
                <div className="text-xl font-bold text-white">{hw.qubits}</div>
                <div className="text-xs text-slate-500">qubits</div>
              </div>
            </div>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500">T₂ coherence</span>
                <span className="font-mono text-slate-300">{hw.t2}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">ε_CX error</span>
                <span className="font-mono text-slate-300">{hw.ecx}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Hilbert space</span>
                <span className="font-mono text-slate-300">2^{hw.qubits}</span>
              </div>
            </div>
            <div className="mt-3 px-2 py-1 rounded text-xs text-center" style={{ background: `${hw.color}10`, color: hw.color }}>
              {hw.status}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Accuracy vs SNR */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card p-5"
        >
          <SectionHeader title="Accuracy vs SNR" subtitle="VQC-QNIM outperforms ResNet-18 at all SNR levels" />
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={SNR_ACCURACY}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="snr" stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }} label={{ value: 'SNR', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 10 }} />
              <YAxis stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }} domain={[60, 100]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <ReferenceLine y={85} stroke="#00d4ff30" strokeDasharray="3 3" label={{ value: 'O3 criterion', fill: '#00d4ff50', fontSize: 9 }} />
              <Line type="monotone" dataKey="vqc12" name="QNIM (12q)" stroke="#00d4ff" strokeWidth={2.5} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="vqc27" name="QNIM (27q)" stroke="#00ff88" strokeWidth={2.5} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="resnet" name="ResNet-18" stroke="#7c3aed" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Hardware comparison bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card p-5"
        >
          <SectionHeader title="Hardware vs Simulator" subtitle="ZNE recovers 12pp accuracy on real hardware" />
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={HARDWARE_RESULTS} margin={{ bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="backend" stroke="#ffffff20" tick={{ fontSize: 9, fill: '#9ca3af' }} interval={0} angle={-15} textAnchor="end" />
              <YAxis stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }} domain={[65, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={85} stroke="#00d4ff30" strokeDasharray="3 3" />
              <Bar dataKey="accuracy" name="Accuracy (%)" radius={[4, 4, 0, 0]}>
                {HARDWARE_RESULTS.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? '#00ff88' : i === 1 ? '#ff4466' : '#00d4ff'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Circuit error formula */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card p-5"
      >
        <SectionHeader title="Circuit Error Scaling" subtitle="ε_total ≈ 1 − (1 − ε_CX)^(n·D/2) · Total error vs qubit count at D=4" />
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={[8, 12, 16, 20, 27, 127, 133].map(n => ({
            qubits: n,
            heron: (1 - (1 - 0.0005) ** (n * 2)) * 100,
            eagle: (1 - (1 - 0.002) ** (n * 2)) * 100,
            kingston: (1 - (1 - 0.001) ** (n * 2)) * 100,
          }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="qubits" stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }} label={{ value: 'Qubits (n)', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 10 }} />
            <YAxis stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }} label={{ value: 'ε_total (%)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: '11px' }} />
            <ReferenceLine y={10} stroke="#ffd70030" strokeDasharray="3 3" label={{ value: '10% error limit', fill: '#ffd70050', fontSize: 9 }} />
            <Line type="monotone" dataKey="heron" name="ibm_torino (Heron)" stroke="#00ff88" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="kingston" name="ibm_kingston" stroke="#00d4ff" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="eagle" name="ibm_brisbane (Eagle)" stroke="#7c3aed" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Cost table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="card p-5"
      >
        <SectionHeader title="Deployment Cost Estimation" subtitle="2026 IBM Quantum pricing · 1,000 synthetic events" />
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b" style={{ borderColor: 'rgba(255,255,255,0.08)' }}>
                <th className="text-left py-2 font-medium">Backend</th>
                <th className="text-center py-2">Time/Event (s)</th>
                <th className="text-center py-2">Rate ($/sec)</th>
                <th className="text-center py-2">Total Est.</th>
                <th className="text-center py-2">Funding</th>
              </tr>
            </thead>
            <tbody>
              {[
                { backend: 'ibm_kingston (27q)', time: 180, rate: 1.60, total: 288000, funding: 'Academic Grant' },
                { backend: 'ibm_torino (133q)', time: 120, rate: 2.10, total: 252000, funding: 'Research Grant' },
                { backend: 'Aer Simulator', time: 2, rate: 0, total: 0, funding: 'Free' },
              ].map((row, i) => (
                <tr key={i} className="border-b hover:bg-white/2" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
                  <td className="py-2.5 font-mono text-slate-300">{row.backend}</td>
                  <td className="py-2.5 text-center font-mono text-cyan-400">{row.time}</td>
                  <td className="py-2.5 text-center font-mono text-purple-400">${row.rate.toFixed(2)}</td>
                  <td className="py-2.5 text-center font-mono" style={{ color: row.total === 0 ? '#00ff88' : '#ffd700' }}>
                    {row.total === 0 ? '$0' : `$${row.total.toLocaleString()}`}
                  </td>
                  <td className="py-2.5 text-center text-slate-400">{row.funding}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-slate-600 mt-2">Note: All real hardware executions performed under Academic Research Grant 2025-042.</p>
      </motion.div>
    </div>
  );
}
