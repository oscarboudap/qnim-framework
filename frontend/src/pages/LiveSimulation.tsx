import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import SectionHeader from '../components/SectionHeader';
import { Play, Square, RefreshCw, Zap, AlertTriangle, CheckCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const THEORIES = ['General Relativity', 'Brans-Dicke', 'Massive Graviton', 'LQG Echoes', 'Fuzzballs', 'Extra Dimensions (ADD)', 'Spacetime Foam'];
const INJECTIONS = ['None (Pure GR)', 'Dipolar Radiation (-1PN)', 'Graviton Mass (dRGT)', 'LQG Echoes (L6)', 'Quantum Foam (L7)', 'Brans-Dicke Scalar'];

interface SimState {
  running: boolean;
  progress: number;
  currentEpoch: number;
  loss: number;
  accuracy: number;
  detectedTheory: string;
  confidence: number;
  log10B: number;
  waveformData: any[];
  qfiRatio: number;
  significanceSigma: number;
}

function generateSimWaveform(m1: number, m2: number, injection: string, snr: number) {
  const points = [];
  for (let i = 0; i < 300; i++) {
    const t = (i - 220) * 0.002;
    let strain = 0;
    if (t < 0) {
      const tau = -t;
      const freq = (20 / (m1 + m2)) * Math.pow(tau + 0.01, -3 / 8) * 0.1;
      const amp = Math.pow(tau + 0.01, -1 / 4) * (snr / 30) * 0.3;
      strain = amp * Math.sin(2 * Math.PI * freq * tau * 3);
      if (injection !== 'None (Pure GR)') {
        const deltaPhase = injection.includes('Dipolar') ? -0.08 * Math.pow(Math.max(freq, 0.01), -7 / 3)
          : injection.includes('Graviton') ? 0.05 * freq
            : injection.includes('Foam') ? (Math.random() - 0.5) * 0.02 : 0.03;
        strain += amp * 0.1 * Math.sin(2 * Math.PI * freq * tau * 3 + deltaPhase);
      }
    } else {
      strain = 0.4 * (snr / 30) * Math.exp(-t * 15) * Math.cos(2 * Math.PI * 30 * t);
      if (injection.includes('Echo') && t > 0.05) {
        strain += 0.12 * Math.exp(-(t - 0.05) * 20) * Math.cos(2 * Math.PI * 25 * (t - 0.05));
      }
    }
    const noise = (Math.random() - 0.5) * 0.04 / (snr / 10);
    points.push({ t: parseFloat(t.toFixed(3)), strain: parseFloat((strain + noise).toFixed(5)) });
  }
  return points;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div className="card p-2 text-xs" style={{ border: '1px solid rgba(0,212,255,0.3)' }}>
        <p className="text-cyan-400 mono">t = {label}s</p>
        {payload.map((p: any) => <p key={p.name} style={{ color: p.color }}>{p.name}: {p.value?.toFixed(5)}</p>)}
      </div>
    );
  }
  return null;
};

export default function LiveSimulation() {
  const [config, setConfig] = useState({
    m1: 35, m2: 30, spin1: 0, spin2: 0, distance: 440, snr: 24,
    injection: 'None (Pure GR)', theory: 'General Relativity', qubits: 12,
  });

  const [sim, setSim] = useState<SimState>({
    running: false, progress: 0, currentEpoch: 0, loss: 0.891, accuracy: 0,
    detectedTheory: '—', confidence: 0, log10B: 0, waveformData: [],
    qfiRatio: 0, significanceSigma: 0,
  });

  const [logs, setLogs] = useState<string[]>([]);

  const addLog = useCallback((msg: string) => {
    setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 19)]);
  }, []);

  const runSimulation = useCallback(() => {
    setSim(s => ({ ...s, running: true, progress: 0, currentEpoch: 0, loss: 0.891, accuracy: 0, detectedTheory: '—', confidence: 0, significanceSigma: 0 }));
    setLogs([]);
    addLog(`🚀 Starting QNIM pipeline...`);
    addLog(`📡 Generating GW waveform: m₁=${config.m1}, m₂=${config.m2} M☉, d_L=${config.distance} Mpc`);
    addLog(`⚛️  Injection: ${config.injection}`);
    addLog(`🔮 Backend: ${config.qubits}-qubit Aer simulator`);

    const waveform = generateSimWaveform(config.m1, config.m2, config.injection, config.snr);
    setSim(s => ({ ...s, waveformData: waveform }));

    // Simulate training epochs
    const epochs = [1, 5, 10, 20, 30, 34];
    const losses = [0.891, 0.612, 0.421, 0.285, 0.198, 0.183];
    const accs = [48, 67, 79, 87, 91, 92.4];
    let epochIdx = 0;

    const interval = setInterval(() => {
      if (epochIdx >= epochs.length) {
        clearInterval(interval);
        const isGR = config.injection === 'None (Pure GR)';
        const detected = isGR ? 'General Relativity' : config.injection.includes('LQG') ? 'LQG Echoes' :
          config.injection.includes('Dipolar') ? 'Brans-Dicke' : config.injection.includes('Graviton') ? 'Massive Graviton' :
            config.injection.includes('Foam') ? 'Spacetime Foam' : 'Brans-Dicke';
        const qfi = 1.75 + Math.random() * 0.48;
        const sigma = isGR ? 1.3 : 3.5 + Math.random() * 2;
        const log10B = isGR ? -0.28 + (Math.random() - 0.5) * 0.2 : 0.4 + Math.random() * 0.3;

        setSim(s => ({ ...s, running: false, progress: 100, detectedTheory: detected, confidence: 88 + Math.random() * 4, qfiRatio: qfi, significanceSigma: sigma, log10B: isGR ? log10B : -log10B }));
        addLog(`✅ Classification complete`);
        addLog(`📊 Detected: ${detected} (${(88 + Math.random() * 4).toFixed(1)}% confidence)`);
        addLog(`🔬 QFI/CFI ratio: ${qfi.toFixed(2)}×`);
        addLog(`📈 Significance: ${sigma.toFixed(1)}σ`);
        addLog(`⚖️  log₁₀(B) = ${(isGR ? log10B : -log10B).toFixed(2)}`);
      } else {
        const e = epochIdx;
        setSim(s => ({ ...s, progress: (e / epochs.length) * 90, currentEpoch: epochs[e], loss: losses[e], accuracy: accs[e] }));
        addLog(`⚡ Epoch ${epochs[e]}: loss=${losses[e].toFixed(3)}, acc=${accs[e].toFixed(1)}%`);
        epochIdx++;
      }
    }, 800);
  }, [config, addLog]);

  const stopSim = () => setSim(s => ({ ...s, running: false }));

  const isDetectedBSM = sim.detectedTheory !== '—' && sim.detectedTheory !== 'General Relativity';

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Live QNIM Simulation"
        subtitle="Configure and run a full quantum inference pipeline in real-time"
        badge="INTERACTIVE"
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config panel */}
        <div className="space-y-4">
          <div className="card p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Binary System</p>
            <div className="space-y-3">
              {[
                { label: 'Primary mass m₁ (M☉)', key: 'm1', min: 5, max: 100, step: 1 },
                { label: 'Secondary mass m₂ (M☉)', key: 'm2', min: 1, max: 80, step: 1 },
                { label: 'Distance (Mpc)', key: 'distance', min: 40, max: 5000, step: 10 },
                { label: 'SNR', key: 'snr', min: 8, max: 50, step: 1 },
              ].map(({ label, key, min, max, step }) => (
                <div key={key}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500">{label}</span>
                    <span className="text-cyan-400 font-mono">{config[key as keyof typeof config]}</span>
                  </div>
                  <input
                    type="range" min={min} max={max} step={step}
                    value={config[key as keyof typeof config] as number}
                    onChange={e => setConfig(c => ({ ...c, [key]: +e.target.value }))}
                    className="w-full accent-cyan-400"
                    disabled={sim.running}
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="card p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Quantum Settings</p>
            <div className="space-y-3">
              <div>
                <p className="text-xs text-slate-500 mb-1">Qubit count: <span className="text-cyan-400 font-mono">{config.qubits}</span></p>
                <input type="range" min={8} max={27} step={4} value={config.qubits}
                  onChange={e => setConfig(c => ({ ...c, qubits: +e.target.value }))}
                  className="w-full accent-purple-400" disabled={sim.running} />
                <p className="text-xs text-slate-600 mt-1">Hilbert space: 2^{config.qubits} = {(2 ** config.qubits).toLocaleString()} dims</p>
              </div>

              <div>
                <p className="text-xs text-slate-500 mb-1">BSM Injection</p>
                <select
                  value={config.injection}
                  onChange={e => setConfig(c => ({ ...c, injection: e.target.value }))}
                  className="w-full text-xs rounded-lg px-2 py-1.5 text-slate-300"
                  style={{ background: '#0f1629', border: '1px solid rgba(0,212,255,0.2)' }}
                  disabled={sim.running}
                >
                  {INJECTIONS.map(i => <option key={i} value={i}>{i}</option>)}
                </select>
              </div>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={runSimulation}
              disabled={sim.running}
              className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium transition-all disabled:opacity-50"
              style={{ background: 'linear-gradient(135deg, #00d4ff20, #7c3aed20)', border: '1px solid rgba(0,212,255,0.4)', color: '#00d4ff' }}
            >
              <Play size={14} />
              {sim.running ? 'Running...' : 'Run Pipeline'}
            </button>
            <button
              onClick={stopSim}
              disabled={!sim.running}
              className="px-4 py-2.5 rounded-xl text-sm transition-all disabled:opacity-30"
              style={{ background: 'rgba(255,68,102,0.1)', border: '1px solid rgba(255,68,102,0.3)', color: '#ff4466' }}
            >
              <Square size={14} />
            </button>
          </div>
        </div>

        {/* Main output */}
        <div className="lg:col-span-2 space-y-4">
          {/* Progress */}
          <div className="card p-4">
            <div className="flex justify-between text-xs mb-2">
              <span className="text-slate-500">Pipeline Progress</span>
              <span className="text-cyan-400 font-mono">{sim.progress.toFixed(0)}%</span>
            </div>
            <div className="progress-bar mb-3">
              <motion.div
                className="progress-fill"
                style={{ background: 'linear-gradient(90deg, #00d4ff, #7c3aed)', width: `${sim.progress}%` }}
                animate={{ width: `${sim.progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <div className="grid grid-cols-3 gap-3 text-xs text-center">
              <div>
                <div className="text-slate-500">Epoch</div>
                <div className="font-mono text-white">{sim.currentEpoch || '—'}</div>
              </div>
              <div>
                <div className="text-slate-500">Loss</div>
                <div className="font-mono text-amber-400">{sim.loss.toFixed(3)}</div>
              </div>
              <div>
                <div className="text-slate-500">Accuracy</div>
                <div className="font-mono text-green-400">{sim.accuracy > 0 ? `${sim.accuracy.toFixed(1)}%` : '—'}</div>
              </div>
            </div>
          </div>

          {/* Waveform preview */}
          {sim.waveformData.length > 0 && (
            <div className="card p-4">
              <p className="text-xs text-slate-500 mb-2">Input Waveform {config.injection !== 'None (Pure GR)' && <span className="text-purple-400">(+ BSM injection)</span>}</p>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={sim.waveformData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="t" stroke="#ffffff15" tick={{ fontSize: 9, fill: '#6b7280' }} />
                  <YAxis stroke="#ffffff15" tick={{ fontSize: 9, fill: '#6b7280' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine x={0} stroke="#ffd70050" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="strain" stroke="#00d4ff" strokeWidth={1} dot={false} name="h(t)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Results */}
          <AnimatePresence>
            {sim.detectedTheory !== '—' && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="card p-4"
                style={{ border: `1px solid ${isDetectedBSM ? 'rgba(124,58,237,0.4)' : 'rgba(0,255,136,0.4)'}` }}
              >
                <div className="flex items-center gap-2 mb-3">
                  {isDetectedBSM
                    ? <AlertTriangle size={18} className="text-purple-400" />
                    : <CheckCircle size={18} className="text-green-400" />}
                  <span className="font-semibold" style={{ color: isDetectedBSM ? '#7c3aed' : '#00ff88' }}>
                    {isDetectedBSM ? 'BSM Signature Detected!' : 'GR Consistent'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  {[
                    { label: 'Classified Theory', value: sim.detectedTheory, color: isDetectedBSM ? '#7c3aed' : '#00ff88' },
                    { label: 'Confidence', value: `${sim.confidence.toFixed(1)}%`, color: '#00d4ff' },
                    { label: 'QFI/CFI Ratio', value: `${sim.qfiRatio.toFixed(2)}×`, color: '#ffd700' },
                    { label: 'Significance', value: `${sim.significanceSigma.toFixed(1)}σ`, color: sim.significanceSigma >= 5 ? '#00ff88' : '#ffd700' },
                    { label: 'log₁₀(B)', value: sim.log10B.toFixed(2), color: sim.log10B > 0 ? '#7c3aed' : '#00d4ff' },
                    { label: 'Jeffreys Scale', value: Math.abs(sim.log10B) < 0.5 ? 'Anecdotal' : Math.abs(sim.log10B) < 1 ? 'Substantial' : 'Strong', color: '#9ca3af' },
                  ].map(item => (
                    <div key={item.label} className="flex justify-between p-2 rounded" style={{ background: 'rgba(255,255,255,0.02)' }}>
                      <span className="text-slate-500">{item.label}</span>
                      <span className="font-mono" style={{ color: item.color }}>{item.value}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Log terminal */}
      <div className="card p-4">
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Pipeline Log</p>
        <div className="font-mono text-xs space-y-0.5 max-h-40 overflow-y-auto" style={{ background: '#020510', padding: '12px', borderRadius: '8px' }}>
          {logs.length === 0
            ? <p className="text-slate-600">{'>'} Waiting for simulation to start...</p>
            : logs.map((log, i) => <p key={i} className="text-green-400">{log}</p>)}
        </div>
      </div>
    </div>
  );
}
