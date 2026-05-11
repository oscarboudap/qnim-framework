import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, RefreshCw, Download } from 'lucide-react';
import SectionHeader from '../components/SectionHeader';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Label
} from 'recharts';
import { GW_EVENTS } from '../data/mockData';

// Generate a realistic GW waveform (chirp + ringdown)
function generateWaveform(m1: number, m2: number, distance: number, snr: number, addBSM = false) {
  const points = [];
  const M = m1 + m2;
  const eta = (m1 * m2) / (M * M);
  const chirpMass = M * Math.pow(eta, 0.6);
  const tMerge = 0.4; // seconds before merger
  const fScale = 20 / chirpMass; // frequency scale

  for (let i = 0; i < 400; i++) {
    const t = (i - 300) * 0.002; // -0.6 to 0.2 seconds
    let strain = 0;
    let strainBSM = 0;

    if (t < 0) {
      // Inspiral: chirp
      const tau = -t;
      const freq = fScale * Math.pow(tau + 0.01, -3 / 8) * 0.1;
      const amp = Math.pow(tau + 0.01, -1 / 4) * (snr / 30) * 0.3;
      strain = amp * Math.sin(2 * Math.PI * freq * tau * 3);
      if (addBSM) {
        // Brans-Dicke dipolar correction
        const deltaPhase = -0.05 * Math.pow(freq * 100, -7 / 3);
        strainBSM = amp * Math.sin(2 * Math.PI * freq * tau * 3 + deltaPhase) * 1.05;
      }
    } else {
      // Ringdown
      const fQNM = (1.5251 - 1.1568 * Math.pow(0.3, 0.1292)) / (2 * Math.PI * M * 4.93e-6);
      const tau_rd = t;
      strain = 0.4 * (snr / 30) * Math.exp(-tau_rd * 15) * Math.cos(2 * Math.PI * Math.min(fQNM / 100, 50) * tau_rd);
      if (addBSM) {
        // LQG echo after ringdown
        const echoDelay = 0.05;
        if (tau_rd > echoDelay) {
          strainBSM = strain + 0.15 * Math.exp(-(tau_rd - echoDelay) * 20) * Math.cos(2 * Math.PI * 30 * (tau_rd - echoDelay));
        } else {
          strainBSM = strain;
        }
      }
    }

    // Add noise
    const noise = (Math.random() - 0.5) * 0.04 / (snr / 10);
    points.push({
      t: parseFloat(t.toFixed(3)),
      strain: parseFloat((strain + noise).toFixed(5)),
      strainGR: parseFloat(strain.toFixed(5)),
      strainBSM: addBSM ? parseFloat((strainBSM + noise).toFixed(5)) : undefined,
    });
  }
  return points;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="card p-2 text-xs" style={{ border: '1px solid rgba(0,212,255,0.3)' }}>
        <p className="text-cyan-400 mono">t = {label}s</p>
        {payload.map((p: any) => (
          <p key={p.name} style={{ color: p.color }}>{p.name}: {p.value?.toFixed(5)}</p>
        ))}
      </div>
    );
  }
  return null;
};

export default function WaveformVisualizer() {
  const [selectedEvent, setSelectedEvent] = useState(GW_EVENTS[0]);
  const [showBSM, setShowBSM] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [snr, setSnr] = useState(24);
  const [data, setData] = useState<any[]>([]);
  const [phase, setPhase] = useState<'inspiral' | 'merger' | 'ringdown'>('inspiral');
  const animRef = useRef<any>(null);

  useEffect(() => {
    setData(generateWaveform(selectedEvent.m1, selectedEvent.m2, selectedEvent.dL, snr, showBSM));
  }, [selectedEvent, snr, showBSM]);

  useEffect(() => {
    if (playing) {
      let t = 0;
      animRef.current = setInterval(() => {
        t = (t + 1) % 100;
        if (t < 40) setPhase('inspiral');
        else if (t < 60) setPhase('merger');
        else setPhase('ringdown');
      }, 50);
    } else {
      clearInterval(animRef.current);
    }
    return () => clearInterval(animRef.current);
  }, [playing]);

  const phaseColors = { inspiral: '#00d4ff', merger: '#ffd700', ringdown: '#7c3aed' };

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Gravitational Wave Visualizer"
        subtitle="Interactive waveform display with beyond-GR signature injection"
        badge="REAL-TIME"
      />

      {/* Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Event selector */}
        <div className="card p-4 lg:col-span-1">
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Select GW Event</p>
          <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
            {GW_EVENTS.map((ev) => (
              <button
                key={ev.id}
                onClick={() => { setSelectedEvent(ev); setSnr(Math.round(ev.snr)); }}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-all ${selectedEvent.id === ev.id ? 'bg-cyan-900/30 border border-cyan-500/40 text-cyan-300' : 'hover:bg-white/5 text-slate-400 border border-transparent'}`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-mono font-medium">{ev.name}</span>
                  <span className="px-1.5 py-0.5 rounded text-xs"
                    style={{ background: ev.type === 'BBH' ? '#00d4ff15' : ev.type === 'BNS' ? '#00ff8815' : '#7c3aed15', color: ev.type === 'BBH' ? '#00d4ff' : ev.type === 'BNS' ? '#00ff88' : '#7c3aed' }}>
                    {ev.type}
                  </span>
                </div>
                <div className="text-slate-500 mt-0.5">SNR: {ev.snr} · {ev.m1}+{ev.m2} M☉</div>
              </button>
            ))}
          </div>
        </div>

        {/* Event parameters */}
        <div className="card p-4">
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Event Parameters</p>
          <div className="space-y-2">
            {[
              { label: 'Primary Mass m₁', value: `${selectedEvent.m1} M☉`, color: '#00d4ff' },
              { label: 'Secondary Mass m₂', value: `${selectedEvent.m2} M☉`, color: '#00d4ff' },
              { label: 'Luminosity Distance', value: `${selectedEvent.dL} Mpc`, color: '#7c3aed' },
              { label: 'SNR', value: selectedEvent.snr.toString(), color: '#00ff88' },
              { label: 'Chirp Mass', value: selectedEvent.chirpMass ? `${selectedEvent.chirpMass} M☉` : 'N/A', color: '#ffd700' },
              { label: 'χ_eff', value: selectedEvent.spinEff !== undefined ? selectedEvent.spinEff.toFixed(3) : 'N/A', color: '#ff4466' },
            ].map((item) => (
              <div key={item.label} className="flex justify-between text-xs">
                <span className="text-slate-500">{item.label}</span>
                <span className="font-mono" style={{ color: item.color }}>{item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="card p-4">
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Visualization Controls</p>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-500">SNR Override</span>
                <span className="text-cyan-400 mono">{snr}</span>
              </div>
              <input type="range" min={8} max={50} value={snr} onChange={e => setSnr(+e.target.value)}
                className="w-full accent-cyan-400" />
            </div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={showBSM} onChange={e => setShowBSM(e.target.checked)} className="accent-purple-400" />
              <span className="text-xs text-slate-400">Show BSM Signature</span>
              <span className="px-1 py-0.5 rounded text-xs" style={{ background: '#7c3aed15', color: '#7c3aed', border: '1px solid #7c3aed30' }}>L5-L7</span>
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setPlaying(p => !p)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
                style={{ background: playing ? 'rgba(255,68,102,0.2)' : 'rgba(0,212,255,0.2)', color: playing ? '#ff4466' : '#00d4ff', border: `1px solid ${playing ? '#ff446640' : '#00d4ff40'}` }}
              >
                {playing ? <Pause size={12} /> : <Play size={12} />}
                {playing ? 'Pause' : 'Animate'}
              </button>
              <button
                onClick={() => setData(generateWaveform(selectedEvent.m1, selectedEvent.m2, selectedEvent.dL, snr, showBSM))}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
                style={{ background: 'rgba(124,58,237,0.2)', color: '#7c3aed', border: '1px solid #7c3aed40' }}
              >
                <RefreshCw size={12} />
                Regenerate
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Phase indicator */}
      <div className="flex gap-4">
        {(['inspiral', 'merger', 'ringdown'] as const).map((p) => (
          <div key={p} className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full" style={{ background: phaseColors[p], opacity: playing && phase === p ? 1 : 0.3 }} />
            <span className="capitalize" style={{ color: playing && phase === p ? phaseColors[p] : '#6b7280' }}>{p}</span>
          </div>
        ))}
        <div className="ml-auto text-xs text-slate-500 mono">
          h(t) × 10<sup>-21</sup> [m/m] · t [s] relative to merger
        </div>
      </div>

      {/* Main waveform chart */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="card p-5"
      >
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 5, right: 20, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="t" stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }}>
              <Label value="Time relative to merger (s)" offset={-10} position="insideBottom" fill="#6b7280" fontSize={11} />
            </XAxis>
            <YAxis stroke="#ffffff20" tick={{ fontSize: 10, fill: '#6b7280' }}>
              <Label value="Strain h(t)" angle={-90} position="insideLeft" fill="#6b7280" fontSize={11} />
            </YAxis>
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine x={0} stroke="#ffd70060" strokeDasharray="4 4" label={{ value: 'Merger', fill: '#ffd700', fontSize: 10, position: 'top' }} />
            <Line type="monotone" dataKey="strainGR" name="GR (QNIM)" stroke="#00d4ff" strokeWidth={1.5} dot={false} />
            {showBSM && <Line type="monotone" dataKey="strainBSM" name="BSM Signal" stroke="#7c3aed" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />}
          </LineChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Frequency domain hint */}
      <div className="card p-4 text-xs text-slate-500 font-mono">
        <span className="text-cyan-400">h̃(f)</span> = ∫ h(t) e<sup>-2πift</sup> dt — Fourier transform used for matched filtering in QNIM's quantum feature map.
        <span className="ml-4 text-purple-400">f_ISCO</span> = c³/(6√6 π G M) ≈{' '}
        <span className="text-fuchsia-400">{(299792458 ** 3 / (6 * Math.sqrt(6) * Math.PI * 6.674e-11 * (selectedEvent.m1 + selectedEvent.m2) * 1.989e30)).toFixed(1)} Hz</span>
      </div>
    </div>
  );
}
