import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Activity, Brain, Zap, Atom,
  FlaskConical, BarChart3, Cpu, ChevronRight
} from 'lucide-react';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard', exact: true },
  { to: '/waveform', icon: Activity, label: 'GW Visualizer' },
  { to: '/classifier', icon: Brain, label: 'Theory Classifier' },
  { to: '/quantum-advantage', icon: Zap, label: 'Quantum Advantage' },
  { to: '/gw150914', icon: Atom, label: 'GW150914 Analysis' },
  { to: '/simulation', icon: FlaskConical, label: 'Live Simulation' },
  { to: '/metrics', icon: BarChart3, label: 'Metrics & Hardware' },
  { to: '/circuit', icon: Cpu, label: 'VQC Circuit' },
];

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-full w-64 flex flex-col z-50"
      style={{ background: 'rgba(10,15,30,0.95)', borderRight: '1px solid rgba(0,212,255,0.15)', backdropFilter: 'blur(20px)' }}>
      {/* Logo */}
      <div className="p-6 border-b" style={{ borderColor: 'rgba(0,212,255,0.1)' }}>
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-full flex items-center justify-center"
              style={{ background: 'linear-gradient(135deg, #00d4ff22, #7c3aed22)', border: '1px solid rgba(0,212,255,0.4)' }}>
              <Atom size={20} className="text-cyan-400" />
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-green-400 pulse-glow" />
          </div>
          <div>
            <div className="font-bold text-white text-sm tracking-wide">QNIM</div>
            <div className="text-xs" style={{ color: '#00d4ff88' }}>Quantum NeuroInspired Manifold</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4 px-2">
        <div className="text-xs font-semibold uppercase tracking-widest mb-3 px-3" style={{ color: '#ffffff33' }}>
          Navigation
        </div>
        {navItems.map(({ to, icon: Icon, label, exact }) => (
          <NavLink
            key={to}
            to={to}
            end={exact}
            className={({ isActive }) =>
              `sidebar-item flex items-center gap-3 px-3 py-2.5 rounded-lg mx-1 mb-1 cursor-pointer group ${isActive ? 'active' : ''}`
            }
          >
            {({ isActive }) => (
              <>
                <Icon size={16} className={isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-cyan-400'} style={{ transition: 'color 0.2s' }} />
                <span className={`text-sm flex-1 ${isActive ? 'text-cyan-300 font-medium' : 'text-slate-400 group-hover:text-slate-200'}`}>
                  {label}
                </span>
                {isActive && <ChevronRight size={12} className="text-cyan-400" />}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer status */}
      <div className="p-4 border-t" style={{ borderColor: 'rgba(0,212,255,0.1)' }}>
        <div className="card p-3 text-xs space-y-1.5">
          <div className="flex justify-between">
            <span className="text-slate-500">Status</span>
            <span className="text-green-400 font-medium">● Active</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Qubits</span>
            <span className="text-cyan-400 mono">12</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Backend</span>
            <span className="text-purple-400 mono">Aer Sim</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Accuracy</span>
            <span className="text-green-400 font-medium">91.0%</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
