import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import WaveformVisualizer from './pages/WaveformVisualizer';
import TheoryClassifier from './pages/TheoryClassifier';
import QuantumAdvantage from './pages/QuantumAdvantage';
import GW150914Analysis from './pages/GW150914Analysis';
import LiveSimulation from './pages/LiveSimulation';
import MetricsHardware from './pages/MetricsHardware';
import VQCCircuit from './pages/VQCCircuit';

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen" style={{ background: 'var(--bg-deep)' }}>
        <div className="starfield" />
        <Sidebar />
        <main className="flex-1 ml-64 relative z-10">
          <div className="max-w-7xl mx-auto p-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/waveform" element={<WaveformVisualizer />} />
              <Route path="/classifier" element={<TheoryClassifier />} />
              <Route path="/quantum-advantage" element={<QuantumAdvantage />} />
              <Route path="/gw150914" element={<GW150914Analysis />} />
              <Route path="/simulation" element={<LiveSimulation />} />
              <Route path="/metrics" element={<MetricsHardware />} />
              <Route path="/circuit" element={<VQCCircuit />} />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;

