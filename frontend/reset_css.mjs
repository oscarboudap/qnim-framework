import { writeFileSync } from 'fs';

const css = `@import "tailwindcss";

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --quantum-blue: #00d4ff;
  --quantum-purple: #7c3aed;
  --quantum-green: #00ff88;
  --quantum-red: #ff4466;
  --quantum-gold: #ffd700;
  --bg-deep: #020510;
  --bg-card: #0a0f1e;
  --bg-card2: #0f1629;
  --border-glow: rgba(0, 212, 255, 0.2);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: 'Inter', sans-serif;
  background-color: var(--bg-deep);
  color: #e2e8f0;
  overflow-x: hidden;
}

code, pre, .mono { font-family: 'JetBrains Mono', monospace; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #020510; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff44; }

.glow-blue { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
.glow-purple { box-shadow: 0 0 20px rgba(124, 58, 237, 0.3); }
.glow-green { box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); }
.text-glow-blue { text-shadow: 0 0 20px rgba(0, 212, 255, 0.6); }
.text-glow-green { text-shadow: 0 0 20px rgba(0, 255, 136, 0.6); }

.starfield {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}

.starfield::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background-image: 
    radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.15) 0%, transparent 100%),
    radial-gradient(1px 1px at 80% 10%, rgba(255,255,255,0.1) 0%, transparent 100%),
    radial-gradient(1px 1px at 40% 70%, rgba(255,255,255,0.12) 0%, transparent 100%),
    radial-gradient(2px 2px at 10% 90%, rgba(0,212,255,0.1) 0%, transparent 100%),
    radial-gradient(2px 2px at 90% 80%, rgba(124,58,237,0.1) 0%, transparent 100%);
  animation: twinkle 8s ease-in-out infinite alternate;
}

@keyframes twinkle {
  0% { transform: translate(0, 0); }
  100% { transform: translate(-20px, -20px); }
}

.card {
  background: var(--bg-card);
  border: 1px solid var(--border-glow);
  border-radius: 12px;
}

.card-hover { transition: all 0.3s ease; }
.card-hover:hover {
  border-color: rgba(0, 212, 255, 0.5);
  box-shadow: 0 0 30px rgba(0, 212, 255, 0.15);
  transform: translateY(-2px);
}

@keyframes pulse-glow {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}
.pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }

@keyframes spin-slow {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
.spin-slow { animation: spin-slow 8s linear infinite; }

.gradient-text-blue {
  background: linear-gradient(135deg, #00d4ff, #7c3aed);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.gradient-text-green {
  background: linear-gradient(135deg, #00ff88, #00d4ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.sidebar-item {
  transition: all 0.2s ease;
  border-left: 3px solid transparent;
}
.sidebar-item:hover {
  border-left-color: rgba(0, 212, 255, 0.5);
  background: rgba(0, 212, 255, 0.05);
}
.sidebar-item.active {
  border-left-color: #00d4ff;
  background: rgba(0, 212, 255, 0.1);
}

.progress-bar {
  height: 6px;
  border-radius: 3px;
  background: #1e3a5f;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 1.2s ease;
}

.recharts-cartesian-grid-horizontal line,
.recharts-cartesian-grid-vertical line {
  stroke: rgba(255, 255, 255, 0.05) !important;
}
`;

writeFileSync('./src/index.css', css);
console.log('CSS written successfully');
