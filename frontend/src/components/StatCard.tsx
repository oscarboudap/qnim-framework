import { motion } from 'framer-motion';

interface StatCardProps {
  label: string;
  value: string;
  unit?: string;
  delta?: string;
  deltaPositive?: boolean;
  color?: string;
  icon?: React.ReactNode;
  delay?: number;
}

export default function StatCard({ label, value, unit, delta, deltaPositive, color = '#00d4ff', icon, delay = 0 }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="card card-hover p-5"
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
        {icon && (
          <div className="p-2 rounded-lg" style={{ background: `${color}15`, border: `1px solid ${color}30` }}>
            <div style={{ color }}>{icon}</div>
          </div>
        )}
      </div>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-bold" style={{ color }}>{value}</span>
        {unit && <span className="text-slate-500 text-sm mb-0.5">{unit}</span>}
      </div>
      {delta && (
        <div className={`mt-2 text-xs flex items-center gap-1 ${deltaPositive ? 'text-green-400' : 'text-red-400'}`}>
          <span>{deltaPositive ? '▲' : '▼'}</span>
          <span>{delta}</span>
        </div>
      )}
    </motion.div>
  );
}
