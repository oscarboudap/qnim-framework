import { motion } from 'framer-motion';

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  badge?: string;
}

export default function SectionHeader({ title, subtitle, badge }: SectionHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4 }}
      className="mb-6"
    >
      <div className="flex items-center gap-3 mb-1">
        <h2 className="text-xl font-semibold text-white">{title}</h2>
        {badge && (
          <span className="px-2 py-0.5 rounded text-xs font-mono"
            style={{ background: 'rgba(0,212,255,0.1)', color: '#00d4ff', border: '1px solid rgba(0,212,255,0.3)' }}>
            {badge}
          </span>
        )}
      </div>
      {subtitle && <p className="text-sm text-slate-500">{subtitle}</p>}
    </motion.div>
  );
}
