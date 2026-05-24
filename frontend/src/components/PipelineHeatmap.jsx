/**
 * PipelineHeatmap.jsx
 * -------------------
 * Grid of all pipeline loans coloured by close probability.
 * Hover → tooltip  |  Click → loan detail sidebar with SHAP factors + action.
 *
 * Props:
 *   loans        {Array}    — full loan array (see data shape below)
 *   onBadgeClick {Function} — optional callback when user clicks a cell
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';

// ─── constants ───────────────────────────────────────────────────────────────

const STAGES = [
  'application',
  'processing',
  'underwriting',
  'cond_approval',
  'clear_to_close',
];

const STAGE_LABELS = {
  application:    'Application',
  processing:     'Processing',
  underwriting:   'Underwriting',
  cond_approval:  'Cond. Approval',
  clear_to_close: 'Clear to Close',
};

const STAGE_ICONS = {
  application:    '📋',
  processing:     '⚙️',
  underwriting:   '🔍',
  cond_approval:  '✅',
  clear_to_close: '🏁',
};

const OWNER_META = {
  processor:  { icon: '📋', label: 'Processor',  color: '#0ea5e9' },
  lo:         { icon: '📞', label: 'Loan Officer', color: '#8b5cf6' },
  closer:     { icon: '🖊️', label: 'Closer',      color: '#f59e0b' },
  management: { icon: '🚨', label: 'Management',  color: '#ef4444' },
};

const PRIORITY_META = {
  urgent: { color: '#ef4444', bg: '#fee2e2', label: 'URGENT' },
  high:   { color: '#f97316', bg: '#ffedd5', label: 'HIGH'   },
  normal: { color: '#3b82f6', bg: '#dbeafe', label: 'NORMAL' },
};

// ─── colour helpers ───────────────────────────────────────────────────────────

const probColors = (p) => {
  if (p >= 0.80) return { bg: '#dcfce7', border: '#16a34a', text: '#15803d', bar: '#16a34a', glow: 'rgba(22,163,74,0.15)'  };
  if (p >= 0.60) return { bg: '#fef9c3', border: '#eab308', text: '#713f12', bar: '#eab308', glow: 'rgba(234,179,8,0.15)' };
  if (p >= 0.40) return { bg: '#ffedd5', border: '#f97316', text: '#9a3412', bar: '#f97316', glow: 'rgba(249,115,22,0.15)' };
  return           { bg: '#fee2e2', border: '#ef4444', text: '#991b1b', bar: '#ef4444', glow: 'rgba(239,68,68,0.15)'  };
};

const fmtPct  = (v) => `${Math.round(v * 100)}%`;
const fmtAmt  = (v) => v >= 1_000_000 ? `$${(v / 1_000_000).toFixed(2)}M` : `$${(v / 1_000).toFixed(0)}K`;

// ─── sub-components ──────────────────────────────────────────────────────────

function Tooltip({ loan, pos }) {
  if (!loan) return null;
  const c = probColors(loan.close_probability);
  const style = {
    position: 'fixed',
    left: pos.x + 16,
    top:  pos.y - 10,
    zIndex: 9999,
    background: '#1e293b',
    border: `1px solid ${c.border}`,
    borderRadius: 10,
    padding: '12px 16px',
    minWidth: 240,
    boxShadow: `0 8px 24px rgba(0,0,0,.35), 0 0 0 1px ${c.border}`,
    pointerEvents: 'none',
    color: '#f1f5f9',
    fontSize: 13,
  };
  // keep tooltip on screen
  if (pos.x + 260 > window.innerWidth)  style.left = pos.x - 256;
  if (pos.y + 200 > window.innerHeight) style.top  = pos.y - 180;

  return (
    <div style={style}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:8 }}>
        <span style={{ fontWeight:700, fontSize:14, color:'#f8fafc' }}>{loan.loan_id}</span>
        <span style={{
          background: c.border, color:'#fff', fontWeight:700, fontSize:11,
          padding:'2px 8px', borderRadius:20,
        }}>{fmtPct(loan.close_probability)}</span>
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'4px 12px', fontSize:12, color:'#94a3b8' }}>
        <span>👤 {loan.borrower_initials}</span>
        <span>💰 {fmtAmt(loan.loan_amount)}</span>
        <span>📍 {STAGE_LABELS[loan.pipeline_stage]}</span>
        <span>📅 {loan.days_to_projected_close}d to close</span>
        <span>🔒 Lock: {loan.rate_lock_expiry_days}d</span>
        <span>⚠️ {loan.condition_count} condition{loan.condition_count !== 1 ? 's' : ''}</span>
      </div>
      {loan.risk_factors?.length > 0 && (
        <div style={{ marginTop:8, borderTop:'1px solid #334155', paddingTop:8 }}>
          <div style={{ fontSize:11, color:'#64748b', marginBottom:4, textTransform:'uppercase', letterSpacing:'.5px' }}>Top Risk Factor</div>
          <div style={{ fontSize:12, color:'#fbbf24' }}>▼ {loan.risk_factors[0]?.plain_label}</div>
        </div>
      )}
      <div style={{ marginTop:8, fontSize:11, color:'#64748b', textAlign:'center' }}>
        Click for full analysis →
      </div>
    </div>
  );
}

function LoanCell({ loan, onClick, onMouseMove, onMouseLeave }) {
  const c = probColors(loan.close_probability);
  const isExpiring = loan.rate_lock_expiry_days <= 7;
  const [hovered, setHovered] = useState(false);

  return (
    <div
      title=""
      onClick={() => onClick(loan)}
      onMouseMove={onMouseMove}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={(e) => { setHovered(false); onMouseLeave(e); }}
      style={{
        background:     hovered ? c.bg : '#fff',
        border:        `1px solid ${hovered ? c.border : '#e2e8f0'}`,
        borderLeft:    `4px solid ${c.border}`,
        borderRadius:   8,
        padding:        '8px 10px',
        cursor:         'pointer',
        transition:     'all .15s ease',
        boxShadow:      hovered ? `0 4px 16px ${c.glow}, 0 1px 4px rgba(0,0,0,.08)` : '0 1px 3px rgba(0,0,0,.06)',
        transform:      hovered ? 'translateY(-2px)' : 'none',
        position:       'relative',
        userSelect:     'none',
        minWidth:       0,
      }}
    >
      {/* Lock expiry warning pip */}
      {isExpiring && (
        <div style={{
          position:'absolute', top:6, right:6,
          width:7, height:7, borderRadius:'50%',
          background:'#ef4444',
          boxShadow:'0 0 0 2px #fee2e2',
          animation:'pulse 1.5s infinite',
        }} title="Rate lock expiring soon!" />
      )}

      {/* Loan ID */}
      <div style={{ fontSize:11, fontWeight:700, color:'#374151', fontFamily:'monospace', letterSpacing:'.3px' }}>
        {loan.loan_id}
      </div>

      {/* Borrower + loan type */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginTop:3 }}>
        <span style={{ fontSize:12, color:'#6b7280', fontWeight:600 }}>{loan.borrower_initials}</span>
        <span style={{ fontSize:10, color:'#9ca3af', background:'#f3f4f6', padding:'1px 5px', borderRadius:4 }}>
          {loan.loan_type}
        </span>
      </div>

      {/* Probability bar */}
      <div style={{ marginTop:6 }}>
        <div style={{ display:'flex', justifyContent:'space-between', marginBottom:3 }}>
          <span style={{ fontSize:10, color:'#9ca3af' }}>{loan.days_to_projected_close}d left</span>
          <span style={{ fontSize:12, fontWeight:700, color: c.text }}>{fmtPct(loan.close_probability)}</span>
        </div>
        <div style={{ height:3, background:'#f3f4f6', borderRadius:4, overflow:'hidden' }}>
          <div style={{
            height:'100%', width:`${loan.close_probability * 100}%`,
            background: c.bar, borderRadius:4,
            transition:'width .3s ease',
          }} />
        </div>
      </div>
    </div>
  );
}

// ─── SHAP factor bar in sidebar ──────────────────────────────────────────────

function ShapBar({ factor, isRisk }) {
  const impact = Math.abs(factor.impact);
  const maxBar = 0.25; // clip visual at 0.25 for readability
  const pct = Math.min(impact / maxBar, 1) * 100;
  const color = isRisk ? '#ef4444' : '#16a34a';

  return (
    <div style={{ marginBottom:12 }}>
      <div style={{ display:'flex', justifyContent:'space-between', fontSize:12, marginBottom:4 }}>
        <span style={{ color:'#374151', fontWeight:600 }}>{factor.factor}</span>
        <span style={{ color, fontWeight:700 }}>
          {factor.impact < 0 ? '▼' : '▲'} {Math.abs(factor.impact).toFixed(3)}
        </span>
      </div>
      <div style={{ height:6, background:'#f3f4f6', borderRadius:4, overflow:'hidden' }}>
        <div style={{ height:'100%', width:`${pct}%`, background:color, borderRadius:4, transition:'width .4s ease' }} />
      </div>
      <div style={{ fontSize:11, color:'#6b7280', marginTop:4, lineHeight:1.4 }}>
        {factor.plain_label}
      </div>
    </div>
  );
}

// ─── Loan detail sidebar ─────────────────────────────────────────────────────

function LoanSidebar({ loan, onClose }) {
  const c      = loan ? probColors(loan.close_probability) : {};
  const action = loan?.action;
  const pri    = action ? PRIORITY_META[action.priority] ?? PRIORITY_META.normal : null;
  const owner  = action ? OWNER_META[action.owner] ?? OWNER_META.processor : null;

  const handleSend = () => {
    if (!action) return;
    const subject = encodeURIComponent(`[Pipeline Action] ${loan.loan_id} — ${action.action.slice(0, 60)}`);
    const body    = encodeURIComponent(
      `Action: ${action.action}\n\nLoan: ${loan.loan_id}\nStage: ${STAGE_LABELS[loan.pipeline_stage]}\n` +
      `Close probability: ${fmtPct(loan.close_probability)}\nDeadline: ${action.deadline}\n\n` +
      `Risk factors:\n${loan.risk_factors?.map(f => `• ${f.plain_label}`).join('\n') ?? ''}`
    );
    window.open(`mailto:?subject=${subject}&body=${body}`);
  };

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position:'fixed', inset:0, background:'rgba(15,23,42,.4)',
          zIndex:200, backdropFilter:'blur(2px)',
          animation:'fadeIn .2s ease',
        }}
      />

      {/* Drawer */}
      <div style={{
        position:'fixed', top:0, right:0, bottom:0, width:420,
        background:'#fff', boxShadow:'-8px 0 40px rgba(0,0,0,.18)',
        zIndex:201, overflowY:'auto', display:'flex', flexDirection:'column',
        animation:'slideIn .25s cubic-bezier(.4,0,.2,1)',
      }}>
        {/* Header */}
        <div style={{ background:'#1e293b', padding:'20px 24px', color:'#fff', flexShrink:0 }}>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start' }}>
            <div>
              <div style={{ fontSize:11, color:'#64748b', textTransform:'uppercase', letterSpacing:'.8px', marginBottom:4 }}>
                {STAGE_ICONS[loan.pipeline_stage]} {STAGE_LABELS[loan.pipeline_stage]}
              </div>
              <div style={{ fontSize:22, fontWeight:700 }}>{loan.loan_id}</div>
              <div style={{ fontSize:14, color:'#94a3b8', marginTop:2 }}>
                {loan.borrower_initials} &nbsp;·&nbsp; {loan.loan_type} &nbsp;·&nbsp; {fmtAmt(loan.loan_amount)}
              </div>
            </div>
            <button onClick={onClose} style={{
              background:'transparent', border:'none', color:'#94a3b8',
              fontSize:22, cursor:'pointer', padding:4, lineHeight:1,
            }}>✕</button>
          </div>

          {/* Probability gauge */}
          <div style={{ marginTop:16 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:6 }}>
              <span style={{ fontSize:12, color:'#94a3b8' }}>Close probability</span>
              <span style={{ fontSize:18, fontWeight:800, color: c.border }}>{fmtPct(loan.close_probability)}</span>
            </div>
            <div style={{ height:8, background:'#334155', borderRadius:6, overflow:'hidden' }}>
              <div style={{
                height:'100%', width:`${loan.close_probability * 100}%`,
                background:`linear-gradient(90deg, ${c.border}cc, ${c.border})`,
                borderRadius:6, transition:'width .5s ease',
              }} />
            </div>
            <div style={{ display:'flex', justifyContent:'space-between', fontSize:11, color:'#64748b', marginTop:4 }}>
              <span>Critical</span><span>At-Risk (65%)</span><span>Safe</span>
            </div>
          </div>
        </div>

        {/* Key metrics */}
        <div style={{ padding:'16px 24px', borderBottom:'1px solid #f1f5f9', background:'#f8fafc' }}>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12 }}>
            {[
              ['Days to Close',  `${loan.days_to_projected_close}d`],
              ['Rate Lock',      `${loan.rate_lock_expiry_days}d`],
              ['Conditions',     loan.condition_count],
              ['LTV',            `${loan.ltv?.toFixed(0)}%`],
              ['DTI',            `${loan.dti?.toFixed(0)}%`],
              ['Days in Stage',  `${loan.days_in_current_stage}d`],
            ].map(([label, val]) => (
              <div key={label} style={{ textAlign:'center' }}>
                <div style={{ fontSize:16, fontWeight:700, color:'#1e293b' }}>{val}</div>
                <div style={{ fontSize:10, color:'#94a3b8', textTransform:'uppercase', letterSpacing:'.5px' }}>{label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* SHAP risk factors */}
        <div style={{ padding:'20px 24px', flexGrow:1 }}>
          <div style={{ fontSize:13, fontWeight:700, color:'#374151', textTransform:'uppercase', letterSpacing:'.6px', marginBottom:16 }}>
            ⚠️ SHAP Risk Factors
          </div>
          {loan.risk_factors?.length > 0
            ? loan.risk_factors.map((f, i) => (
                <ShapBar key={i} factor={f} isRisk={f.impact < 0} />
              ))
            : <div style={{ fontSize:13, color:'#9ca3af' }}>No significant risk factors detected.</div>
          }
        </div>

        {/* Recommended action */}
        {action && (
          <div style={{ padding:'16px 24px', background:`${pri.bg}`, borderTop:`2px solid ${pri.color}`, flexShrink:0 }}>
            <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:10 }}>
              <span style={{
                background: pri.color, color:'#fff', fontSize:10, fontWeight:700,
                padding:'2px 8px', borderRadius:12, letterSpacing:'.5px',
              }}>{pri.label}</span>
              <span style={{ fontSize:12, color:'#6b7280' }}>
                {owner.icon} {owner.label} &nbsp;·&nbsp; Due: {action.deadline.replace('_',' ')}
              </span>
            </div>
            <div style={{ fontSize:13, color:'#1e293b', lineHeight:1.6, fontWeight:500, marginBottom:12 }}>
              {action.action}
            </div>
            <div style={{ display:'flex', gap:8 }}>
              <button
                onClick={handleSend}
                style={{
                  flex:1, padding:'9px 12px', background:'#1e293b', color:'#fff',
                  border:'none', borderRadius:8, fontSize:12, fontWeight:600,
                  cursor:'pointer', display:'flex', alignItems:'center', justifyContent:'center', gap:6,
                }}
              >
                ✉️ Send to {owner.label}
              </button>
              <button
                onClick={onClose}
                style={{
                  padding:'9px 14px', background:'#fff', color:'#374151',
                  border:'1px solid #e2e8f0', borderRadius:8, fontSize:12, cursor:'pointer',
                }}
              >
                Close
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

// ─── Legend ──────────────────────────────────────────────────────────────────

function Legend() {
  const items = [
    { label: '> 80% — On track',   color: '#16a34a' },
    { label: '60–80% — Monitor',   color: '#eab308' },
    { label: '40–60% — At risk',   color: '#f97316' },
    { label: '< 40% — Critical',   color: '#ef4444' },
    { label: 'Lock ≤ 7 days',      color: '#ef4444', dot: true },
  ];
  return (
    <div style={{ display:'flex', gap:16, alignItems:'center', flexWrap:'wrap' }}>
      {items.map(({ label, color, dot }) => (
        <div key={label} style={{ display:'flex', alignItems:'center', gap:6, fontSize:12, color:'#6b7280' }}>
          {dot
            ? <div style={{ width:8, height:8, borderRadius:'50%', background:color, boxShadow:`0 0 0 2px ${color}44` }} />
            : <div style={{ width:12, height:12, borderRadius:3, background:color }} />
          }
          {label}
        </div>
      ))}
    </div>
  );
}

// ─── Main export ─────────────────────────────────────────────────────────────

export default function PipelineHeatmap({ loans = [] }) {
  const [tooltip, setTooltip]       = useState({ loan: null, pos: { x: 0, y: 0 } });
  const [selectedLoan, setSelected] = useState(null);

  const handleMouseMove = useCallback((loan, e) => {
    setTooltip({ loan, pos: { x: e.clientX, y: e.clientY } });
  }, []);

  const handleMouseLeave = useCallback(() => {
    setTooltip({ loan: null, pos: { x: 0, y: 0 } });
  }, []);

  const handleClick = useCallback((loan) => {
    setSelected(loan);
    setTooltip({ loan: null, pos: { x: 0, y: 0 } });
  }, []);

  // Group loans by stage, worst probability first within each group
  const byStage = STAGES.reduce((acc, s) => {
    acc[s] = loans
      .filter(l => l.pipeline_stage === s)
      .sort((a, b) => a.close_probability - b.close_probability);
    return acc;
  }, {});

  return (
    <div style={{ position:'relative' }}>
      {/* Keyframe animations (injected once) */}
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
        @keyframes fadeIn { from{opacity:0} to{opacity:1} }
        @keyframes slideIn { from{transform:translateX(100%)} to{transform:translateX(0)} }
      `}</style>

      {/* Legend + count */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:16, flexWrap:'wrap', gap:8 }}>
        <Legend />
        <span style={{ fontSize:12, color:'#9ca3af' }}>
          {loans.length} loans total · {loans.filter(l => l.close_probability < 0.65).length} at-risk
        </span>
      </div>

      {/* Stage columns */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(5, 1fr)', gap:12 }}>
        {STAGES.map(stage => {
          const stageLoanns = byStage[stage] ?? [];
          const atRisk = stageLoanns.filter(l => l.close_probability < 0.65).length;
          return (
            <div key={stage}>
              {/* Stage header */}
              <div style={{
                background:'#f8fafc', border:'1px solid #e2e8f0',
                borderRadius:'8px 8px 0 0', padding:'10px 12px',
                marginBottom:4,
              }}>
                <div style={{ fontSize:13, fontWeight:700, color:'#374151' }}>
                  {STAGE_ICONS[stage]} {STAGE_LABELS[stage]}
                </div>
                <div style={{ display:'flex', gap:8, marginTop:4, fontSize:11 }}>
                  <span style={{ color:'#6b7280' }}>{stageLoanns.length} loans</span>
                  {atRisk > 0 && (
                    <span style={{ color:'#ef4444', fontWeight:600 }}>⚠ {atRisk} at-risk</span>
                  )}
                </div>
              </div>

              {/* Loan cells */}
              <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
                {stageLoanns.length === 0 ? (
                  <div style={{
                    border:'1px dashed #e2e8f0', borderRadius:8,
                    padding:'24px 12px', textAlign:'center',
                    fontSize:12, color:'#d1d5db',
                  }}>
                    No loans
                  </div>
                ) : (
                  stageLoanns.map(loan => (
                    <LoanCell
                      key={loan.loan_id}
                      loan={loan}
                      onClick={handleClick}
                      onMouseMove={(e) => handleMouseMove(loan, e)}
                      onMouseLeave={handleMouseLeave}
                    />
                  ))
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Floating tooltip */}
      <Tooltip loan={tooltip.loan} pos={tooltip.pos} />

      {/* Loan detail sidebar */}
      {selectedLoan && (
        <LoanSidebar loan={selectedLoan} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}
