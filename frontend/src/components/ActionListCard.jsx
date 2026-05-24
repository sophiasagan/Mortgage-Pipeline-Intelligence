/**
 * ActionListCard.jsx
 * ------------------
 * Prioritised daily action list for at-risk mortgage loans.
 *
 * Features:
 *   • Filter bar — All | Urgent | High | Normal
 *   • Priority badge (red / amber / blue) + owner icon + deadline chip
 *   • "Mark complete" toggle per action (optimistic UI + optional API call)
 *   • "Send to [owner]" — pre-filled email or Teams deep-link
 *   • Compact summary bar (counts by priority)
 *
 * Props:
 *   actions        {Array}     — sorted action list from generate_daily_actions()
 *   onComplete     {Function}  — optional callback(action_id, completed: bool)
 *   apiBaseUrl     {string}    — optional base URL for complete PATCH call
 */

import React, { useState, useMemo, useCallback } from 'react';

// ─── constants ───────────────────────────────────────────────────────────────

const PRIORITY = {
  urgent: {
    color:  '#ef4444',
    bg:     '#fee2e2',
    ring:   '#fca5a5',
    label:  'URGENT',
    icon:   '🔴',
    order:  0,
  },
  high: {
    color:  '#f97316',
    bg:     '#ffedd5',
    ring:   '#fdba74',
    label:  'HIGH',
    icon:   '🟡',
    order:  1,
  },
  normal: {
    color:  '#3b82f6',
    bg:     '#dbeafe',
    ring:   '#93c5fd',
    label:  'NORMAL',
    icon:   '🔵',
    order:  2,
  },
};

const DEADLINE = {
  today:     { label: 'Today',     color: '#ef4444', bg: '#fee2e2' },
  tomorrow:  { label: 'Tomorrow',  color: '#f97316', bg: '#ffedd5' },
  this_week: { label: 'This Week', color: '#3b82f6', bg: '#dbeafe' },
};

const OWNER = {
  processor:  { icon: '📋', label: 'Processor',   email: 'processor@cu-mortgage.internal' },
  lo:         { icon: '📞', label: 'Loan Officer', email: 'lo@cu-mortgage.internal' },
  closer:     { icon: '🖊️', label: 'Closer',       email: 'closer@cu-mortgage.internal' },
  management: { icon: '🚨', label: 'Management',   email: 'management@cu-mortgage.internal' },
};

const STAGE_LABELS = {
  application:    'Application',
  processing:     'Processing',
  underwriting:   'Underwriting',
  cond_approval:  'Cond. Approval',
  clear_to_close: 'Clear to Close',
};

const fmtPct = (v) => `${Math.round(v * 100)}%`;

// ─── ActionItem ──────────────────────────────────────────────────────────────

function ActionItem({ action, completed, onToggleComplete, onSend }) {
  const [expanded, setExpanded] = useState(false);
  const [sendConfirm, setSendConfirm] = useState(false);

  const pri     = PRIORITY[action.priority]  ?? PRIORITY.normal;
  const dl      = DEADLINE[action.deadline]  ?? DEADLINE.this_week;
  const owner   = OWNER[action.owner]        ?? OWNER.processor;

  const handleSend = () => {
    if (!sendConfirm) { setSendConfirm(true); return; }
    setSendConfirm(false);
    onSend(action);
  };

  return (
    <div style={{
      background: completed ? '#f8fafc' : '#fff',
      border:     `1px solid ${completed ? '#e2e8f0' : pri.ring}`,
      borderLeft: `4px solid ${completed ? '#d1d5db' : pri.color}`,
      borderRadius: 10,
      padding: '14px 16px',
      opacity: completed ? 0.55 : 1,
      transition: 'all .2s ease',
      marginBottom: 0,
    }}>
      {/* Row 1: badge + action text + complete toggle */}
      <div style={{ display:'flex', alignItems:'flex-start', gap:10 }}>

        {/* Priority badge */}
        <div style={{
          background: completed ? '#e5e7eb' : pri.bg,
          color:      completed ? '#9ca3af' : pri.color,
          fontSize:   10, fontWeight:800, letterSpacing:'.6px',
          padding:    '3px 8px', borderRadius:12, whiteSpace:'nowrap', flexShrink:0,
          marginTop: 1,
        }}>
          {pri.icon} {pri.label}
        </div>

        {/* Action text */}
        <div
          style={{ flex:1, fontSize:13, color: completed ? '#9ca3af' : '#1e293b',
                   lineHeight:1.5, cursor:'pointer', fontWeight:500,
                   textDecoration: completed ? 'line-through' : 'none' }}
          onClick={() => setExpanded(e => !e)}
        >
          {action.action}
        </div>

        {/* Complete toggle */}
        <button
          onClick={() => onToggleComplete(action)}
          title={completed ? 'Mark incomplete' : 'Mark complete'}
          style={{
            width:22, height:22, borderRadius:6, flexShrink:0,
            border:     `2px solid ${completed ? '#16a34a' : '#d1d5db'}`,
            background: completed ? '#16a34a' : '#fff',
            color:      '#fff', fontSize:13, display:'flex',
            alignItems:'center', justifyContent:'center',
            cursor:'pointer', transition:'all .15s ease', padding:0,
          }}
        >
          {completed ? '✓' : ''}
        </button>
      </div>

      {/* Row 2: owner | deadline | stage | probability */}
      <div style={{ display:'flex', gap:10, marginTop:10, flexWrap:'wrap', alignItems:'center' }}>
        <span style={{ fontSize:12, color:'#6b7280', display:'flex', alignItems:'center', gap:4 }}>
          {owner.icon} {owner.label}
        </span>

        <span style={{
          fontSize:11, fontWeight:600, padding:'2px 8px', borderRadius:10,
          background: dl.bg, color: dl.color,
        }}>
          ⏰ {dl.label}
        </span>

        <span style={{ fontSize:11, color:'#9ca3af' }}>
          📍 {STAGE_LABELS[action.pipeline_stage] ?? action.pipeline_stage}
        </span>

        <span style={{
          fontSize:11, fontWeight:700, marginLeft:'auto',
          color: action.close_probability < 0.40 ? '#ef4444'
               : action.close_probability < 0.55 ? '#f97316'
               : '#6b7280',
        }}>
          {fmtPct(action.close_probability)} close prob
        </span>
      </div>

      {/* Expanded: risk factors */}
      {expanded && action.risk_factors?.length > 0 && (
        <div style={{
          marginTop:12, background:'#f8fafc', borderRadius:8,
          padding:'10px 12px', borderTop:'1px solid #f1f5f9',
        }}>
          <div style={{ fontSize:11, fontWeight:700, color:'#6b7280', marginBottom:6,
                        textTransform:'uppercase', letterSpacing:'.5px' }}>
            Risk Factors
          </div>
          {action.risk_factors.map((rf, i) => (
            <div key={i} style={{ fontSize:12, color:'#374151', marginBottom:4, paddingLeft:8,
                                   borderLeft:'2px solid #fbbf24', lineHeight:1.4 }}>
              {rf}
            </div>
          ))}
        </div>
      )}

      {/* Row 3: Send button (shown when not completed) */}
      {!completed && (
        <div style={{ marginTop:12, display:'flex', gap:8 }}>
          <button
            onClick={() => setExpanded(e => !e)}
            style={{
              padding:'6px 12px', background:'transparent', color:'#6b7280',
              border:'1px solid #e2e8f0', borderRadius:7, fontSize:12, cursor:'pointer',
            }}
          >
            {expanded ? '▲ Less' : '▼ Risk details'}
          </button>

          <button
            onClick={handleSend}
            onBlur={() => setSendConfirm(false)}
            style={{
              padding:'6px 14px',
              background: sendConfirm ? pri.color : '#f8fafc',
              color:       sendConfirm ? '#fff'    : '#374151',
              border:     `1px solid ${sendConfirm ? pri.color : '#e2e8f0'}`,
              borderRadius:7, fontSize:12, cursor:'pointer',
              transition:'all .15s ease', fontWeight: sendConfirm ? 700 : 400,
              display:'flex', alignItems:'center', gap:5,
            }}
          >
            {sendConfirm ? `✉️ Confirm send to ${owner.label}` : `↗ Send to ${owner.label}`}
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Summary bar ─────────────────────────────────────────────────────────────

function SummaryBar({ actions, completed }) {
  const urgentTodo = actions.filter(a => a.priority === 'urgent' && !completed.has(a.loan_id)).length;
  const highTodo   = actions.filter(a => a.priority === 'high'   && !completed.has(a.loan_id)).length;
  const totalDone  = actions.filter(a => completed.has(a.loan_id)).length;
  const pct        = actions.length > 0 ? Math.round((totalDone / actions.length) * 100) : 0;

  return (
    <div style={{ display:'flex', gap:0, marginBottom:16, borderRadius:10, overflow:'hidden',
                  border:'1px solid #e2e8f0', background:'#fff' }}>
      {[
        { label:'Urgent',   value:urgentTodo, color:'#ef4444' },
        { label:'High',     value:highTodo,   color:'#f97316' },
        { label:'Done',     value:`${pct}%`,  color:'#16a34a' },
        { label:'Total',    value:actions.length, color:'#64748b' },
      ].map(({ label, value, color }, i) => (
        <div key={label} style={{
          flex:1, padding:'12px 8px', textAlign:'center',
          borderRight: i < 3 ? '1px solid #f1f5f9' : 'none',
        }}>
          <div style={{ fontSize:20, fontWeight:800, color }}>{value}</div>
          <div style={{ fontSize:11, color:'#9ca3af', textTransform:'uppercase', letterSpacing:'.5px' }}>{label}</div>
        </div>
      ))}
    </div>
  );
}

// ─── Filter tab bar ───────────────────────────────────────────────────────────

function FilterBar({ active, onChange, counts }) {
  const tabs = [
    { key:'all',    label:'All',    count: counts.all    },
    { key:'urgent', label:'🔴 Urgent', count: counts.urgent },
    { key:'high',   label:'🟡 High',   count: counts.high   },
    { key:'normal', label:'🔵 Normal', count: counts.normal },
  ];

  return (
    <div style={{ display:'flex', gap:4, marginBottom:14, background:'#f8fafc',
                  borderRadius:8, padding:3 }}>
      {tabs.map(({ key, label, count }) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          style={{
            flex:1, padding:'7px 4px', border:'none', borderRadius:6,
            fontSize:12, fontWeight: active === key ? 700 : 500,
            cursor:'pointer', transition:'all .15s ease',
            background: active === key ? '#fff' : 'transparent',
            color:       active === key ? '#1e293b' : '#6b7280',
            boxShadow:   active === key ? '0 1px 4px rgba(0,0,0,.1)' : 'none',
          }}
        >
          {label}
          <span style={{
            marginLeft:5, fontSize:10, fontWeight:700, padding:'1px 5px',
            borderRadius:10, background: active === key ? '#f1f5f9' : 'transparent',
          }}>
            {count}
          </span>
        </button>
      ))}
    </div>
  );
}

// ─── Main export ─────────────────────────────────────────────────────────────

export default function ActionListCard({ actions = [], onComplete, apiBaseUrl = '/api' }) {
  const [filter, setFilter]       = useState('all');
  const [completed, setCompleted] = useState(new Set());

  // Count by priority (total, not filtered by completion)
  const counts = useMemo(() => ({
    all:    actions.length,
    urgent: actions.filter(a => a.priority === 'urgent').length,
    high:   actions.filter(a => a.priority === 'high'  ).length,
    normal: actions.filter(a => a.priority === 'normal' ).length,
  }), [actions]);

  // Filtered + sorted (already sorted by backend, but re-sort after filter)
  const visible = useMemo(() => {
    const list = filter === 'all' ? actions : actions.filter(a => a.priority === filter);
    // Push completed to bottom within each group
    return [...list].sort((a, b) => {
      const aDone = completed.has(a.loan_id) ? 1 : 0;
      const bDone = completed.has(b.loan_id) ? 1 : 0;
      return aDone - bDone;
    });
  }, [actions, filter, completed]);

  const handleToggle = useCallback(async (action) => {
    const id       = action.loan_id;
    const nowDone  = !completed.has(id);

    // Optimistic UI
    setCompleted(prev => {
      const next = new Set(prev);
      nowDone ? next.add(id) : next.delete(id);
      return next;
    });

    // Notify parent
    onComplete?.(id, nowDone);

    // Optional API call
    try {
      await fetch(`${apiBaseUrl}/actions/${id}/complete`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ completed: nowDone }),
      });
    } catch {
      // Non-fatal — optimistic state stays
    }
  }, [completed, onComplete, apiBaseUrl]);

  const handleSend = useCallback((action) => {
    const owner  = OWNER[action.owner] ?? OWNER.processor;
    const dl     = DEADLINE[action.deadline]?.label ?? action.deadline;
    const stage  = STAGE_LABELS[action.pipeline_stage] ?? action.pipeline_stage;

    const subject = encodeURIComponent(
      `[Mortgage Pipeline] Action Required — ${stage} · Due: ${dl}`
    );
    const body = encodeURIComponent(
      `Hi ${owner.label},\n\n` +
      `Please take the following action on loan ${action.anon_id ?? action.loan_id}:\n\n` +
      `ACTION: ${action.action}\n\n` +
      `Priority: ${action.priority.toUpperCase()}\n` +
      `Deadline: ${dl}\n` +
      `Stage: ${stage}\n` +
      `Close probability: ${fmtPct(action.close_probability)}\n\n` +
      (action.risk_factors?.length
        ? `Risk factors:\n${action.risk_factors.map(f => `• ${f}`).join('\n')}\n\n`
        : '') +
      `— Mortgage Pipeline Intelligence (automated)`
    );

    window.open(`mailto:${owner.email}?subject=${subject}&body=${body}`);
  }, []);

  const allDone = visible.length > 0 && visible.every(a => completed.has(a.loan_id));

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100%' }}>
      {/* Card header */}
      <div style={{
        display:'flex', justifyContent:'space-between', alignItems:'center',
        marginBottom:14,
      }}>
        <div>
          <h2 style={{ margin:0, fontSize:16, fontWeight:700, color:'#1e293b' }}>
            📋 Daily Actions
          </h2>
          <div style={{ fontSize:12, color:'#9ca3af', marginTop:2 }}>
            {new Date().toLocaleDateString('en-US', { weekday:'long', month:'long', day:'numeric' })}
          </div>
        </div>
        {allDone && (
          <div style={{
            background:'#dcfce7', color:'#16a34a', fontWeight:700,
            fontSize:12, padding:'4px 12px', borderRadius:20,
          }}>
            ✓ All done!
          </div>
        )}
      </div>

      {/* Summary counts */}
      <SummaryBar actions={actions} completed={completed} />

      {/* Filter bar */}
      <FilterBar active={filter} onChange={setFilter} counts={counts} />

      {/* Action list */}
      {visible.length === 0 ? (
        <div style={{
          flex:1, display:'flex', flexDirection:'column',
          alignItems:'center', justifyContent:'center', padding:40,
          color:'#9ca3af', textAlign:'center',
        }}>
          <div style={{ fontSize:40, marginBottom:12 }}>✅</div>
          <div style={{ fontSize:15, fontWeight:600, color:'#374151' }}>
            {filter === 'all' ? 'No at-risk loans today!' : `No ${filter} actions`}
          </div>
          <div style={{ fontSize:13, marginTop:4 }}>
            {filter === 'all' ? 'Great work — pipeline looks healthy.' : `Try viewing all actions.`}
          </div>
        </div>
      ) : (
        <div style={{
          flex:1, overflowY:'auto', display:'flex', flexDirection:'column', gap:8,
          paddingRight:4,
          /* Custom scrollbar */
          scrollbarWidth:'thin',
          scrollbarColor:'#e2e8f0 transparent',
        }}>
          {visible.map(action => (
            <ActionItem
              key={action.loan_id}
              action={action}
              completed={completed.has(action.loan_id)}
              onToggleComplete={handleToggle}
              onSend={handleSend}
            />
          ))}

          {visible.length > 5 && (
            <div style={{ textAlign:'center', fontSize:12, color:'#9ca3af', padding:'8px 0' }}>
              {visible.length} actions · {completed.size} completed
            </div>
          )}
        </div>
      )}
    </div>
  );
}
