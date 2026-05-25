/**
 * PipelineDashboard.jsx
 * ---------------------
 * Main intelligence dashboard for the CU mortgage pipeline.
 *
 * Layout:
 *   ┌─ Header ──────────────────────────────────────────────┐
 *   ├─ KPI row: $M | avg prob | at-risk | projected closings ┤
 *   ├─ Stage funnel (count, $vol, conversion %, avg-days)    ┤
 *   ├─ Pipeline Heatmap ────────┬── Daily Actions ───────────┤
 *   └───────────────────────────┴───────────────────────────┘
 *
 * Data: fetches from /api/pipeline-summary.  Falls back to MOCK_DATA
 *       when the API is unreachable (dev mode).
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import PipelineHeatmap from '../components/PipelineHeatmap';
import ActionListCard  from '../components/ActionListCard';

// ─── stage config ─────────────────────────────────────────────────────────────

const STAGES = ['application','processing','underwriting','cond_approval','clear_to_close'];

const STAGE_META = {
  application:    { label:'Application',    icon:'📋', benchmarkDays: 5,  color:'#6366f1' },
  processing:     { label:'Processing',     icon:'⚙️',  benchmarkDays:14,  color:'#0ea5e9' },
  underwriting:   { label:'Underwriting',   icon:'🔍', benchmarkDays:21,  color:'#f59e0b' },
  cond_approval:  { label:'Cond. Approval', icon:'✅', benchmarkDays:10,  color:'#10b981' },
  clear_to_close: { label:'Clear to Close', icon:'🏁', benchmarkDays: 3,  color:'#16a34a' },
};

// ─── mock data (dev fallback) ─────────────────────────────────────────────────

const MOCK_LOANS = [
  // ── application ─────────────────────────────────────────────────────────────
  { loan_id:'LN-1001', borrower_initials:'A.R.', loan_amount:312000, pipeline_stage:'application', days_in_current_stage:3, days_to_projected_close:52, close_probability:0.82, risk_level:'low',      loan_type:'conventional', ltv:76, dti:34, rate_lock_expiry_days:55, condition_count:1, risk_factors:[{ factor:'DTI',     impact:0.04, plain_label:'DTI of 34% — within guideline range' }],                                                                             action:{ priority:'normal', action:'Collect and verify 2-year employment history and W-2 documents from borrower.',           owner:'processor',  deadline:'this_week' } },
  { loan_id:'LN-1002', borrower_initials:'B.T.', loan_amount:480000, pipeline_stage:'application', days_in_current_stage:8, days_to_projected_close:47, close_probability:0.61, risk_level:'moderate', loan_type:'FHA',          ltv:89, dti:42, rate_lock_expiry_days:50, condition_count:3, risk_factors:[{ factor:'LTV',     impact:-0.09, plain_label:'LTV of 89% is high — PMI required' }],                                                                              action:{ priority:'high',   action:'Order FHA case number and confirm MIP calculations before underwriting submission.',     owner:'processor',  deadline:'tomorrow'  } },
  { loan_id:'LN-1003', borrower_initials:'C.M.', loan_amount:215000, pipeline_stage:'application', days_in_current_stage:2, days_to_projected_close:55, close_probability:0.88, risk_level:'low',      loan_type:'VA',           ltv:72, dti:31, rate_lock_expiry_days:58, condition_count:0, risk_factors:[],                                                                                                                                                               action:{ priority:'normal', action:'Complete VA COE verification and submit initial disclosures within 3 business days.',     owner:'lo',         deadline:'this_week' } },
  { loan_id:'LN-1004', borrower_initials:'D.K.', loan_amount:560000, pipeline_stage:'application', days_in_current_stage:11,days_to_projected_close:43, close_probability:0.44, risk_level:'high',     loan_type:'conventional', ltv:91, dti:46, rate_lock_expiry_days:46, condition_count:5, risk_factors:[{ factor:'DTI',     impact:-0.11, plain_label:'DTI of 46% is elevated — above typical guideline thresholds' },{ factor:'LTV', impact:-0.09, plain_label:'LTV of 91% is high — PMI required' }], action:{ priority:'high',   action:'Schedule borrower call to review DTI reduction options before file reaches underwriting.', owner:'lo',         deadline:'today'     } },

  // ── processing ──────────────────────────────────────────────────────────────
  { loan_id:'LN-2001', borrower_initials:'E.P.', loan_amount:395000, pipeline_stage:'processing', days_in_current_stage:9,  days_to_projected_close:38, close_probability:0.79, risk_level:'moderate', loan_type:'conventional', ltv:81, dti:38, rate_lock_expiry_days:42, condition_count:2, risk_factors:[{ factor:'Days in Stage', impact:-0.06, plain_label:'Loan has been in processing for 9 days — above average' }],                                                    action:{ priority:'normal', action:'Chase title company for preliminary title report — ordered 9 days ago with no update.',   owner:'processor',  deadline:'tomorrow'  } },
  { loan_id:'LN-2002', borrower_initials:'F.J.', loan_amount:628000, pipeline_stage:'processing', days_in_current_stage:18, days_to_projected_close:28, close_probability:0.52, risk_level:'high',     loan_type:'conventional', ltv:85, dti:44, rate_lock_expiry_days:31, condition_count:4, risk_factors:[{ factor:'Days Without Status Change', impact:-0.14, plain_label:'No status change in 18 days — file may be stalling' },{ factor:'Outstanding Conditions', impact:-0.10, plain_label:'4 outstanding conditions not yet cleared' }], action:{ priority:'high', action:'Escalate stalled processor file to supervisor — no movement in 18 days, 4 conditions open.', owner:'management', deadline:'today' } },
  { loan_id:'LN-2003', borrower_initials:'G.W.', loan_amount:272000, pipeline_stage:'processing', days_in_current_stage:6,  days_to_projected_close:41, close_probability:0.83, risk_level:'low',      loan_type:'FHA',          ltv:78, dti:36, rate_lock_expiry_days:45, condition_count:1, risk_factors:[],                                                                                                                                                               action:{ priority:'normal', action:'Confirm FHA appraisal appointment is scheduled with appraiser for this week.',            owner:'processor',  deadline:'this_week' } },
  { loan_id:'LN-2004', borrower_initials:'H.N.', loan_amount:445000, pipeline_stage:'processing', days_in_current_stage:23, days_to_projected_close:22, close_probability:0.37, risk_level:'critical', loan_type:'USDA',         ltv:92, dti:48, rate_lock_expiry_days:25, condition_count:7, risk_factors:[{ factor:'Income Docs Complete', impact:-0.16, plain_label:'Income documentation incomplete — required for underwriting sign-off' },{ factor:'Outstanding Conditions', impact:-0.14, plain_label:'7 outstanding conditions not yet cleared' },{ factor:'Days in Stage', impact:-0.12, plain_label:'Loan has been in processing for 23 days — above average' }], action:{ priority:'urgent', action:'Collect all 7 outstanding income documents immediately — USDA approval timeline at risk.', owner:'lo', deadline:'today' } },
  { loan_id:'LN-2005', borrower_initials:'I.C.', loan_amount:338000, pipeline_stage:'processing', days_in_current_stage:4,  days_to_projected_close:44, close_probability:0.91, risk_level:'low',      loan_type:'VA',           ltv:70, dti:29, rate_lock_expiry_days:48, condition_count:0, risk_factors:[],                                                                                                                                                               action:{ priority:'normal', action:'Verify VA appraisal request submitted and confirm FGMC portal login for processor.',      owner:'processor',  deadline:'this_week' } },

  // ── underwriting ─────────────────────────────────────────────────────────────
  { loan_id:'LN-3001', borrower_initials:'J.D.', loan_amount:512000, pipeline_stage:'underwriting', days_in_current_stage:31, days_to_projected_close:9,  close_probability:0.31, risk_level:'critical', loan_type:'USDA',         ltv:94, dti:50, rate_lock_expiry_days:4,  condition_count:9, risk_factors:[{ factor:'Rate Lock Expiry',       impact:-0.20, plain_label:'Rate lock expires in 4 days — extremely tight timeline' },{ factor:'Appraisal Status', impact:-0.16, plain_label:'Appraisal is currently disputed — can delay or kill the deal' },{ factor:'Outstanding Conditions', impact:-0.14, plain_label:'9 outstanding conditions not yet cleared' }], action:{ priority:'urgent', action:'Call underwriter NOW to prioritise condition sign-off — rate lock expires in 4 days and appraisal dispute unresolved.', owner:'closer', deadline:'today' } },
  { loan_id:'LN-3002', borrower_initials:'K.L.', loan_amount:388000, pipeline_stage:'underwriting', days_in_current_stage:14, days_to_projected_close:22, close_probability:0.68, risk_level:'moderate', loan_type:'FHA',          ltv:87, dti:41, rate_lock_expiry_days:26, condition_count:3, risk_factors:[{ factor:'Appraisal Status',       impact:-0.08, plain_label:'Appraisal ordered but not yet received' }],                                                      action:{ priority:'normal', action:'Follow up with appraiser on FHA appraisal ordered 8 days ago — confirm delivery by Friday.', owner:'processor', deadline:'tomorrow' } },
  { loan_id:'LN-3003', borrower_initials:'L.S.', loan_amount:296000, pipeline_stage:'underwriting', days_in_current_stage:7,  days_to_projected_close:30, close_probability:0.85, risk_level:'low',      loan_type:'conventional', ltv:74, dti:33, rate_lock_expiry_days:34, condition_count:1, risk_factors:[],                                                                                                                                                               action:{ priority:'normal', action:'Clear remaining PTI (prior to insurance) condition — homeowners binder not yet uploaded.',  owner:'processor', deadline:'this_week' } },
  { loan_id:'LN-3004', borrower_initials:'M.V.', loan_amount:721000, pipeline_stage:'underwriting', days_in_current_stage:26, days_to_projected_close:11, close_probability:0.43, risk_level:'high',     loan_type:'conventional', ltv:88, dti:45, rate_lock_expiry_days:13, condition_count:6, risk_factors:[{ factor:'Days Without Status Change', impact:-0.12, plain_label:'No status change in 26 days — file may be stalling' },{ factor:'Outstanding Conditions', impact:-0.11, plain_label:'6 outstanding conditions not yet cleared' },{ factor:'DTI', impact:-0.09, plain_label:'DTI of 45% is elevated — above typical guideline thresholds' }], action:{ priority:'urgent', action:'Request underwriting supervisor review — stalled 26 days with 6 conditions and lock expiring in 13 days.', owner:'management', deadline:'today' } },
  { loan_id:'LN-3005', borrower_initials:'N.B.', loan_amount:465000, pipeline_stage:'underwriting', days_in_current_stage:11, days_to_projected_close:18, close_probability:0.72, risk_level:'moderate', loan_type:'VA',           ltv:79, dti:37, rate_lock_expiry_days:22, condition_count:2, risk_factors:[{ factor:'Prior Fall-Out History', impact:-0.07, plain_label:'Prior fall-out recorded at underwriting — elevated recurrence risk' }],                          action:{ priority:'normal', action:'Obtain updated VA appraisal NOV and confirm veteran eligibility certificate is current.',  owner:'processor', deadline:'this_week' } },

  // ── cond_approval ────────────────────────────────────────────────────────────
  { loan_id:'LN-4001', borrower_initials:'O.F.', loan_amount:349000, pipeline_stage:'cond_approval', days_in_current_stage:8,  days_to_projected_close:12, close_probability:0.76, risk_level:'moderate', loan_type:'conventional', ltv:80, dti:39, rate_lock_expiry_days:15, condition_count:3, risk_factors:[{ factor:'Rate Lock Expiry', impact:-0.10, plain_label:'Rate lock expires in 15 days — tight timeline' }],                                                    action:{ priority:'high',   action:'Collect final 3 PTD conditions and submit to underwriter today to avoid rate lock extension fee.', owner:'processor', deadline:'today' } },
  { loan_id:'LN-4002', borrower_initials:'P.G.', loan_amount:582000, pipeline_stage:'cond_approval', days_in_current_stage:5,  days_to_projected_close:16, close_probability:0.88, risk_level:'low',      loan_type:'FHA',          ltv:83, dti:35, rate_lock_expiry_days:20, condition_count:1, risk_factors:[],                                                                                                                                                               action:{ priority:'normal', action:'Upload final homeowners insurance binder to LOS to clear last remaining PTD condition.',   owner:'processor', deadline:'this_week' } },
  { loan_id:'LN-4003', borrower_initials:'Q.H.', loan_amount:417000, pipeline_stage:'cond_approval', days_in_current_stage:15, days_to_projected_close:6,  close_probability:0.58, risk_level:'high',     loan_type:'conventional', ltv:86, dti:43, rate_lock_expiry_days:7,  condition_count:5, risk_factors:[{ factor:'Rate Lock Expiry', impact:-0.18, plain_label:'Rate lock expires in 7 days — tight timeline' },{ factor:'Title Status', impact:-0.12, plain_label:'Title has an open issue — must be resolved before closing' },{ factor:'Outstanding Conditions', impact:-0.11, plain_label:'5 outstanding conditions not yet cleared' }], action:{ priority:'urgent', action:'Call title company to resolve open lien on property — closing cannot proceed and lock expires in 7 days.', owner:'closer', deadline:'today' } },
  { loan_id:'LN-4004', borrower_initials:'R.Y.', loan_amount:263000, pipeline_stage:'cond_approval', days_in_current_stage:3,  days_to_projected_close:20, close_probability:0.92, risk_level:'low',      loan_type:'VA',           ltv:71, dti:30, rate_lock_expiry_days:24, condition_count:0, risk_factors:[],                                                                                                                                                               action:{ priority:'normal', action:'Prepare initial CD and schedule closing disclosure review call with veteran borrower.',    owner:'closer',    deadline:'this_week' } },

  // ── clear_to_close ───────────────────────────────────────────────────────────
  { loan_id:'LN-5001', borrower_initials:'S.Z.', loan_amount:434000, pipeline_stage:'clear_to_close', days_in_current_stage:1, days_to_projected_close:3, close_probability:0.95, risk_level:'low',      loan_type:'conventional', ltv:75, dti:32, rate_lock_expiry_days:7,  condition_count:0, risk_factors:[{ factor:'Rate Lock Expiry', impact:-0.05, plain_label:'Rate lock expires in 7 days — monitor closely' }],                                                    action:{ priority:'normal', action:'Confirm wire instructions with title and schedule signing for Wednesday at 2pm.',          owner:'closer',    deadline:'tomorrow'  } },
  { loan_id:'LN-5002', borrower_initials:'T.A.', loan_amount:371000, pipeline_stage:'clear_to_close', days_in_current_stage:2, days_to_projected_close:2, close_probability:0.97, risk_level:'low',      loan_type:'FHA',          ltv:82, dti:37, rate_lock_expiry_days:5,  condition_count:0, risk_factors:[],                                                                                                                                                               action:{ priority:'normal', action:'Send final CD to borrower and confirm receipt of cashier\'s check for closing funds.',    owner:'closer',    deadline:'today'     } },
  { loan_id:'LN-5003', borrower_initials:'U.O.', loan_amount:688000, pipeline_stage:'clear_to_close', days_in_current_stage:4, days_to_projected_close:4, close_probability:0.62, risk_level:'moderate', loan_type:'conventional', ltv:88, dti:44, rate_lock_expiry_days:5,  condition_count:1, risk_factors:[{ factor:'Rate Lock Expiry', impact:-0.15, plain_label:'Rate lock expires in 5 days — extremely tight timeline' },{ factor:'Title Status', impact:-0.08, plain_label:'Title has an open issue — must be resolved before closing' }], action:{ priority:'urgent', action:'Resolve outstanding property tax lien with title attorney — lock expires Friday and closing is blocked.', owner:'closer', deadline:'today' } },
];

// Actions derived from mock loans (at-risk only, sorted)
const MOCK_ACTIONS = MOCK_LOANS
  .filter(l => l.close_probability < 0.65)
  .map(l => ({
    ...l.action,
    loan_id:          l.loan_id,
    anon_id:          l.loan_id,
    close_probability: l.close_probability,
    risk_level:       l.risk_level,
    pipeline_stage:   l.pipeline_stage,
    days_to_close:    l.days_to_projected_close,
    risk_factors:     l.risk_factors?.map(f => f.plain_label) ?? [],
  }))
  .sort((a, b) => {
    const ORDER = { urgent:0, high:1, normal:2 };
    return (ORDER[a.priority] - ORDER[b.priority]) || (a.close_probability - b.close_probability);
  });

// ─── format helpers ────────────────────────────────────────────────────────────

const fmtM    = (v) => `$${(v / 1_000_000).toFixed(1)}M`;
const fmtPct  = (v) => `${Math.round(v * 100)}%`;
const fmtK    = (v) => v >= 1_000_000 ? `$${(v/1_000_000).toFixed(1)}M` : `$${(v/1_000).toFixed(0)}K`;

// ─── Pipeline list view ───────────────────────────────────────────────────────

const STAGE_LABELS = {
  application:    'Application',
  processing:     'Processing',
  underwriting:   'Underwriting',
  cond_approval:  'Cond. Approval',
  clear_to_close: 'Clear to Close',
};

const probColors = (p) => {
  if (p >= 0.80) return { text:'#15803d', bg:'#dcfce7', border:'#16a34a' };
  if (p >= 0.60) return { text:'#713f12', bg:'#fef9c3', border:'#eab308' };
  if (p >= 0.40) return { text:'#9a3412', bg:'#ffedd5', border:'#f97316' };
  return           { text:'#991b1b', bg:'#fee2e2', border:'#ef4444' };
};

function PipelineListView({ loans }) {
  const [sortKey, setSortKey]   = useState('close_probability');
  const [sortDir, setSortDir]   = useState('asc');   // asc = worst first

  const sorted = useMemo(() => {
    return [...loans].sort((a, b) => {
      const av = a[sortKey] ?? 0;
      const bv = b[sortKey] ?? 0;
      return sortDir === 'asc' ? av - bv : bv - av;
    });
  }, [loans, sortKey, sortDir]);

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('asc'); }
  };

  const arrow = (key) => sortKey !== key ? '' : sortDir === 'asc' ? ' ▲' : ' ▼';

  const TH = ({ k, label, right }) => (
    <th
      onClick={() => toggleSort(k)}
      style={{
        padding:'9px 12px', textAlign: right ? 'right' : 'left',
        fontSize:11, fontWeight:700, color:'#64748b', textTransform:'uppercase',
        letterSpacing:'.5px', cursor:'pointer', userSelect:'none',
        whiteSpace:'nowrap', borderBottom:'2px solid #e2e8f0',
        background: sortKey === k ? '#f8fafc' : '#fff',
      }}
    >
      {label}{arrow(k)}
    </th>
  );

  return (
    <div style={{ overflowX:'auto' }}>
      <table style={{ width:'100%', borderCollapse:'collapse', fontSize:13 }}>
        <thead>
          <tr>
            <TH k="loan_id"           label="Loan"           />
            <TH k="pipeline_stage"    label="Stage"          />
            <TH k="close_probability" label="Close %"   right />
            <TH k="days_to_projected_close" label="Days left" right />
            <TH k="rate_lock_expiry_days"   label="Lock"     right />
            <TH k="condition_count"         label="Conds"    right />
            <TH k="dti"               label="DTI"       right />
            <TH k="ltv"               label="LTV"       right />
          </tr>
        </thead>
        <tbody>
          {sorted.map((loan, i) => {
            const c          = probColors(loan.close_probability);
            const lockAlert  = loan.rate_lock_expiry_days <= 7;
            const isAtRisk   = loan.close_probability < 0.65;
            return (
              <tr
                key={loan.loan_id}
                style={{
                  background: i % 2 === 0 ? '#fff' : '#fafafa',
                  borderLeft: `3px solid ${isAtRisk ? c.border : 'transparent'}`,
                  transition: 'background .1s',
                }}
                onMouseEnter={e => e.currentTarget.style.background = '#f1f5f9'}
                onMouseLeave={e => e.currentTarget.style.background = i % 2 === 0 ? '#fff' : '#fafafa'}
              >
                <td style={{ padding:'9px 12px', borderBottom:'1px solid #f1f5f9' }}>
                  <div style={{ fontWeight:700, color:'#1e293b', fontFamily:'monospace', fontSize:12 }}>
                    {loan.loan_id}
                  </div>
                  <div style={{ fontSize:11, color:'#9ca3af', marginTop:1 }}>
                    {loan.borrower_initials} · {loan.loan_type}
                  </div>
                </td>
                <td style={{ padding:'9px 12px', borderBottom:'1px solid #f1f5f9' }}>
                  <span style={{
                    fontSize:11, padding:'2px 8px', borderRadius:10,
                    background: STAGE_META[loan.pipeline_stage]?.color + '22',
                    color: STAGE_META[loan.pipeline_stage]?.color,
                    fontWeight:600,
                  }}>
                    {STAGE_META[loan.pipeline_stage]?.icon} {STAGE_LABELS[loan.pipeline_stage]}
                  </span>
                </td>
                <td style={{ padding:'9px 12px', borderBottom:'1px solid #f1f5f9', textAlign:'right' }}>
                  <span style={{
                    fontWeight:800, fontSize:13, padding:'2px 8px',
                    borderRadius:8, background: c.bg, color: c.text,
                  }}>
                    {Math.round(loan.close_probability * 100)}%
                  </span>
                </td>
                <td style={{ padding:'9px 12px', borderBottom:'1px solid #f1f5f9', textAlign:'right',
                             color: loan.days_to_projected_close <= 7 ? '#ef4444' : '#374151',
                             fontWeight: loan.days_to_projected_close <= 7 ? 700 : 400 }}>
                  {loan.days_to_projected_close}d
                </td>
                <td style={{ padding:'9px 12px', borderBottom:'1px solid #f1f5f9', textAlign:'right' }}>
                  <span style={{
                    color: lockAlert ? '#ef4444' : '#374151',
                    fontWeight: lockAlert ? 700 : 400,
                    display:'flex', alignItems:'center', justifyContent:'flex-end', gap:4,
                  }}>
                    {lockAlert && '🔴'}{loan.rate_lock_expiry_days}d
                  </span>
                </td>
                <td style={{ padding:'9px 12px', borderBottom:'1px solid #f1f5f9', textAlign:'right',
                             color: loan.condition_count >= 5 ? '#ef4444'
                                  : loan.condition_count >= 3 ? '#f97316' : '#374151',
                             fontWeight: loan.condition_count >= 3 ? 700 : 400 }}>
                  {loan.condition_count}
                </td>
                <td style={{ padding:'9px 12px', borderBottom:'1px solid #f1f5f9', textAlign:'right',
                             color: loan.dti >= 45 ? '#ef4444' : '#6b7280' }}>
                  {loan.dti?.toFixed(0)}%
                </td>
                <td style={{ padding:'9px 12px', borderBottom:'1px solid #f1f5f9', textAlign:'right',
                             color: loan.ltv >= 90 ? '#ef4444' : '#6b7280' }}>
                  {loan.ltv?.toFixed(0)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div style={{ padding:'10px 12px', fontSize:11, color:'#9ca3af', borderTop:'1px solid #f1f5f9' }}>
        {sorted.length} loans · click any column header to sort · red left border = at-risk
      </div>
    </div>
  );
}

// ─── KPI card ─────────────────────────────────────────────────────────────────

function KpiCard({ label, value, sub, icon, color = '#1e293b', trend, alert }) {
  return (
    <div style={{
      background:'#fff', borderRadius:12, padding:'18px 22px',
      boxShadow:'0 1px 4px rgba(0,0,0,.07)', border:'1px solid #f1f5f9',
      borderTop:`3px solid ${color}`, flex:1, minWidth:0,
    }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start' }}>
        <div style={{ fontSize:22, lineHeight:1 }}>{icon}</div>
        {trend && (
          <span style={{
            fontSize:11, fontWeight:700, padding:'2px 7px', borderRadius:10,
            background: trend > 0 ? '#dcfce7' : '#fee2e2',
            color:      trend > 0 ? '#16a34a' : '#ef4444',
          }}>
            {trend > 0 ? '▲' : '▼'} {Math.abs(trend)}%
          </span>
        )}
        {alert && (
          <span style={{
            fontSize:11, fontWeight:700, padding:'2px 7px', borderRadius:10,
            background:'#fee2e2', color:'#ef4444',
          }}>
            ⚠ {alert}
          </span>
        )}
      </div>
      <div style={{ fontSize:28, fontWeight:800, color, marginTop:10, letterSpacing:'-.5px' }}>
        {value}
      </div>
      <div style={{ fontSize:13, fontWeight:600, color:'#374151', marginTop:2 }}>{label}</div>
      {sub && <div style={{ fontSize:11, color:'#9ca3af', marginTop:4 }}>{sub}</div>}
    </div>
  );
}

// ─── Stage funnel ─────────────────────────────────────────────────────────────

function StageFunnel({ loans }) {
  // Compute per-stage stats
  const stats = useMemo(() => {
    return STAGES.map((stage, i) => {
      const stageLoans = loans.filter(l => l.pipeline_stage === stage);
      const count      = stageLoans.length;
      const totalAmt   = stageLoans.reduce((s, l) => s + l.loan_amount, 0);
      const avgDays    = count > 0
        ? stageLoans.reduce((s, l) => s + l.days_in_current_stage, 0) / count
        : 0;
      const atRisk     = stageLoans.filter(l => l.close_probability < 0.65).length;
      const benchmark  = STAGE_META[stage].benchmarkDays;
      const overBench  = avgDays > benchmark;

      // Conversion: (loans in this stage) / (loans in this + previous stage)
      // approximates the % that make it through from previous
      let conversionPct = null;
      if (i > 0) {
        const prevCount = loans.filter(l => l.pipeline_stage === STAGES[i - 1]).length;
        conversionPct = prevCount > 0 ? Math.round((count / (prevCount + count)) * 100) : null;
      }

      return { stage, count, totalAmt, avgDays, atRisk, overBench, conversionPct, benchmark };
    });
  }, [loans]);

  const maxCount = Math.max(...stats.map(s => s.count), 1);

  return (
    <div style={{ background:'#fff', borderRadius:12, padding:'20px 24px',
                  boxShadow:'0 1px 4px rgba(0,0,0,.07)', border:'1px solid #f1f5f9' }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:18 }}>
        <h3 style={{ margin:0, fontSize:15, fontWeight:700, color:'#1e293b' }}>
          🔀 Stage Funnel
        </h3>
        <div style={{ display:'flex', gap:16, fontSize:11, color:'#9ca3af' }}>
          <span>🟥 Avg days over benchmark</span>
          <span>⚠ At-risk count</span>
        </div>
      </div>

      <div style={{ display:'flex', alignItems:'stretch', gap:0 }}>
        {stats.map(({ stage, count, totalAmt, avgDays, atRisk, overBench, conversionPct, benchmark }, i) => {
          const meta    = STAGE_META[stage];
          const barPct  = (count / maxCount) * 100;
          const isLast  = i === STAGES.length - 1;

          return (
            <React.Fragment key={stage}>
              {/* Conversion arrow between stages */}
              {i > 0 && (
                <div style={{
                  display:'flex', flexDirection:'column', alignItems:'center',
                  justifyContent:'center', padding:'0 6px', flexShrink:0, gap:4,
                }}>
                  <div style={{ fontSize:18, color:'#d1d5db' }}>›</div>
                  {conversionPct !== null && (
                    <div style={{
                      fontSize:10, fontWeight:700, color:'#64748b',
                      background:'#f8fafc', border:'1px solid #e2e8f0',
                      borderRadius:8, padding:'2px 5px', whiteSpace:'nowrap',
                    }}>
                      {conversionPct}%
                    </div>
                  )}
                </div>
              )}

              {/* Stage column */}
              <div style={{ flex:1, minWidth:0 }}>
                {/* Stage header */}
                <div style={{
                  background: meta.color + '15', border:`1px solid ${meta.color}40`,
                  borderRadius:'8px 8px 0 0', padding:'8px 10px',
                }}>
                  <div style={{ fontSize:12, fontWeight:700, color:meta.color, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
                    {meta.icon} {meta.label}
                  </div>
                </div>

                {/* Bar */}
                <div style={{
                  padding:'10px 10px 6px',
                  background:'#fafafa', borderLeft:`1px solid #f1f5f9`,
                  borderRight:`1px solid #f1f5f9`,
                }}>
                  {/* Count */}
                  <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline', marginBottom:6 }}>
                    <span style={{ fontSize:20, fontWeight:800, color:'#1e293b' }}>{count}</span>
                    {atRisk > 0 && (
                      <span style={{ fontSize:11, color:'#ef4444', fontWeight:700 }}>
                        ⚠ {atRisk}
                      </span>
                    )}
                  </div>

                  {/* Volume bar */}
                  <div style={{ height:6, background:'#e2e8f0', borderRadius:4, overflow:'hidden', marginBottom:6 }}>
                    <div style={{
                      height:'100%', width:`${barPct}%`,
                      background:`linear-gradient(90deg, ${meta.color}aa, ${meta.color})`,
                      borderRadius:4, transition:'width .5s ease',
                    }} />
                  </div>
                </div>

                {/* Metrics */}
                <div style={{
                  background:'#fff', border:'1px solid #f1f5f9',
                  borderRadius:'0 0 8px 8px', padding:'8px 10px',
                }}>
                  <div style={{ fontSize:11, color:'#64748b', marginBottom:4, fontWeight:600 }}>
                    {fmtK(totalAmt)}
                  </div>

                  {/* Avg days with benchmark comparison */}
                  <div style={{
                    display:'flex', alignItems:'center', gap:4,
                    background: overBench ? '#fee2e2' : '#f0fdf4',
                    borderRadius:6, padding:'4px 7px',
                  }}>
                    <span style={{ fontSize:10, color: overBench ? '#991b1b' : '#166534', fontWeight:600 }}>
                      {overBench ? '🔴' : '✅'} {avgDays.toFixed(0)}d avg
                    </span>
                    <span style={{ fontSize:10, color:'#9ca3af' }}>
                      / {benchmark}d bench
                    </span>
                  </div>
                </div>
              </div>
            </React.Fragment>
          );
        })}
      </div>

      {/* Legend */}
      <div style={{ marginTop:12, fontSize:11, color:'#9ca3af', display:'flex', gap:16 }}>
        <span>Bar width = loan count relative to largest stage</span>
        <span>$ = total loan volume in stage</span>
        <span>% arrow = stage-to-stage throughput</span>
      </div>
    </div>
  );
}

// ─── data fetching hook ───────────────────────────────────────────────────────

function usePipelineData(apiBaseUrl) {
  const [loans,   setLoans]   = useState([]);
  const [actions, setActions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBaseUrl}/pipeline-summary`, {
        signal: AbortSignal.timeout(8_000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setLoans(data.loans   ?? []);
      setActions(data.actions ?? []);
    } catch (err) {
      // Fall back to mock data in dev
      console.warn('[PipelineDashboard] API unreachable — using mock data.', err.message);
      setLoans(MOCK_LOANS);
      setActions(MOCK_ACTIONS);
      setError('Using demo data — API not reachable');
    } finally {
      setLoading(false);
      setLastRefresh(new Date());
    }
  }, [apiBaseUrl]);

  useEffect(() => { fetchData(); }, [fetchData]);

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const id = setInterval(fetchData, 5 * 60 * 1_000);
    return () => clearInterval(id);
  }, [fetchData]);

  return { loans, actions, loading, error, lastRefresh, refresh: fetchData };
}

// ─── KPI computations ─────────────────────────────────────────────────────────

function useKpis(loans) {
  return useMemo(() => {
    if (!loans.length) return {};
    const totalPipeline   = loans.reduce((s, l) => s + l.loan_amount, 0);
    const avgProb         = loans.reduce((s, l) => s + l.close_probability, 0) / loans.length;
    const atRisk          = loans.filter(l => l.close_probability < 0.65).length;
    const projectedClose  = loans.filter(l =>
      l.pipeline_stage === 'clear_to_close' ||
      (l.pipeline_stage === 'cond_approval' && l.days_to_projected_close <= 30)
    ).length;
    const criticalLock    = loans.filter(l => l.rate_lock_expiry_days <= 7).length;

    return { totalPipeline, avgProb, atRisk, projectedClose, criticalLock, total: loans.length };
  }, [loans]);
}

// ─── Main export ──────────────────────────────────────────────────────────────

export default function PipelineDashboard({ apiBaseUrl = '/api' }) {
  const { loans, actions, loading, error, lastRefresh, refresh } = usePipelineData(apiBaseUrl);
  const kpis = useKpis(loans);
  const [view, setView] = useState('heatmap'); // 'heatmap' | 'list'

  const handleActionComplete = useCallback((loanId, done) => {
    console.log(`Action for ${loanId} marked ${done ? 'complete' : 'incomplete'}`);
  }, []);

  const probRisk = kpis.avgProb < 0.65 ? '#ef4444' : kpis.avgProb < 0.75 ? '#f97316' : '#16a34a';

  return (
    <div style={{
      minHeight:'100vh',
      background:'#f1f5f9',
      fontFamily:"-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif",
    }}>
      {/* ── Header ── */}
      <header style={{
        background:'linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%)',
        padding:'16px 28px',
        display:'flex', justifyContent:'space-between', alignItems:'center',
        boxShadow:'0 2px 8px rgba(0,0,0,.2)',
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:14 }}>
          <div style={{ fontSize:28 }}>🏦</div>
          <div>
            <h1 style={{ margin:0, fontSize:20, fontWeight:800, color:'#f8fafc', letterSpacing:'-.3px' }}>
              Mortgage Pipeline Intelligence
            </h1>
            <div style={{ fontSize:12, color:'#64748b', marginTop:2 }}>
              Credit Union Pipeline Dashboard
              {lastRefresh && ` · Updated ${lastRefresh.toLocaleTimeString()}`}
            </div>
          </div>
        </div>

        <div style={{ display:'flex', gap:10, alignItems:'center' }}>
          {error && (
            <div style={{
              background:'#fef3c7', border:'1px solid #fbbf24',
              borderRadius:8, padding:'5px 12px', fontSize:12, color:'#92400e',
            }}>
              ⚠ {error}
            </div>
          )}
          <button
            onClick={refresh}
            disabled={loading}
            style={{
              padding:'8px 16px', borderRadius:8, border:'1px solid #334155',
              background: loading ? '#1e293b' : '#334155',
              color:'#94a3b8', fontSize:12, cursor: loading ? 'wait' : 'pointer',
              display:'flex', alignItems:'center', gap:6,
              transition:'all .15s ease',
            }}
          >
            {loading ? '⟳ Refreshing…' : '↻ Refresh'}
          </button>

          <div style={{ fontSize:13, color:'#64748b' }}>
            {new Date().toLocaleDateString('en-US', { weekday:'long', month:'long', day:'numeric', year:'numeric' })}
          </div>
        </div>
      </header>

      <main style={{ padding:'24px 28px', maxWidth:1600, margin:'0 auto' }}>

        {/* ── KPI Row ── */}
        <div style={{ display:'flex', gap:16, marginBottom:20 }}>
          <KpiCard
            icon="💰"
            label="Total Pipeline"
            value={kpis.totalPipeline ? fmtM(kpis.totalPipeline) : '—'}
            sub={`${kpis.total ?? 0} loans in pipeline`}
            color="#1e3a5f"
          />
          <KpiCard
            icon="📊"
            label="Avg Close Probability"
            value={kpis.avgProb ? fmtPct(kpis.avgProb) : '—'}
            sub="Across all active loans"
            color={probRisk}
            trend={kpis.avgProb ? (kpis.avgProb > 0.73 ? 3 : -2) : null}
          />
          <KpiCard
            icon="⚠️"
            label="At-Risk Loans"
            value={kpis.atRisk ?? '—'}
            sub={`< 65% close probability`}
            color={kpis.atRisk > 5 ? '#ef4444' : '#f97316'}
            alert={kpis.criticalLock ? `${kpis.criticalLock} lock expiring` : null}
          />
          <KpiCard
            icon="🎯"
            label="Projected Closings"
            value={kpis.projectedClose ?? '—'}
            sub="CTC + Cond. Approval ≤ 30 days"
            color="#16a34a"
          />
        </div>

        {/* ── Stage Funnel ── */}
        <div style={{ marginBottom:20 }}>
          {!loading && <StageFunnel loans={loans} />}
        </div>

        {/* ── Bottom row: Heatmap + Actions ── */}
        <div style={{ display:'grid', gridTemplateColumns:'1fr 400px', gap:20, alignItems:'start' }}>

          {/* Heatmap panel */}
          <div style={{
            background:'#fff', borderRadius:12, padding:'20px 24px',
            boxShadow:'0 1px 4px rgba(0,0,0,.07)', border:'1px solid #f1f5f9',
          }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:18 }}>
              <h3 style={{ margin:0, fontSize:15, fontWeight:700, color:'#1e293b' }}>
                {view === 'heatmap' ? '🗺 Pipeline Heatmap' : '☰ Pipeline List'}
              </h3>
              {/* View toggle */}
              <div style={{ display:'flex', background:'#f8fafc', borderRadius:8, padding:3, border:'1px solid #e2e8f0' }}>
                {[
                  { key:'heatmap', icon:'⊞', label:'Grid' },
                  { key:'list',    icon:'☰', label:'List' },
                ].map(({ key, icon, label }) => (
                  <button
                    key={key}
                    onClick={() => setView(key)}
                    style={{
                      padding:'5px 12px', border:'none', borderRadius:6, cursor:'pointer',
                      background: view === key ? '#fff' : 'transparent',
                      color:       view === key ? '#1e293b' : '#9ca3af',
                      fontWeight:  view === key ? 700 : 400, fontSize:12,
                      boxShadow:   view === key ? '0 1px 3px rgba(0,0,0,.1)' : 'none',
                      display:'flex', alignItems:'center', gap:5, transition:'all .15s',
                    }}
                  >
                    {icon} {label}
                  </button>
                ))}
              </div>
            </div>

            {loading ? (
              <div style={{ textAlign:'center', padding:'60px 0', color:'#9ca3af' }}>
                <div style={{ fontSize:32, marginBottom:12 }}>⟳</div>
                <div style={{ fontSize:14 }}>Loading pipeline data…</div>
              </div>
            ) : view === 'heatmap' ? (
              <PipelineHeatmap loans={loans} />
            ) : (
              <PipelineListView loans={loans} />
            )}
          </div>

          {/* Actions panel — height is content-driven; sticky tracks scroll */}
          <div style={{
            background:'#fff', borderRadius:12, padding:'20px 20px',
            boxShadow:'0 1px 4px rgba(0,0,0,.07)', border:'1px solid #f1f5f9',
            position:'sticky', top:24, alignSelf:'start',
          }}>
            {loading ? (
              <div style={{ textAlign:'center', padding:'40px 0', color:'#9ca3af', flex:1,
                            display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center' }}>
                <div style={{ fontSize:28 }}>⟳</div>
                <div style={{ fontSize:13, marginTop:8 }}>Loading actions…</div>
              </div>
            ) : (
              <ActionListCard
                actions={actions}
                onComplete={handleActionComplete}
                apiBaseUrl={apiBaseUrl}
              />
            )}
          </div>
        </div>

      </main>
    </div>
  );
}
