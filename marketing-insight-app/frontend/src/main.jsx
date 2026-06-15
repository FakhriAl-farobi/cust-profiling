import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  Pie, PieChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis
} from 'recharts'
import { askAssistant, getCategories, getHealth, getInsights, runCluster } from './api'
import './index.css'

const currency = new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 })
const number = new Intl.NumberFormat('id-ID')
const COLORS = ['#205abb', '#3775df', '#6598f5', '#00a6b4', '#6a8caf', '#f0a43a', '#2f855a', '#9f7aea', '#d95f76', '#60708a']
const DEFAULT_FEATURES = ['trx_count', 'total_amount', 'recency_days', 'avg_hour_sin', 'avg_hour_cos']

function cls(...items) {
  return items.filter(Boolean).join(' ')
}

function applyClusterNames(result, names) {
  if (!result) return null
  const renameRow = (row) => row?.cluster ? { ...row, cluster: names[row.cluster] || row.cluster } : row

  return {
    ...result,
    pca: (result.pca || []).map(renameRow),
    share: (result.share || []).map(renameRow),
    profile: (result.profile || []).map(renameRow),
    spending: {
      time_segment: (result.spending?.time_segment || []).map(renameRow),
      day_type: (result.spending?.day_type || []).map(renameRow),
      salary_type: (result.spending?.salary_type || []).map(renameRow)
    }
  }
}

function MiniIcon({ path }) {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path strokeLinecap="round" strokeLinejoin="round" d={path} />
    </svg>
  )
}

function MetricCard({ label, value, caption, tone = 'blue' }) {
  const toneClass = tone === 'green' ? 'text-emerald-600 bg-emerald-50' : tone === 'amber' ? 'text-amber-600 bg-amber-50' : 'text-corporate-700 bg-corporate-50'
  return (
    <div className="panel p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</p>
          <p className="mt-2 text-2xl font-bold text-ink">{value}</p>
        </div>
        <div className={cls('rounded-md p-2', toneClass)}>
          <MiniIcon path="M4 19V5m0 14h16M8 17V9m4 8V7m4 10v-5" />
        </div>
      </div>
      {caption && <p className="mt-2 text-xs text-muted">{caption}</p>}
    </div>
  )
}

function ChartShell({ title, subtitle, children }) {
  return (
    <div className="panel p-5">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-bold text-ink">{title}</h3>
          {subtitle && <p className="mt-1 text-xs text-muted">{subtitle}</p>}
        </div>
      </div>
      {children}
    </div>
  )
}

function ShortYAxisTick({ x, y, payload }) {
  const raw = String(payload.value || '')
  const label = raw.length > 24 ? `${raw.slice(0, 22)}...` : raw

  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={0}
        y={0}
        dy={4}
        textAnchor="end"
        fill="#66758a"
        fontSize={11}
        fontWeight={600}
      >
        {label}
        <title>{raw}</title>
      </text>
    </g>
  )
}

function EmptyState({ message }) {
  return (
    <div className="panel flex min-h-[280px] items-center justify-center p-8 text-center">
      <div>
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-md bg-corporate-100 text-corporate-700">
          <MiniIcon path="M12 9v4m0 4h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
        </div>
        <p className="text-sm font-semibold text-ink">{message}</p>
        <p className="mt-1 text-xs text-muted">Make sure the FastAPI backend is running and the filters are not too narrow.</p>
      </div>
    </div>
  )
}

function MultiSelect({ label, value, options, onChange }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-muted">{label}</span>
      <select
        multiple
        value={value}
        onChange={(event) => onChange(Array.from(event.target.selectedOptions).map((option) => option.value))}
        className="field min-h-[94px]"
      >
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
    </label>
  )
}

function Sidebar({ categories, filter, setFilter, onRefresh, loading, apiOk, source, isOpen, onToggle }) {
  return (
    <>
      <button
        type="button"
        onClick={onToggle}
        className={cls(
          'fixed left-4 top-4 z-30 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-line bg-white text-corporate-800 shadow-soft transition hover:bg-corporate-50',
          isOpen && 'pointer-events-none -translate-x-3 opacity-0'
        )}
        aria-label="Show sidebar"
        title="Show sidebar"
      >
        <MiniIcon path="M4 6h16M4 12h16M4 18h16" />
      </button>

      <aside className={cls(
        'fixed inset-y-0 left-0 z-20 flex w-80 flex-col border-r border-line bg-white transition-transform duration-300 ease-out',
        isOpen ? 'translate-x-0' : '-translate-x-full'
      )}>
      <div className="border-b border-line px-6 py-5">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-corporate-700 text-white">
            <MiniIcon path="M3 13h4l3 7 4-16 3 9h4" />
          </div>
          <div>
            <p className="text-sm font-bold text-ink">Artajasa</p>
            <p className="text-xs font-semibold text-corporate-600">Customer Profiling</p>
          </div>
          </div>
          <button
            type="button"
            onClick={onToggle}
            className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-line bg-white text-muted transition hover:bg-corporate-50 hover:text-corporate-800"
            aria-label="Hide sidebar"
            title="Hide sidebar"
          >
            <MiniIcon path="M15 18 9 12l6-6" />
          </button>
        </div>
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-5">
        <div className="subtle-panel p-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-muted">API Status</span>
            <span className={cls('rounded-full px-2 py-0.5 text-xs font-bold', apiOk ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700')}>
              {apiOk ? 'Connected' : 'Offline'}
            </span>
          </div>
          <p className="mt-2 text-xs text-muted">Source: {source || 'waiting for data'}</p>
        </div>

        <MultiSelect
          label="Industry Sector"
          value={filter.industries}
          options={categories}
          onChange={(industries) => setFilter({ ...filter, industries })}
        />

        <label className="block">
          <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-muted">CPAN Sample Limit</span>
          <select className="field" value={filter.cpan_limit} onChange={(e) => setFilter({ ...filter, cpan_limit: Number(e.target.value) })}>
            {[1000, 3000, 5000, 10000, 20000, 50000, 100000].map((v) => <option key={v} value={v}>{number.format(v)}</option>)}
          </select>
        </label>

        <div className="grid grid-cols-2 gap-3">
          <label>
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-muted">Start</span>
            <input className="field" type="date" value={filter.start_date || ''} onChange={(e) => setFilter({ ...filter, start_date: e.target.value || null })} />
          </label>
          <label>
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-muted">End</span>
            <input className="field" type="date" value={filter.end_date || ''} onChange={(e) => setFilter({ ...filter, end_date: e.target.value || null })} />
          </label>
        </div>

        <button className="btn-primary w-full gap-2" onClick={onRefresh} disabled={loading || filter.industries.length === 0}>
          <MiniIcon path="M20 11a8.1 8.1 0 0 0-15.5-2M4 5v4h4m-4 4a8.1 8.1 0 0 0 15.5 2M20 19v-4h-4" />
          {loading ? 'Loading data...' : 'Refresh Insight'}
        </button>
      </div>
      </aside>
    </>
  )
}

function Overview({ data }) {
  const metrics = data.metrics
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Total CPAN" value={number.format(metrics.customers)} caption="Unique customer identifier" />
        <MetricCard label="Transactions" value={number.format(metrics.transactions)} caption="Total filtered records" />
        <MetricCard label="Total Amount" value={currency.format(metrics.total_amount)} caption="Transaction value" tone="green" />
        <MetricCard label="Average Ticket" value={currency.format(metrics.avg_amount)} caption="Average transaction value" tone="amber" />
      </div>

      <div className="grid grid-cols-[1.4fr_1fr] gap-5">
        <ChartShell title="Transaction Value Trend" subtitle="A smoother daily view of transaction momentum">
          <div className="h-72">
            <ResponsiveContainer>
              <AreaChart data={data.daily}>
                <defs>
                  <linearGradient id="amountGradient" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="5%" stopColor="#3775df" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#3775df" stopOpacity={0.03} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e3ebf7" />
                <XAxis dataKey="time" tick={{ fontSize: 11, fill: '#66758a' }} tickLine={false} axisLine={false} minTickGap={28} />
                <YAxis tick={{ fontSize: 11, fill: '#66758a' }} tickLine={false} axisLine={false} width={80} tickFormatter={(v) => `${Math.round(v / 1000000)}M`} />
                <Tooltip formatter={(v, n) => [n === 'value' ? currency.format(v) : number.format(v), n === 'value' ? 'Amount' : 'Transactions']} />
                <Area type="monotone" dataKey="value" stroke="#205abb" strokeWidth={3} fill="url(#amountGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </ChartShell>

        <ChartShell title="Top Merchants" subtitle="Merchants with the largest amount contribution">
          <div className="h-80">
            <ResponsiveContainer>
              <BarChart data={data.top_merchants} layout="vertical" margin={{ top: 8, left: 22, right: 20, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e3ebf7" />
                <XAxis type="number" hide />
                <YAxis
                  type="category"
                  dataKey="merchant_name"
                  width={160}
                  interval={0}
                  tick={<ShortYAxisTick />}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  formatter={(v, name) => [name === 'amount' ? currency.format(v) : number.format(v), name === 'amount' ? 'Amount' : 'Transactions']}
                  labelFormatter={(label) => `Merchant: ${label}`}
                />
                <Bar dataKey="amount" radius={[0, 8, 8, 0]} fill="#3775df" barSize={18} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartShell>
      </div>

      <ChartShell title="Preview Data" subtitle="Filtered transaction sample">
        <DataTable rows={data.table} columns={['timestamp', 'customer_id_masked', 'merchant_name', 'mcc_description', 'acquirer', 'issuer', 'amount']} />
      </ChartShell>
    </div>
  )
}

function Rfm({ data }) {
  const total = data.rfm.reduce((sum, row) => sum + row.Count, 0)
  return (
    <div className="grid grid-cols-[1fr_1.2fr] gap-5">
      <ChartShell title="RFM Segment Composition" subtitle={`${number.format(total)} customers segmented`}>
        <div className="h-96">
          <ResponsiveContainer>
            <PieChart>
              <Pie data={data.rfm} dataKey="Count" nameKey="Segment" innerRadius={72} outerRadius={132} paddingAngle={2}>
                {data.rfm.map((entry, index) => <Cell key={entry.Segment} fill={COLORS[index % COLORS.length]} />)}
              </Pie>
              <Tooltip formatter={(value, name, item) => [number.format(value), item.payload.Segment]} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </ChartShell>

      <div className="space-y-5">
        <ChartShell title="Segment Detail" subtitle="Average metrics for marketing strategy">
          <DataTable rows={data.rfm} columns={['Segment', 'Count', 'Percentage', 'Avg_Frequency', 'Avg_Monetary', 'Avg_Recency']} />
        </ChartShell>
        <ChartShell title="Quick Strategy" subtitle="Ready-to-use action examples">
          <div className="grid grid-cols-2 gap-3">
            {data.rfm.slice(0, 4).map((row) => (
              <div key={row.Segment} className="subtle-panel p-3">
                <p className="text-sm font-bold text-corporate-800">{row.Segment}</p>
                <p className="mt-1 text-xs leading-5 text-muted">
                  Prioritize personalized campaigns based on recency and transaction value. This segment contains {row.Percentage}% of customers.
                </p>
              </div>
            ))}
          </div>
        </ChartShell>
      </div>
    </div>
  )
}

function Clustering({ filter, data, cluster, setCluster, clusterResult, setClusterResult, clusterNames, setClusterNames }) {
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const availableSegments = data?.rfm?.map((row) => row.Segment) || []
  const displayResult = useMemo(() => applyClusterNames(clusterResult, clusterNames), [clusterResult, clusterNames])

  const submit = async () => {
    setRunning(true)
    setError('')
    try {
      const res = await runCluster({ ...filter, ...cluster })
      setClusterResult(res.data)
      const discovered = Array.from(new Set((res.data.share || []).map((row) => row.cluster)))
      setClusterNames((previous) => {
        const next = { ...previous }
        discovered.forEach((name) => {
          if (!next[name]) next[name] = name
        })
        return next
      })
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setRunning(false)
    }
  }

  const pcaGroups = useMemo(() => {
    const groups = {}
    ;(displayResult?.pca || []).forEach((point) => {
      groups[point.cluster] = groups[point.cluster] || []
      groups[point.cluster].push(point)
    })
    return Object.entries(groups)
  }, [displayResult])

  return (
    <div className="space-y-5">
      <div className="panel grid grid-cols-[1.2fr_1fr_1fr_auto] gap-4 p-5">
        <MultiSelect
          label="RFM Segment"
          value={cluster.rfm_segments}
          options={availableSegments}
          onChange={(rfm_segments) => setCluster({ ...cluster, rfm_segments })}
        />
        <label>
          <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-muted">Model</span>
          <select className="field" value={cluster.model_type} onChange={(e) => setCluster({ ...cluster, model_type: e.target.value })}>
            {['MiniBatch KMeans', 'KMeans', 'Gaussian Mixture', 'DBSCAN', 'HDBSCAN'].map((model) => <option key={model}>{model}</option>)}
          </select>
        </label>
        <label>
          <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-muted">Clusters</span>
          <input className="field" type="number" min="2" max="20" value={cluster.n_clusters} onChange={(e) => setCluster({ ...cluster, n_clusters: Number(e.target.value) })} />
        </label>
        <div className="flex items-end">
          <button className="btn-primary min-w-36 gap-2" onClick={submit} disabled={running || cluster.rfm_segments.length === 0}>
            <MiniIcon path="m5 3 14 9-14 9V3Z" />
            {running ? 'Running...' : 'Run Profiling'}
          </button>
        </div>
      </div>

      {error && <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm font-semibold text-rose-700">{error}</div>}

      {displayResult ? (
        <>
          <ChartShell title="Cluster Naming" subtitle="Rename generated clusters to match your business interpretation">
            <div className="grid grid-cols-4 gap-3">
              {(clusterResult.share || []).map((row) => (
                <label key={row.cluster} className="block">
                  <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-muted">{row.cluster}</span>
                  <input
                    className="field"
                    value={clusterNames[row.cluster] || row.cluster}
                    onChange={(event) => setClusterNames((previous) => ({ ...previous, [row.cluster]: event.target.value }))}
                    placeholder="Cluster name"
                  />
                </label>
              ))}
            </div>
          </ChartShell>

          <div className="grid grid-cols-5 gap-4">
            <MetricCard label="Selected CPAN" value={number.format(displayResult.selected_customers)} />
            <MetricCard label="Clusters" value={number.format(displayResult.metrics.clusters_found)} />
            <MetricCard label="Outliers" value={number.format(displayResult.metrics.outliers)} tone="amber" />
            <MetricCard label="Silhouette" value={displayResult.metrics.silhouette ?? '-'} />
            <MetricCard label="PCA Variance" value={`${Math.round(displayResult.metrics.explained_variance_2d * 100)}%`} />
          </div>

          <div className="grid grid-cols-[1.1fr_0.9fr] gap-5">
            <ChartShell title="PCA Cluster Projection" subtitle="A lighter scatter view for reading cluster patterns">
              <div className="h-[420px]">
                <ResponsiveContainer>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e3ebf7" />
                    <XAxis dataKey="x" type="number" tick={{ fontSize: 11, fill: '#66758a' }} tickLine={false} axisLine={false} />
                    <YAxis dataKey="y" type="number" tick={{ fontSize: 11, fill: '#66758a' }} tickLine={false} axisLine={false} />
                    <ZAxis range={[24, 72]} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    {pcaGroups.map(([name, points], index) => (
                      <Scatter key={name} name={name} data={points} fill={COLORS[index % COLORS.length]} fillOpacity={0.72} />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </ChartShell>

            <ChartShell title="Cluster Market Share" subtitle="Value percentage by cluster">
              <div className="h-[420px]">
                <ResponsiveContainer>
                  <BarChart data={displayResult.share}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e3ebf7" />
                    <XAxis dataKey="cluster" tick={{ fontSize: 11, fill: '#66758a' }} tickLine={false} axisLine={false} />
                    <YAxis tick={{ fontSize: 11, fill: '#66758a' }} tickLine={false} axisLine={false} />
                    <Tooltip formatter={(v, name) => [name === 'percentage' ? `${v}%` : currency.format(v), name]} />
                    <Bar dataKey="percentage" radius={[8, 8, 0, 0]} barSize={34}>
                      {displayResult.share.map((_, index) => <Cell key={index} fill={COLORS[index % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartShell>
          </div>

          <ChartShell title="Customer Profile Result" subtitle="Average feature value by cluster">
            <DataTable rows={displayResult.profile} columns={Object.keys(displayResult.profile[0] || {})} />
          </ChartShell>
        </>
      ) : (
        <EmptyState message="Run profiling to view clustering results." />
      )}
    </div>
  )
}

function Spending({ clusterResult }) {
  if (!clusterResult) return <EmptyState message="Run clustering first to view spending patterns." />

  const renderGrouped = (title, rows, keyName) => (
    <ChartShell title={title} subtitle="Frequency and amount compared by cluster">
      <div className="h-80">
        <ResponsiveContainer>
          <BarChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e3ebf7" />
            <XAxis dataKey={keyName} tick={{ fontSize: 11, fill: '#66758a' }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fontSize: 11, fill: '#66758a' }} tickLine={false} axisLine={false} />
            <Tooltip formatter={(v, name) => [name === 'amount' ? currency.format(v) : number.format(v), name]} />
            <Legend />
            <Bar dataKey="transactions" fill="#6598f5" radius={[6, 6, 0, 0]} />
            <Bar dataKey="amount" fill="#205abb" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartShell>
  )

  return (
    <div className="grid grid-cols-2 gap-5">
      {renderGrouped('By Time Segment', clusterResult.spending.time_segment, 'time_segment')}
      {renderGrouped('Weekend vs Weekday', clusterResult.spending.day_type, 'day_type')}
      {renderGrouped('Salary Day Effect', clusterResult.spending.salary_type, 'salary_type')}
      <ChartShell title="Cluster Narrative" subtitle="Quick summary for interpretation">
        <div className="space-y-3">
          {clusterResult.share.map((row) => (
            <div key={row.cluster} className="subtle-panel p-3">
              <div className="flex items-center justify-between">
                <p className="text-sm font-bold text-corporate-800">{row.cluster}</p>
                <p className="text-xs font-bold text-corporate-700">{row.percentage}% value</p>
              </div>
              <p className="mt-1 text-xs leading-5 text-muted">Use timing and transaction value patterns to decide voucher timing, retargeting, or education campaigns.</p>
            </div>
          ))}
        </div>
      </ChartShell>
    </div>
  )
}

function Assistant({ clusterResult }) {
  const [message, setMessage] = useState('')
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)

  const submit = async () => {
    if (!message.trim()) return
    const userMessage = message.trim()
    setHistory((items) => [...items, { role: 'user', text: userMessage }])
    setMessage('')
    setLoading(true)
    try {
      const res = await askAssistant({ message: userMessage, profile: { share: clusterResult?.share, profile: clusterResult?.profile } })
      setHistory((items) => [...items, { role: 'assistant', text: res.data.reply }])
    } catch (err) {
      setHistory((items) => [...items, { role: 'assistant', text: err.response?.data?.detail || err.message }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel flex h-[620px] flex-col">
      <div className="border-b border-line p-5">
        <h3 className="text-sm font-bold text-ink">AI Assistant</h3>
        <p className="mt-1 text-xs text-muted">The assistant reads the currently active cluster summary.</p>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto bg-corporate-50/60 p-5">
        {history.length === 0 && <p className="text-sm text-muted">Ask for campaign recommendations, cluster interpretation, or segment naming ideas.</p>}
        {history.map((item, index) => (
          <div key={index} className={cls('max-w-[80%] rounded-lg px-4 py-3 text-sm leading-6', item.role === 'user' ? 'ml-auto bg-corporate-700 text-white' : 'bg-white text-ink shadow-sm')}>
            {item.text}
          </div>
        ))}
      </div>
      <div className="flex gap-3 border-t border-line p-4">
        <input className="field" value={message} onChange={(e) => setMessage(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && submit()} placeholder="Write a question..." />
        <button className="btn-primary" onClick={submit} disabled={loading}>{loading ? '...' : 'Send'}</button>
      </div>
    </div>
  )
}

function DataTable({ rows, columns }) {
  return (
    <div className="max-h-[430px] overflow-auto rounded-md border border-line">
      <table className="min-w-full divide-y divide-line text-left text-sm">
        <thead className="sticky top-0 bg-corporate-50">
          <tr>
            {columns.map((col) => <th key={col} className="whitespace-nowrap px-3 py-2 text-xs font-bold uppercase tracking-wide text-muted">{col}</th>)}
          </tr>
        </thead>
        <tbody className="divide-y divide-line bg-white">
          {rows.map((row, index) => (
            <tr key={index} className="hover:bg-corporate-50/70">
              {columns.map((col) => (
                <td key={col} className="whitespace-nowrap px-3 py-2 text-xs text-ink">
                  {typeof row[col] === 'number' ? number.format(Number(row[col].toFixed ? row[col].toFixed(2) : row[col])) : row[col] ?? '-'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function App() {
  const [categories, setCategories] = useState([])
  const [apiOk, setApiOk] = useState(false)
  const [data, setData] = useState(null)
  const [source, setSource] = useState('')
  const [loading, setLoading] = useState(false)
  const [active, setActive] = useState('overview')
  const [error, setError] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [clusterResult, setClusterResult] = useState(null)
  const [clusterNames, setClusterNames] = useState({})
  const [filter, setFilter] = useState({
    industries: ['Retail Outlet Services', 'Miscellaneous Stores'],
    cpan_limit: 5000,
    row_limit: 200000,
    start_date: null,
    end_date: null,
    acquirers: [],
    issuers: [],
    merchant_types: []
  })
  const [cluster, setCluster] = useState({
    rfm_segments: ['Hibernating', 'At Risk', 'About To Sleep'],
    features: DEFAULT_FEATURES,
    model_type: 'MiniBatch KMeans',
    n_clusters: 4,
    batch_size: 1024,
    eps: 0.5,
    min_samples: 8,
    min_cluster_size: 20
  })

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await getInsights(filter)
      setData(res.data)
      setSource(res.data.source)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    Promise.all([getHealth(), getCategories()])
      .then(([, cat]) => {
        setApiOk(true)
        setCategories(cat.data.categories)
      })
      .catch(() => setApiOk(false))
  }, [])

  useEffect(() => {
    if (categories.length) load()
  }, [categories.length])

  const namedClusterResult = useMemo(() => applyClusterNames(clusterResult, clusterNames), [clusterResult, clusterNames])

  const tabs = [
    ['overview', 'Data'],
    ['rfm', 'RFM'],
    ['cluster', 'Advanced Clustering'],
    ['spending', 'Spending Pattern'],
    ['assistant', 'AI Chatbot']
  ]

  return (
    <div className="min-h-screen">
      <Sidebar
        categories={categories}
        filter={filter}
        setFilter={setFilter}
        onRefresh={load}
        loading={loading}
        apiOk={apiOk}
        source={source}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen((open) => !open)}
      />
      <main className={cls('px-8 py-7 transition-all duration-300 ease-out', sidebarOpen ? 'ml-80' : 'ml-0')}>
        <header className="mb-6 flex items-start justify-between gap-5">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-corporate-600">Marketing Insight Platform</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-ink">Customer Profiling Dashboard</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">RFM segmentation, customer clustering, and spending patterns for reading QRIS transaction behavior in a more operational way.</p>
          </div>
          <div className="rounded-lg border border-corporate-100 bg-white px-4 py-3 text-right shadow-sm">
            <p className="text-xs font-semibold text-muted">Observation Window</p>
            <p className="mt-1 text-sm font-bold text-corporate-800">{data?.filters?.min_date || '-'} to {data?.filters?.max_date || '-'}</p>
          </div>
        </header>

        <div className="mb-6 flex flex-wrap gap-2">
          {tabs.map(([key, label]) => (
            <button key={key} onClick={() => setActive(key)} className={cls('tab', active === key && 'tab-active')}>{label}</button>
          ))}
        </div>

        {loading && <EmptyState message="Loading insights..." />}
        {!loading && error && <EmptyState message={error} />}
        {!loading && !error && data && (
          <>
            {active === 'overview' && <Overview data={data} />}
            {active === 'rfm' && <Rfm data={data} />}
            {active === 'cluster' && <Clustering filter={filter} data={data} cluster={cluster} setCluster={setCluster} clusterResult={clusterResult} setClusterResult={setClusterResult} clusterNames={clusterNames} setClusterNames={setClusterNames} />}
            {active === 'spending' && <Spending clusterResult={namedClusterResult} />}
            {active === 'assistant' && <Assistant clusterResult={namedClusterResult} />}
          </>
        )}
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)
