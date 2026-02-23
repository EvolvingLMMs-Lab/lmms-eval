import { useEffect, useMemo, useState } from 'react'

const API_BASE = ''
const DEFAULT_LOGS_PATH = './logs/'
const SAMPLE_PAGE_SIZE = 20

interface LogRunSummary {
  run_id: string
  model_name: string
  date: string
  tasks: string[]
  metrics: Record<string, Record<string, unknown>>
  total_evaluation_time_seconds: unknown
  config: Record<string, unknown>
  n_samples: Record<string, unknown>
}

interface RunResults {
  results?: Record<string, Record<string, unknown>>
  config?: Record<string, unknown>
  model_name?: string
  date?: string
  total_evaluation_time_seconds?: unknown
  ['n-samples']?: Record<string, unknown>
  [key: string]: unknown
}

interface SamplesResponse {
  samples: Record<string, unknown>[]
  total: number
  offset: number
  limit: number
}

interface MetricRow {
  task: string
  metric: string
  value: unknown
  stderr: unknown
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function formatDate(date: string): string {
  const match = date.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/)
  if (!match) return date
  const [, year, month, day, hour, minute, second] = match
  return `${year}-${month}-${day} ${hour}:${minute}:${second}`
}

function formatDuration(value: unknown): string {
  if (typeof value === 'number') {
    return `${value.toFixed(2)}s`
  }

  if (typeof value === 'string') {
    const parsed = Number(value)
    if (!Number.isNaN(parsed)) {
      return `${parsed.toFixed(2)}s`
    }
    return value
  }

  return 'N/A'
}

function valueToText(value: unknown): string {
  if (value === null || value === undefined) {
    return ''
  }
  if (typeof value === 'string') {
    return value
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function metricValueToText(value: unknown): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4)
  }
  const text = valueToText(value)
  return text || 'N/A'
}

function compactValue(value: unknown): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4)
  }
  if (typeof value === 'string') {
    return value.length > 36 ? `${value.slice(0, 33)}...` : value
  }
  if (typeof value === 'boolean') {
    return String(value)
  }
  if (Array.isArray(value)) {
    return `${value.length} items`
  }
  if (isRecord(value) && typeof value.score === 'number') {
    return `score=${Number.isInteger(value.score) ? value.score : value.score.toFixed(4)}`
  }
  const text = valueToText(value)
  return text.length > 36 ? `${text.slice(0, 33)}...` : text
}

function filteredResponsesText(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map(item => valueToText(item)).join('\n\n')
  }
  return valueToText(value)
}

function extractMetricRows(runResults: RunResults | null): MetricRow[] {
  if (!runResults?.results || !isRecord(runResults.results)) {
    return []
  }

  const rows: MetricRow[] = []
  for (const [taskName, taskMetrics] of Object.entries(runResults.results)) {
    if (!isRecord(taskMetrics)) {
      continue
    }

    for (const [metricName, metricValue] of Object.entries(taskMetrics)) {
      if (metricName === 'alias' || metricName.includes('_stderr')) {
        continue
      }

      const stderrKey = metricName.endsWith(',none')
        ? metricName.replace(/,none$/, '_stderr,none')
        : `${metricName}_stderr`

      rows.push({
        task: taskName,
        metric: metricName.replace(/,none$/, ''),
        value: metricValue,
        stderr: taskMetrics[stderrKey] ?? 'N/A',
      })
    }
  }

  rows.sort((a, b) => {
    const taskDiff = a.task.localeCompare(b.task)
    if (taskDiff !== 0) return taskDiff
    return a.metric.localeCompare(b.metric)
  })

  return rows
}

function extractSampleBadges(sample: Record<string, unknown>): Array<[string, unknown]> {
  const ignored = new Set([
    'doc_id',
    'input',
    'target',
    'filtered_resps',
    'doc_hash',
    'prompt_hash',
    'target_hash',
  ])

  return Object.entries(sample).filter(([key]) => !ignored.has(key))
}

export default function LogViewer() {
  const [logsPath, setLogsPath] = useState(DEFAULT_LOGS_PATH)
  const [runs, setRuns] = useState<LogRunSummary[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [runsError, setRunsError] = useState<string | null>(null)

  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [runResults, setRunResults] = useState<RunResults | null>(null)
  const [runLoading, setRunLoading] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)

  const [selectedTask, setSelectedTask] = useState('')
  const [sampleOffset, setSampleOffset] = useState(0)
  const [samplesResponse, setSamplesResponse] = useState<SamplesResponse>({
    samples: [],
    total: 0,
    offset: 0,
    limit: SAMPLE_PAGE_SIZE,
  })
  const [samplesLoading, setSamplesLoading] = useState(false)
  const [samplesError, setSamplesError] = useState<string | null>(null)

  const selectedRun = useMemo(
    () => runs.find(run => run.run_id === selectedRunId) ?? null,
    [runs, selectedRunId],
  )

  const availableTasks = useMemo(() => {
    const fromResults = runResults?.results ? Object.keys(runResults.results) : []
    if (fromResults.length > 0) {
      return fromResults
    }
    return selectedRun?.tasks ?? []
  }, [runResults, selectedRun])

  const metricRows = useMemo(() => extractMetricRows(runResults), [runResults])

  const scanRuns = async () => {
    setRunsLoading(true)
    setRunsError(null)

    try {
      const response = await fetch(
        `${API_BASE}/logs/runs?logs_path=${encodeURIComponent(logsPath)}`,
      )
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const data = (await response.json()) as LogRunSummary[]
      setRuns(data)
      setSelectedRunId(prev => {
        if (prev && data.some(run => run.run_id === prev)) {
          return prev
        }
        return data[0]?.run_id ?? null
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to scan logs'
      setRuns([])
      setSelectedRunId(null)
      setRunsError(message)
    } finally {
      setRunsLoading(false)
    }
  }

  useEffect(() => {
    void scanRuns()
  }, [])

  useEffect(() => {
    if (!selectedRunId) {
      setRunResults(null)
      setRunError(null)
      setSelectedTask('')
      setSampleOffset(0)
      return
    }

    const loadRunResults = async () => {
      setRunLoading(true)
      setRunError(null)

      try {
        const response = await fetch(
          `${API_BASE}/logs/runs/${encodeURIComponent(selectedRunId)}/results?logs_path=${encodeURIComponent(logsPath)}`,
        )
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const data = (await response.json()) as RunResults
        setRunResults(data)

        const tasksFromResults = data.results ? Object.keys(data.results) : []
        const nextTasks = tasksFromResults.length > 0 ? tasksFromResults : selectedRun?.tasks ?? []

        setSelectedTask(prev => {
          if (prev && nextTasks.includes(prev)) {
            return prev
          }
          return nextTasks[0] ?? ''
        })
        setSampleOffset(0)
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to load results'
        setRunResults(null)
        setRunError(message)
      } finally {
        setRunLoading(false)
      }
    }

    void loadRunResults()
  }, [selectedRunId, logsPath, selectedRun])

  useEffect(() => {
    if (!selectedRunId || !selectedTask) {
      setSamplesResponse({
        samples: [],
        total: 0,
        offset: sampleOffset,
        limit: SAMPLE_PAGE_SIZE,
      })
      setSamplesError(null)
      return
    }

    const loadSamples = async () => {
      setSamplesLoading(true)
      setSamplesError(null)

      try {
        const response = await fetch(
          `${API_BASE}/logs/runs/${encodeURIComponent(selectedRunId)}/samples/${encodeURIComponent(selectedTask)}?logs_path=${encodeURIComponent(logsPath)}&offset=${sampleOffset}&limit=${SAMPLE_PAGE_SIZE}`,
        )
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        const data = (await response.json()) as SamplesResponse
        setSamplesResponse(data)
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to load samples'
        setSamplesResponse({
          samples: [],
          total: 0,
          offset: sampleOffset,
          limit: SAMPLE_PAGE_SIZE,
        })
        setSamplesError(message)
      } finally {
        setSamplesLoading(false)
      }
    }

    void loadSamples()
  }, [selectedRunId, selectedTask, sampleOffset, logsPath])

  const modelName = runResults?.model_name ?? selectedRun?.model_name ?? ''
  const runDate = runResults?.date ?? selectedRun?.date ?? ''
  const evalTime = runResults?.total_evaluation_time_seconds ?? selectedRun?.total_evaluation_time_seconds
  const config = runResults?.config ?? selectedRun?.config ?? {}

  const configKeys = ['model', 'model_args', 'batch_size', 'limit', 'verbosity', 'device', 'output_path']
  const configEntries = configKeys
    .filter(key => key in config)
    .map(key => [key, config[key]] as const)

  const sampleRangeStart = samplesResponse.total === 0 ? 0 : samplesResponse.offset + 1
  const sampleRangeEnd = Math.min(
    samplesResponse.offset + samplesResponse.limit,
    samplesResponse.total,
  )

  return (
    <div className="flex flex-1 overflow-hidden">
      <div className="w-full md:w-[400px] lg:w-[450px] xl:w-[500px] 2xl:w-[550px] min-w-[320px] max-w-[600px] bg-white border-r border-neutral-200 flex flex-col flex-shrink-0">
        <div className="p-6 border-b border-neutral-100 space-y-3">
          <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider">
            Logs Path
          </label>
          <div className="flex items-center gap-2">
            <input
              value={logsPath}
              onChange={event => setLogsPath(event.target.value)}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  void scanRuns()
                }
              }}
              placeholder="./logs/"
              className="flex-1 bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors text-neutral-600"
            />
            <button
              onClick={() => void scanRuns()}
              disabled={runsLoading}
              className={`px-4 py-2 text-[10px] uppercase tracking-wider font-medium border transition-colors ${
                runsLoading
                  ? 'text-neutral-300 border-neutral-200 cursor-not-allowed'
                  : 'text-neutral-500 border-neutral-200 hover:border-black hover:text-black'
              }`}
            >
              {runsLoading ? 'Scanning...' : 'Scan'}
            </button>
          </div>
          {runsError && <div className="text-[10px] font-mono text-red-600">{runsError}</div>}
          <div className="text-[10px] uppercase tracking-wider text-neutral-400">
            {runs.length} run{runs.length === 1 ? '' : 's'}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-2 bg-white">
          {runs.length === 0 && !runsLoading ? (
            <div className="p-3 text-xs text-neutral-400 italic">No runs found.</div>
          ) : (
            runs.map(run => (
              <button
                key={run.run_id}
                onClick={() => {
                  setSelectedRunId(run.run_id)
                  setSampleOffset(0)
                }}
                className={`w-full text-left border px-3 py-3 transition-colors ${
                  run.run_id === selectedRunId
                    ? 'border-neutral-300 bg-neutral-100'
                    : 'border-neutral-200 bg-white hover:bg-neutral-50'
                }`}
              >
                <div className="text-xs font-bold text-neutral-900 break-all">{run.model_name || 'Unknown model'}</div>
                <div className="mt-1 text-[10px] font-mono text-neutral-500">{formatDate(run.date)}</div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {run.tasks.slice(0, 6).map(task => (
                    <span
                      key={`${run.run_id}-${task}`}
                      className="px-1.5 py-0.5 text-[10px] font-mono bg-white border border-neutral-200 text-neutral-500"
                    >
                      {task}
                    </span>
                  ))}
                  {run.tasks.length > 6 && (
                    <span className="px-1.5 py-0.5 text-[10px] font-mono bg-white border border-neutral-200 text-neutral-400">
                      +{run.tasks.length - 6} more
                    </span>
                  )}
                </div>
                <div className="mt-2 text-[10px] uppercase tracking-wider text-neutral-400">
                  Eval time: {formatDuration(run.total_evaluation_time_seconds)}
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      <div className="flex-1 min-w-0 flex flex-col bg-neutral-50/30">
        {!selectedRun ? (
          <div className="flex-1 flex items-center justify-center text-neutral-400 text-xs uppercase tracking-wider">
            Select a run to view details
          </div>
        ) : (
          <>
            <div className="h-[46%] min-h-[260px] border-b border-neutral-200 flex flex-col bg-white">
              <div className="px-6 py-4 border-b border-neutral-100">
                <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-widest">Results</h2>
              </div>

              <div className="flex-1 overflow-auto p-6 space-y-4 scrollbar-thin">
                {runLoading ? (
                  <div className="text-xs text-neutral-400 italic">Loading results...</div>
                ) : runError ? (
                  <div className="text-xs text-red-600 font-mono">{runError}</div>
                ) : (
                  <>
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
                      <div className="border border-neutral-200 bg-neutral-50/70 p-3">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-400 mb-1">Model</div>
                        <div className="text-xs font-mono text-neutral-700 break-all">{modelName || 'N/A'}</div>
                      </div>
                      <div className="border border-neutral-200 bg-neutral-50/70 p-3">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-400 mb-1">Date</div>
                        <div className="text-xs font-mono text-neutral-700">{formatDate(runDate)}</div>
                      </div>
                      <div className="border border-neutral-200 bg-neutral-50/70 p-3">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-400 mb-1">Evaluation Time</div>
                        <div className="text-xs font-mono text-neutral-700">{formatDuration(evalTime)}</div>
                      </div>
                      <div className="border border-neutral-200 bg-neutral-50/70 p-3">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-400 mb-1">Tasks</div>
                        <div className="text-xs font-mono text-neutral-700">{availableTasks.length}</div>
                      </div>
                    </div>

                    {configEntries.length > 0 && (
                      <div className="border border-neutral-200 bg-white p-3">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-400 mb-2">Config</div>
                        <div className="flex flex-wrap gap-2">
                          {configEntries.map(([key, value]) => (
                            <span
                              key={key}
                              className="inline-flex items-center gap-1 px-2 py-1 text-[10px] font-mono border border-neutral-200 bg-neutral-50 text-neutral-600"
                              title={`${key}=${valueToText(value)}`}
                            >
                              <span className="text-neutral-400">{key}</span>
                              <span className="max-w-[280px] truncate">{valueToText(value) || 'N/A'}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="border border-neutral-200 overflow-hidden">
                      <table className="min-w-full text-xs font-mono">
                        <thead className="bg-neutral-50 border-b border-neutral-200">
                          <tr>
                            <th className="text-left px-3 py-2 text-[10px] uppercase tracking-wider text-neutral-400">Task</th>
                            <th className="text-left px-3 py-2 text-[10px] uppercase tracking-wider text-neutral-400">Metric</th>
                            <th className="text-left px-3 py-2 text-[10px] uppercase tracking-wider text-neutral-400">Value</th>
                            <th className="text-left px-3 py-2 text-[10px] uppercase tracking-wider text-neutral-400">Stderr</th>
                          </tr>
                        </thead>
                        <tbody>
                          {metricRows.length === 0 ? (
                            <tr>
                              <td colSpan={4} className="px-3 py-6 text-center text-neutral-400 italic">
                                No metrics available.
                              </td>
                            </tr>
                          ) : (
                            metricRows.map((row, index) => (
                              <tr
                                key={`${row.task}-${row.metric}-${index}`}
                                className="border-b border-neutral-100 last:border-b-0"
                              >
                                <td className="px-3 py-2 text-neutral-700">{row.task}</td>
                                <td className="px-3 py-2 text-neutral-700">{row.metric}</td>
                                <td className="px-3 py-2 text-neutral-900">{metricValueToText(row.value)}</td>
                                <td className="px-3 py-2 text-neutral-500">{metricValueToText(row.stderr)}</td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="flex-1 min-h-0 flex flex-col bg-white">
              <div className="px-6 py-4 border-b border-neutral-100 flex items-center justify-between gap-4">
                <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-widest">Samples</h2>
                <div className="flex items-center gap-3">
                  {availableTasks.length > 1 && (
                    <select
                      value={selectedTask}
                      onChange={event => {
                        setSelectedTask(event.target.value)
                        setSampleOffset(0)
                      }}
                      className="bg-white border border-neutral-200 px-2 py-1 text-xs font-mono text-neutral-600 focus:border-black focus:outline-none"
                    >
                      {availableTasks.map(task => (
                        <option key={task} value={task}>
                          {task}
                        </option>
                      ))}
                    </select>
                  )}
                  <div className="text-[10px] uppercase tracking-wider text-neutral-400">
                    {sampleRangeStart}-{sampleRangeEnd} of {samplesResponse.total}
                  </div>
                  <button
                    onClick={() => setSampleOffset(Math.max(0, sampleOffset - SAMPLE_PAGE_SIZE))}
                    disabled={sampleOffset === 0 || samplesLoading}
                    className={`px-2 py-1 text-[10px] uppercase tracking-wider font-medium border transition-colors ${
                      sampleOffset === 0 || samplesLoading
                        ? 'text-neutral-300 border-neutral-200 cursor-not-allowed'
                        : 'text-neutral-500 border-neutral-200 hover:border-black hover:text-black'
                    }`}
                  >
                    Prev
                  </button>
                  <button
                    onClick={() => setSampleOffset(sampleOffset + SAMPLE_PAGE_SIZE)}
                    disabled={samplesLoading || sampleOffset + SAMPLE_PAGE_SIZE >= samplesResponse.total}
                    className={`px-2 py-1 text-[10px] uppercase tracking-wider font-medium border transition-colors ${
                      samplesLoading || sampleOffset + SAMPLE_PAGE_SIZE >= samplesResponse.total
                        ? 'text-neutral-300 border-neutral-200 cursor-not-allowed'
                        : 'text-neutral-500 border-neutral-200 hover:border-black hover:text-black'
                    }`}
                  >
                    Next
                  </button>
                </div>
              </div>

              <div className="flex-1 min-h-0 overflow-auto p-4 space-y-3 bg-neutral-50/30 scrollbar-thin">
                {!selectedTask ? (
                  <div className="text-xs text-neutral-400 italic">No task selected for samples.</div>
                ) : samplesLoading ? (
                  <div className="text-xs text-neutral-400 italic">Loading samples...</div>
                ) : samplesError ? (
                  <div className="text-xs text-red-600 font-mono">{samplesError}</div>
                ) : samplesResponse.samples.length === 0 ? (
                  <div className="text-xs text-neutral-400 italic">No samples available.</div>
                ) : (
                  samplesResponse.samples.map((sample, index) => {
                    const badges = extractSampleBadges(sample)
                    const docId = sample.doc_id ?? sample.doc_hash ?? `${samplesResponse.offset + index}`

                    return (
                      <div key={`${samplesResponse.offset}-${index}`} className="border border-neutral-200 bg-white p-3 space-y-2">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-[10px] uppercase tracking-wider text-neutral-400">
                            doc_id
                            <span className="ml-1 text-neutral-700 font-mono normal-case">{valueToText(docId)}</span>
                          </div>
                          <div className="flex flex-wrap justify-end gap-1">
                            {badges.map(([key, value]) => (
                              <span
                                key={key}
                                className="px-1.5 py-0.5 text-[10px] font-mono border border-neutral-200 bg-neutral-50 text-neutral-600"
                                title={`${key}: ${valueToText(value)}`}
                              >
                                {key}: {compactValue(value)}
                              </span>
                            ))}
                          </div>
                        </div>

                        <div className="space-y-2">
                          <div>
                            <div className="text-[10px] uppercase tracking-wider text-neutral-400 mb-1">Input</div>
                            <pre className="whitespace-pre-wrap break-words text-xs font-mono text-neutral-700 border border-neutral-200 bg-neutral-50 p-2">
                              {valueToText(sample.input) || 'N/A'}
                            </pre>
                          </div>

                          <div>
                            <div className="text-[10px] uppercase tracking-wider text-neutral-400 mb-1">Target</div>
                            <pre className="whitespace-pre-wrap break-words text-xs font-mono text-green-900 border border-green-200 bg-green-50 p-2">
                              {valueToText(sample.target) || 'N/A'}
                            </pre>
                          </div>

                          <div>
                            <div className="text-[10px] uppercase tracking-wider text-neutral-400 mb-1">Filtered Responses</div>
                            <pre className="whitespace-pre-wrap break-words text-xs font-mono text-blue-900 border border-blue-200 bg-blue-50 p-2">
                              {filteredResponsesText(sample.filtered_resps) || 'N/A'}
                            </pre>
                          </div>
                        </div>
                      </div>
                    )
                  })
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
