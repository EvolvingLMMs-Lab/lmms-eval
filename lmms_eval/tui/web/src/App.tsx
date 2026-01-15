import { useState, useEffect, useRef } from 'react'

const API_BASE = '/api'

interface ModelInfo {
  id: string
  name: string
}

interface TaskInfo {
  id: string
  name: string
  group: boolean
}

interface Config {
  model: string
  model_args: string
  tasks: string[]
  batch_size: number
  limit: number | null
  output_path: string
  log_samples: boolean
  verbosity: string
  device: string | null
}

type Status = 'ready' | 'running' | 'stopped' | 'completed' | 'error'

export default function App() {
  const [version, setVersion] = useState('...')
  const [models, setModels] = useState<ModelInfo[]>([])
  const [tasks, setTasks] = useState<TaskInfo[]>([])
  
  const [model, setModel] = useState('openai_compatible')
  const [modelArgs, setModelArgs] = useState('model_version=allenai/molmo-2-8b:free')
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set())
  const [taskFilter, setTaskFilter] = useState('')
  const [batchSize, setBatchSize] = useState('1')
  const [limit, setLimit] = useState('')
  const [device, setDevice] = useState('')
  const [outputPath, setOutputPath] = useState('./logs/')
  const [verbosity, setVerbosity] = useState('INFO')
  
  const [status, setStatus] = useState<Status>('ready')
  const [jobId, setJobId] = useState<string | null>(null)
  const [output, setOutput] = useState<string[]>(['Ready to evaluate'])
  const [command, setCommand] = useState('')
  
  const outputRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => setVersion(d.version))
      .catch(() => setVersion('error'))
    
    fetch(`${API_BASE}/models`)
      .then(r => r.json())
      .then(setModels)
      .catch(() => setModels([]))
    
    fetch(`${API_BASE}/tasks`)
      .then(r => r.json())
      .then(setTasks)
      .catch(() => setTasks([]))
  }, [])

  useEffect(() => {
    const config: Config = {
      model,
      model_args: modelArgs,
      tasks: Array.from(selectedTasks),
      batch_size: parseInt(batchSize) || 1,
      limit: limit ? parseInt(limit) : null,
      output_path: outputPath,
      log_samples: true,
      verbosity,
      device: device || null,
    }
    
    fetch(`${API_BASE}/eval/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
      .then(r => r.json())
      .then(d => setCommand(d.command))
      .catch(() => setCommand('# Error generating command'))
  }, [model, modelArgs, selectedTasks, batchSize, limit, device, outputPath, verbosity])

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [output])

  const filteredTasks = tasks.filter(t => 
    t.id.toLowerCase().includes(taskFilter.toLowerCase()) ||
    t.name.toLowerCase().includes(taskFilter.toLowerCase())
  )

  const toggleTask = (taskId: string) => {
    const newSet = new Set(selectedTasks)
    if (newSet.has(taskId)) {
      newSet.delete(taskId)
    } else {
      newSet.add(taskId)
    }
    setSelectedTasks(newSet)
  }

  const startEval = async () => {
    if (selectedTasks.size === 0) {
      setOutput(['Error: No tasks selected'])
      return
    }
    
    setStatus('running')
    setOutput(['Starting evaluation...'])
    
    const config: Config = {
      model,
      model_args: modelArgs,
      tasks: Array.from(selectedTasks),
      batch_size: parseInt(batchSize) || 1,
      limit: limit ? parseInt(limit) : null,
      output_path: outputPath,
      log_samples: true,
      verbosity,
      device: device || null,
    }
    
    try {
      const res = await fetch(`${API_BASE}/eval/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      const data = await res.json()
      setJobId(data.job_id)
      
      const eventSource = new EventSource(`${API_BASE}/eval/${data.job_id}/stream`)
      
      eventSource.onmessage = (event) => {
        const d = JSON.parse(event.data)
        
        if (d.type === 'output') {
          setOutput(prev => [...prev, d.line])
        } else if (d.type === 'done') {
          setStatus(d.exit_code === 0 ? 'completed' : 'error')
          setOutput(prev => [...prev, '', `Evaluation ${d.exit_code === 0 ? 'completed' : 'failed'} (exit: ${d.exit_code})`])
          eventSource.close()
        } else if (d.type === 'stopped') {
          setStatus('stopped')
          setOutput(prev => [...prev, '', 'Evaluation stopped'])
          eventSource.close()
        } else if (d.type === 'error') {
          setOutput(prev => [...prev, `Error: ${d.message}`])
          setStatus('error')
          eventSource.close()
        }
      }
      
      eventSource.onerror = () => {
        setStatus('error')
        setOutput(prev => [...prev, 'Connection error'])
        eventSource.close()
      }
    } catch (e) {
      setOutput([`Failed to start: ${e}`])
      setStatus('error')
    }
  }

  const stopEval = async () => {
    if (!jobId) return
    try {
      await fetch(`${API_BASE}/eval/${jobId}/stop`, { method: 'POST' })
      setStatus('stopped')
    } catch (e) {
      setOutput(prev => [...prev, `Failed to stop: ${e}`])
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-2xl font-bold text-cyan-400">LMMs-Eval</div>
            <span className="text-sm text-gray-400">v{version}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className={`px-3 py-1 rounded-full text-sm ${
              status === 'ready' ? 'bg-gray-700 text-gray-300' :
              status === 'running' ? 'bg-blue-900 text-blue-300' :
              status === 'completed' ? 'bg-green-900 text-green-300' :
              status === 'error' ? 'bg-red-900 text-red-300' :
              'bg-yellow-900 text-yellow-300'
            }`}>
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </span>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-73px)]">
        <div className="w-80 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Configuration</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Model</label>
              <select 
                value={model} 
                onChange={e => setModel(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
              >
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Model Arguments</label>
              <textarea 
                value={modelArgs}
                onChange={e => setModelArgs(e.target.value)}
                placeholder="model_version=gpt-4o,pretrained=..."
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm h-20 resize-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Batch Size</label>
                <input 
                  type="number"
                  value={batchSize}
                  onChange={e => setBatchSize(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Limit</label>
                <input 
                  type="number"
                  value={limit}
                  onChange={e => setLimit(e.target.value)}
                  placeholder="All"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Device</label>
              <input 
                value={device}
                onChange={e => setDevice(e.target.value)}
                placeholder="cuda:0 (auto if empty)"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Verbosity</label>
              <select 
                value={verbosity}
                onChange={e => setVerbosity(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
              >
                <option value="DEBUG">DEBUG</option>
                <option value="INFO">INFO</option>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Output Path</label>
              <input 
                value={outputPath}
                onChange={e => setOutputPath(e.target.value)}
                placeholder="./logs/"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="mt-6 flex gap-2">
            <button
              onClick={startEval}
              disabled={status === 'running'}
              className={`flex-1 py-2.5 rounded-lg font-medium transition ${
                status === 'running'
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-500 text-white'
              }`}
            >
              {status === 'running' ? 'Running...' : 'Start'}
            </button>
            <button
              onClick={stopEval}
              disabled={status !== 'running'}
              className={`flex-1 py-2.5 rounded-lg font-medium transition ${
                status !== 'running'
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : 'bg-red-600 hover:bg-red-500 text-white'
              }`}
            >
              Stop
            </button>
          </div>
        </div>

        <div className="w-96 border-r border-gray-700 flex flex-col">
          <div className="p-4 border-b border-gray-700">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Tasks ({selectedTasks.size} selected)
            </h2>
            <input
              value={taskFilter}
              onChange={e => setTaskFilter(e.target.value)}
              placeholder="Filter tasks..."
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
            />
          </div>
          <div className="flex-1 overflow-y-auto p-2 scrollbar-thin">
            {filteredTasks.map(task => (
              <div
                key={task.id}
                onClick={() => toggleTask(task.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition ${
                  selectedTasks.has(task.id)
                    ? 'bg-cyan-900/50 text-cyan-300'
                    : 'hover:bg-gray-700/50 text-gray-300'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedTasks.has(task.id)}
                  onChange={() => {}}
                  className="rounded bg-gray-600 border-gray-500 text-cyan-500 focus:ring-cyan-500"
                />
                <span className="text-xs text-gray-500">{task.group ? '📁' : '📄'}</span>
                <span className="text-sm truncate">{task.id}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="flex-1 flex flex-col">
          <div className="h-1/3 border-b border-gray-700 flex flex-col">
            <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Command Preview</h2>
              <button 
                onClick={() => navigator.clipboard.writeText(command)}
                className="text-xs text-gray-400 hover:text-white px-2 py-1 rounded hover:bg-gray-700"
              >
                Copy
              </button>
            </div>
            <pre className="flex-1 overflow-auto p-4 text-sm text-yellow-300 bg-gray-800/50 font-mono scrollbar-thin">
              {command}
            </pre>
          </div>

          <div className="flex-1 flex flex-col">
            <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Output</h2>
              <button 
                onClick={() => setOutput([])}
                className="text-xs text-gray-400 hover:text-white px-2 py-1 rounded hover:bg-gray-700"
              >
                Clear
              </button>
            </div>
            <div ref={outputRef} className="flex-1 overflow-auto p-4 bg-black/30 font-mono text-sm scrollbar-thin">
              {output.map((line, i) => (
                <div key={i} className="text-gray-300 whitespace-pre-wrap">{line}</div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
