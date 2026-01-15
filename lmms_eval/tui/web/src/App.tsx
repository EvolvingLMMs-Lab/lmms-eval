import { useState, useEffect, useRef, useMemo } from 'react'

const API_BASE = ''

interface SelectProps {
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
  placeholder?: string
}

function Select({ value, onChange, options, placeholder }: SelectProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const selectedOption = options.find(o => o.value === value)

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between bg-white border border-neutral-200 px-3 py-2 text-sm focus:border-black focus:outline-none transition-colors text-left text-neutral-900 hover:border-neutral-300"
      >
        <span className={selectedOption ? 'text-neutral-900' : 'text-neutral-400'}>
          {selectedOption?.label || placeholder || 'Select...'}
        </span>
        <span className={`text-[10px] text-neutral-400 transition-transform ${open ? 'rotate-180' : ''}`}>▼</span>
      </button>
      {open && (
        <div className="absolute z-50 left-0 right-0 mt-1 bg-white border border-neutral-200 shadow-lg max-h-60 overflow-auto">
          {options.map(option => (
            <div
              key={option.value}
              onClick={() => {
                onChange(option.value)
                setOpen(false)
              }}
              className={`px-3 py-2 text-sm cursor-pointer transition-colors ${
                option.value === value
                  ? 'bg-black text-white'
                  : 'text-neutral-900 hover:bg-neutral-50'
              }`}
            >
              {option.label}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

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
  env_vars: string
  batch_size: number
  limit: number | null
  output_path: string
  log_samples: boolean
  verbosity: string
  device: string | null
}

type Status = 'ready' | 'running' | 'stopped' | 'completed' | 'error'

interface GitInfo {
  branch: string
  commit: string
}

interface SysInfo {
  hostname: string
  cwd: string
  repo_root?: string
}

type TaskNode = 
  | { type: 'group', id: string, label: string, children: TaskInfo[] }
  | { type: 'leaf', task: TaskInfo }

export default function App() {
  const [version, setVersion] = useState('...')
  const [gitInfo, setGitInfo] = useState<GitInfo>({ branch: '', commit: '' })
  const [sysInfo, setSysInfo] = useState<SysInfo>({ hostname: '', cwd: '' })
  const [models, setModels] = useState<ModelInfo[]>([])
  const [tasks, setTasks] = useState<TaskInfo[]>([])
  

  const [model, setModel] = useState('openai_compatible')
  const [modelArgs, setModelArgs] = useState('model_version=allenai/molmo-2-8b:free')
  const [envVars, setEnvVars] = useState(
    'export HF_HOME=${HF_HOME:-~/.cache/huggingface}\n' +
      'export OPENAI_API_KEY=${OPENROUTER_API_KEY}\n' +
      'export OPENAI_API_BASE=https://openrouter.ai/api/v1'
  )
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set(['mme']))
  const [taskFilter, setTaskFilter] = useState('')
  const [batchSize, setBatchSize] = useState('1')
  const [limit, setLimit] = useState('5')
  const [device, setDevice] = useState('')
  const [outputPath, setOutputPath] = useState('./logs/openrouter_test/')
  const [verbosity, setVerbosity] = useState('DEBUG')
  
  const [status, setStatus] = useState<Status>('ready')
  const [jobId, setJobId] = useState<string | null>(null)
  const [output, setOutput] = useState<string[]>(['Ready to evaluate'])
  const [command, setCommand] = useState('')
  
  const outputRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => {
        setVersion(d.version)
        if (d.git) setGitInfo(d.git)
        if (d.system) setSysInfo(d.system)
      })
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
      env_vars: envVars,
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
  }, [model, modelArgs, selectedTasks, envVars, batchSize, limit, device, outputPath, verbosity])

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [output])

  const visibleNodes = useMemo(() => {
    const nodes: TaskNode[] = []
    
    const allGroups = tasks.filter(t => t.group)
    const allLeaves = tasks.filter(t => !t.group)

    const filteredLeaves = allLeaves.filter(t => 
      t.id.toLowerCase().includes(taskFilter.toLowerCase()) ||
      t.name.toLowerCase().includes(taskFilter.toLowerCase())
    )

    const groupChildrenMap = new Map<string, TaskInfo[]>()
    const assignedLeafIds = new Set<string>()

    for (const group of allGroups) {
      const children = filteredLeaves.filter(leaf => 
        leaf.id.startsWith(`${group.id}_`) || leaf.id.startsWith(`${group.id}-`)
      )
      
      if (children.length > 0) {
        groupChildrenMap.set(group.id, children)
        children.forEach(c => assignedLeafIds.add(c.id))
        nodes.push({
          type: 'group',
          id: group.id,
          label: group.id, 
          children: children
        })
      }
    }

    const topLevelLeaves = filteredLeaves.filter(leaf => !assignedLeafIds.has(leaf.id))
    topLevelLeaves.forEach(leaf => {
      nodes.push({ type: 'leaf', task: leaf })
    })

    nodes.sort((a, b) => {
      const idA = a.type === 'group' ? a.id : a.task.id
      const idB = b.type === 'group' ? b.id : b.task.id
      return idA.localeCompare(idB)
    })

    return nodes
  }, [tasks, taskFilter])

  const invertSelection = () => {
    const newSet = new Set(selectedTasks)
    
    const visibleLeafIds: string[] = []
    visibleNodes.forEach(node => {
      if (node.type === 'leaf') {
        visibleLeafIds.push(node.task.id)
      } else {
        node.children.forEach(c => visibleLeafIds.push(c.id))
      }
    })

    visibleLeafIds.forEach(id => {
      if (newSet.has(id)) {
        newSet.delete(id)
      } else {
        newSet.add(id)
      }
    })
    
    setSelectedTasks(newSet)
  }

  const toggleTask = (taskId: string) => {
    const newSet = new Set(selectedTasks)
    if (newSet.has(taskId)) {
      newSet.delete(taskId)
    } else {
      newSet.add(taskId)
    }
    setSelectedTasks(newSet)
  }

  const toggleGroup = (children: TaskInfo[]) => {
    const newSet = new Set(selectedTasks)
    const allSelected = children.every(c => newSet.has(c.id))
    
    if (allSelected) {
      children.forEach(c => newSet.delete(c.id))
    } else {
      children.forEach(c => newSet.add(c.id))
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
      env_vars: envVars,
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
    <div className="flex flex-col h-screen bg-white text-neutral-900 font-light selection:bg-black selection:text-white">
      <header className="h-14 flex items-center justify-between px-6 border-b border-neutral-200 bg-white/80 backdrop-blur-md z-10">
        <div className="flex items-center gap-4">
          <div className="text-lg font-bold tracking-tight text-neutral-900">LMMs-Eval</div>
          <div className="flex items-center gap-3 text-[10px] font-mono text-neutral-400">
            <span className="bg-neutral-100 px-1.5 py-0.5 rounded border border-neutral-200 text-neutral-600">v{version}</span>
            {(gitInfo.branch || gitInfo.commit) && (
              <>
                <span className="text-neutral-300">/</span>
                <span className="flex items-center gap-1">
                  {gitInfo.branch && <span>{gitInfo.branch}</span>}
                  {gitInfo.branch && gitInfo.commit && <span className="text-neutral-300">@</span>}
                  {gitInfo.commit && <span>{gitInfo.commit}</span>}
                </span>
              </>
            )}
            {(sysInfo.repo_root || sysInfo.cwd) && (
              <>
                <span className="text-neutral-300">/</span>
                <span className="max-w-[200px] truncate" title={sysInfo.repo_root || sysInfo.cwd}>
                  {sysInfo.repo_root || sysInfo.cwd}
                </span>
              </>
            )}

          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className={`px-2.5 py-0.5 text-[10px] uppercase tracking-wider font-medium border ${
            status === 'ready' ? 'border-neutral-200 text-neutral-400' :
            status === 'running' ? 'border-black text-black animate-pulse' :
            status === 'completed' ? 'border-green-600 text-green-600' :
            status === 'error' ? 'border-red-600 text-red-600' :
            'border-neutral-200 text-neutral-400'
          }`}>
            {status}
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <div className="w-[450px] bg-white border-r border-neutral-200 flex flex-col overflow-y-auto scrollbar-thin">
          <div className="flex-shrink-0 p-6 border-b border-neutral-100">
            <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-widest mb-4">Configuration</h2>
            
            <div className="space-y-4">
              <div className="group">
                <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5">Model</label>
                <Select
                  value={model}
                  onChange={setModel}
                  options={models.map(m => ({ value: m.id, label: m.name }))}
                  placeholder="Select model..."
                />
              </div>

              <div className="group">
                <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Arguments</label>
                <textarea 
                  value={modelArgs}
                  onChange={e => setModelArgs(e.target.value)}
                  placeholder="model_version=..."
                  className="w-full bg-white border border-neutral-200 px-3 py-2 text-sm h-16 resize-none focus:border-black focus:outline-none transition-colors placeholder-neutral-300 leading-relaxed text-neutral-900"
                />
              </div>

              <div className="group">
                <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Environment Variables</label>
                <textarea 
                  value={envVars}
                  onChange={e => setEnvVars(e.target.value)}
                  placeholder="export KEY=VALUE..."
                  className="w-full bg-white border border-neutral-200 px-3 py-2 text-sm h-16 resize-none focus:border-black focus:outline-none transition-colors placeholder-neutral-300 leading-relaxed text-neutral-900 font-mono"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="group">
                  <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Batch Size</label>
                  <input 
                    type="number"
                    value={batchSize}
                    onChange={e => setBatchSize(e.target.value)}
                    className="w-full bg-white border border-neutral-200 px-3 py-2 text-sm focus:border-black focus:outline-none transition-colors text-neutral-900"
                  />
                </div>
                <div className="group">
                  <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Limit</label>
                  <input 
                    type="number"
                    value={limit}
                    onChange={e => setLimit(e.target.value)}
                    placeholder="All"
                    className="w-full bg-white border border-neutral-200 px-3 py-2 text-sm focus:border-black focus:outline-none transition-colors placeholder-neutral-300 text-neutral-900"
                  />
                </div>
              </div>
              
              <div className="group">
                  <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Device</label>
                  <input 
                    value={device}
                    onChange={e => setDevice(e.target.value)}
                    placeholder="cuda:0"
                    className="w-full bg-white border border-neutral-200 px-3 py-2 text-sm focus:border-black focus:outline-none transition-colors placeholder-neutral-300 text-neutral-900"
                  />
              </div>

              <div className="group">
                  <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Output Path</label>
                  <input 
                    value={outputPath}
                    onChange={e => setOutputPath(e.target.value)}
                    placeholder="./logs/"
                    className="w-full bg-white border border-neutral-200 px-3 py-2 text-sm focus:border-black focus:outline-none transition-colors placeholder-neutral-300 text-neutral-900"
                  />
              </div>

              <div className="group">
                <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5">Verbosity</label>
                <Select
                  value={verbosity}
                  onChange={setVerbosity}
                  options={[
                    { value: 'DEBUG', label: 'DEBUG' },
                    { value: 'INFO', label: 'INFO' },
                    { value: 'WARNING', label: 'WARNING' },
                    { value: 'ERROR', label: 'ERROR' },
                  ]}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col min-h-0">
            <div className="p-6 border-b border-neutral-100 bg-white">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-widest">
                  Tasks <span className="text-neutral-900 ml-1">{selectedTasks.size}</span>
                </h2>
              </div>
              <div className="flex gap-2">
                <input
                  value={taskFilter}
                  onChange={e => setTaskFilter(e.target.value)}
                  placeholder="Search tasks..."
                  className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-900"
                />
                <button
                  onClick={invertSelection}
                  className="px-3 py-2 text-xs font-medium uppercase tracking-wider bg-neutral-100 border border-neutral-200 hover:bg-neutral-200 hover:border-neutral-300 transition-colors text-neutral-900"
                  title="Invert selection for filtered tasks"
                >
                  Invert
                </button>
              </div>
            </div>
            <div className="">
              {visibleNodes.map((node) => {
                if (node.type === 'group') {
                  const allChildrenSelected = node.children.every(c => selectedTasks.has(c.id))
                  const someChildrenSelected = node.children.some(c => selectedTasks.has(c.id))
                  
                  return (
                    <div key={node.id} className="border-b border-neutral-50">
                      <div 
                        onClick={() => toggleGroup(node.children)}
                        className="flex items-center gap-3 px-6 py-2 bg-neutral-50/50 cursor-pointer hover:bg-neutral-100 transition-colors"
                      >
                        <div className={`w-3.5 h-3.5 flex items-center justify-center border transition-colors ${
                          allChildrenSelected 
                          ? 'border-black bg-black' 
                          : someChildrenSelected ? 'border-black' : 'border-neutral-300 hover:border-black'
                        }`}>
                          {allChildrenSelected && <div className="w-1.5 h-1.5 bg-white" />}
                          {!allChildrenSelected && someChildrenSelected && <div className="w-1.5 h-1.5 bg-black" />}
                        </div>
                        <span className="text-[10px] uppercase font-bold tracking-wider text-neutral-500">Group</span>
                        <span className="text-xs font-semibold text-neutral-800">{node.id}</span>
                      </div>
                      {node.children.map(child => (
                        <div
                          key={child.id}
                          onClick={() => toggleTask(child.id)}
                          className={`group flex items-center gap-3 pl-10 pr-6 py-2 border-l-4 border-l-transparent hover:border-l-neutral-200 cursor-pointer transition-colors ${
                            selectedTasks.has(child.id)
                              ? 'bg-neutral-50 text-neutral-900'
                              : 'text-neutral-500 hover:bg-neutral-50/50 hover:text-neutral-900'
                          }`}
                        >
                          <div className={`w-3.5 h-3.5 flex items-center justify-center border transition-colors ${
                            selectedTasks.has(child.id) 
                            ? 'border-black bg-black' 
                            : 'border-neutral-300 group-hover:border-black'
                          }`}>
                            {selectedTasks.has(child.id) && <div className="w-1.5 h-1.5 bg-white" />}
                          </div>
                          <span className="text-[10px] uppercase tracking-wider text-neutral-400">Task</span>
                          <span className="text-xs font-medium truncate">{child.id}</span>
                        </div>
                      ))}
                    </div>
                  )
                } else {
                  // Leaf node
                  return (
                    <div
                      key={node.task.id}
                      onClick={() => toggleTask(node.task.id)}
                      className={`group flex items-center gap-3 px-6 py-3 border-b border-neutral-50 cursor-pointer transition-colors duration-200 ${
                        selectedTasks.has(node.task.id)
                          ? 'bg-neutral-50 text-neutral-900'
                          : 'hover:bg-neutral-50 text-neutral-500 hover:text-neutral-900'
                      }`}
                    >
                      <div className={`w-3.5 h-3.5 flex items-center justify-center border transition-colors ${
                        selectedTasks.has(node.task.id) 
                        ? 'border-black bg-black' 
                        : 'border-neutral-300 group-hover:border-black'
                      }`}>
                        {selectedTasks.has(node.task.id) && <div className="w-1.5 h-1.5 bg-white" />}
                      </div>
                      <span className="text-[10px] uppercase tracking-wider text-neutral-400">Task</span>
                      <span className="text-xs font-medium truncate">{node.task.id}</span>
                    </div>
                  )
                }
              })}
            </div>
          </div>
        </div>

        <div className="flex-1 flex flex-col bg-neutral-50/30">
          <div className="px-6 py-4 border-b border-neutral-200 bg-white flex gap-3 justify-start">
            <button
              onClick={startEval}
              disabled={status === 'running'}
              className={`w-40 py-2.5 text-xs font-medium uppercase tracking-wider transition-all duration-200 ${
                status === 'running'
                  ? 'bg-neutral-100 text-neutral-400 cursor-not-allowed border border-neutral-200'
                  : 'bg-black text-white hover:bg-neutral-800 border border-black shadow-sm'
              }`}
            >
              {status === 'running' ? 'Running...' : 'Start'}
            </button>
            <button
              onClick={stopEval}
              disabled={status !== 'running'}
              className={`w-40 py-2.5 text-xs font-medium uppercase tracking-wider transition-all duration-200 ${
                status !== 'running'
                  ? 'bg-transparent text-neutral-300 border border-neutral-200 cursor-not-allowed'
                  : 'bg-white text-neutral-900 border border-neutral-200 hover:border-black shadow-sm'
              }`}
            >
              Stop
            </button>
          </div>

          <div className="h-1/3 border-b border-neutral-200 flex flex-col bg-white">
            <div className="px-6 py-4 border-b border-neutral-100 flex items-center justify-between bg-white">
              <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-widest">Command</h2>
              <button 
                onClick={() => navigator.clipboard.writeText(command)}
                className="text-[10px] text-neutral-400 hover:text-neutral-900 uppercase tracking-wider transition-colors"
              >
                Copy
              </button>
            </div>
            <div className="flex-1 overflow-auto p-6 font-mono text-xs text-neutral-600 bg-neutral-50/50 scrollbar-thin selection:bg-neutral-100">
              <pre className="whitespace-pre-wrap leading-relaxed">{command}</pre>
            </div>
          </div>

          <div className="flex-1 flex flex-col bg-white">
            <div className="px-6 py-4 border-b border-neutral-100 flex items-center justify-between bg-white">
              <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-widest">Log Output</h2>
              <button 
                onClick={() => setOutput([])}
                className="text-[10px] text-neutral-400 hover:text-neutral-900 uppercase tracking-wider transition-colors"
              >
                Clear
              </button>
            </div>
            <div ref={outputRef} className="flex-1 overflow-auto p-6 font-mono text-xs leading-relaxed bg-white text-neutral-600 scrollbar-thin">
              {output.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap mb-1 opacity-90">{line}</div>
              ))}
              {output.length === 0 && (
                <div className="text-neutral-400 italic">Waiting for process...</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
