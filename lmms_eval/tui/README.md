# LMMS-Eval Web UI

Web-based User Interface for LMMS-Eval built with React + Vite + Tailwind CSS and FastAPI.

## Architecture

- **Backend**: FastAPI (Python) - handles model/task discovery, evaluation runs
- **Frontend**: React + Vite + Tailwind CSS - modern web UI

## Requirements

- Python 3.10+
- Node.js 18+ (for building the frontend)

## Installation

The web UI will be automatically built on first run. Or build manually:

```bash
cd lmms_eval/tui/web
npm install
npm run build
```

## Usage

### Quick Start

```bash
uv run lmms-eval-ui
```

This starts the server on http://localhost:8000 and opens your browser.

### Manual Startup

```bash
# Start server only
uv run uvicorn lmms_eval.tui.server:app --host 0.0.0.0 --port 8000

# Then open http://localhost:8000 in your browser
```

### Custom Port

```bash
LMMS_SERVER_PORT=3000 uv run lmms-eval-ui
```

## Features

- Model selection from all available models
- Task selection with search/filter
- Real-time command preview
- Live evaluation output streaming
- Start/Stop evaluation controls
- Configuration: batch size, limit, device, verbosity

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/models` | GET | List available models |
| `/tasks` | GET | List available tasks |
| `/eval/preview` | POST | Generate command preview |
| `/eval/start` | POST | Start evaluation |
| `/eval/{job_id}/stream` | GET | Stream evaluation output (SSE) |
| `/eval/{job_id}/stop` | POST | Stop evaluation |

## File Structure

```
lmms_eval/tui/
├── __init__.py        # Python exports
├── cli.py             # lmms-eval-ui entry point
├── server.py          # FastAPI server
├── discovery.py       # Model/task discovery
├── README.md          # This file
└── web/               # React frontend
    ├── src/
    │   ├── App.tsx    # Main React component
    │   ├── main.tsx   # Entry point
    │   └── index.css  # Tailwind CSS
    ├── package.json
    ├── vite.config.ts
    └── dist/          # Built static files
```

## Development

For frontend development with hot reload:

```bash
# Terminal 1: Start backend server
uv run uvicorn lmms_eval.tui.server:app --port 8000

# Terminal 2: Start Vite dev server
cd lmms_eval/tui/web
npm run dev
```

Then open http://localhost:5173 (Vite proxies API requests to :8000)
