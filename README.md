# Aperture

Aperture is the foundational backend for an agentic AI system. Currently at **Checkpoint 07 of 15** in its build phase, this repository houses a high-performance, asynchronous FastAPI web server designed to handle real-time streaming LLM responses and robust tool-calling capabilities.

This backend is built to bypass heavy abstractions (like LangChain) in favor of native SDKs, strong data contracts, and raw speed.

## 🚀 Features

* **Full ReAct (Reason + Act) Loop:** The core agent runs on an autonomous `Think -> Act -> Observe` cycle. It can execute multi-turn workflows (e.g., scrape a page, read the result, realize it needs to use a math tool, and loop again) until the task is complete, protected by a max-iteration safety cap.
* **Resilient Tool Execution Engine:** Tool dispatching is wrapped in a `safe_dispatch` layer that provides 30-second timeout guards, catches unhandled exceptions, and enforces a bounded retry policy (max 2 identical attempts) to prevent infinite loops while allowing recovery from transient network errors.
* **Interleaved SSE Streaming:** Utilizes Server-Sent Events (`text/event-stream`) with a custom chunk accumulator. The API streams real-time `text_delta` chunks to the user instantly, while silently trapping and reassembling JSON tool fragments in the background, allowing for seamless UI rendering during complex executions.
* **Persistent Agent Memory:** Utilizes `SQLAlchemy` (with the `asyncpg` driver) to automatically log every `Session`, `Message`, and `ToolCall` into PostgreSQL, providing a complete, queryable audit trail of the agent's actions inside the ReAct loop.
* **Regex-Powered Defensive Traps:** Built-in extraction traps use `re.search` to catch and parse rogue JSON tool calls from open-source models, even when the model wraps its JSON in conversational filler text. 
* **Automated Schema Migrations:** Fully integrated `Alembic` environment configured for asynchronous execution to manage database evolution without breaking the containerized stack.
* **Multi-Tool Dispatch Registry:** Features a custom, asynchronous tool registry that parses LLM intentions, dynamically matches requests to active functions, validates JSON arguments, and runs tool code seamlessly.
* **Headless DOM-to-Markdown Extraction:** Integrates automated browser instances to load JavaScript-rendered components, bypass basic bot blocks via custom headers, and process heavy raw DOM footprints into clean, context-efficient Markdown text.
* **Containerized Infrastructure:** A unified `docker-compose` stack running a hot-reloading Linux API container, PostgreSQL 16, and Redis 7 on an isolated internal network.
* **OpenAI-Compatible Architecture:** Built with the `openai` Python SDK, seamlessly supporting local models (like Qwen via Ollama) and production APIs with zero code changes.

## 📁 Repository Structure

The project is modularized to support independent tools, data schemas, and routers.

```text
aperture/
├── core/
│   ├── config.py        # Pydantic environment validation
│   ├── database.py      # Async Postgres engine and dependency injection
│   └── llm.py           # OpenAI client Singleton initialization
├── migrations/          # Alembic asynchronous migration scripts and environment
├── models/
│   ├── chat.py          # Pydantic Request/Response schemas
│   └── db.py            # SQLAlchemy DeclarativeBase ORM models
├── routers/             
│   └── agent.py         # ReAct loop, chunk accumulator, safe dispatch, and SSE stream
├── tools/
│   ├── __init__.py      
│   ├── functions.py     # Executable Python functions (Playwright extraction, string math)
│   ├── registry.py      # Async tool dispatcher map
│   └── schemas.py       # JSON schemas representing the tool menu to the LLM
├── .env                 # Secrets and routing config (git-ignored)
├── .env.example         # Template for environment variables
├── .gitignore           # Ignored files and directories
├── .python-version      # Defined python version for uv
├── alembic.ini          # Alembic configuration for migrations
├── docker-compose.yml   # Infrastructure orchestration (API, Postgres, Redis)
├── Dockerfile           # Python 3.12 Linux environment with embedded Playwright binaries
├── main.py              # Application entry point and router inclusion
├── pyproject.toml       # Project metadata
└── uv.lock              # Dependency lockfile
```

## 🛠️ Setup & Execution

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/Aperture.git](https://github.com/yourusername/Aperture.git)
cd Aperture
```

### 2. Set up the environment
Create a `.env` file in the root directory by copying the example:
```bash
cp .env.example .env
```
Ensure your database credentials and LLM endpoints are configured:
```env
# Example configuration for a local Ollama tunnel
OPENAI_API_BASE_URL=[https://your-ngrok-url.ngrok-free.app/v1](https://your-ngrok-url.ngrok-free.app/v1)
OPENAI_API_KEY=not_needed_for_local
LLM_MODEL_NAME=qwen2.5-coder:7b

# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=aperture
```

### 3. Boot the Infrastructure (Recommended)
Aperture is designed to run inside Docker to ensure environment parity. This spins up the API, PostgreSQL, and Redis together.
```bash
docker compose up --build
```
*Note: The local directory is mounted as a volume. Editing code in your IDE will instantly hot-reload the containerized API.*

### Alternative: Local Host Execution
If you prefer running without Docker, use `uv` for dependency management:
```bash
uv add playwright html2text sqlalchemy alembic asyncpg
uv run playwright install chromium
uv run uvicorn main:app --reload
```

## 🧪 Testing the Agent

Test the multi-turn ReAct capabilities using `curl`. Note the addition of the `-N` flag, which disables curl's buffering so you can watch the Server-Sent Events (SSE) `text_delta` stream live in your terminal.

**Test 1: Core Knowledge Conversing (No Tools)**
```bash
curl -N -X POST "http://localhost:8000/agent/run" \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"What is the core difference between synchronous and asynchronous code execution?\"}"
```

**Test 2: Internal Python Function Execution (String Length Tool)**
```bash
curl -N -X POST "http://localhost:8000/agent/run" \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"Can you calculate the length of this string for me: 'The quick brown fox jumps over the lazy dog'?\"}"
```

**Test 3: The Full ReAct Loop (Multi-Turn Execution)**
```bash
curl -N -X POST "http://localhost:8000/agent/run" \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"Go to news.ycombinator.com, find the title of the top post, and calculate its string length.\"}"
```

## 🗺️ Roadmap
- [x] **01** FastAPI Application Foundation
- [x] **02** Manual Tool Dispatch System
- [x] **03** Two-Tool Agent Loop
- [x] **04** `dom_to_markdown` via Playwright
- [x] **05** Docker Compose Stack (Postgres + Redis)
- [x] **06** Postgres Models + Alembic Migrations
- [x] **07** Full ReAct Agent Loop
- [ ] **08** pgvector Semantic Memory
- [ ] **09** Workflow Graph + Deterministic Replay
- [ ] **10** Redis Task Queue + SSE
- [ ] **11** Next.js Agent UI
- [ ] **12** Memory Explorer + Replay UI
- [ ] **13** Agent Control Interface
- [ ] **14** Tests + CI with GitHub Actions
- [ ] **15** Cloudflare Worker Deployment
