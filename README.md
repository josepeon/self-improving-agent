# Self-Improving Agent (SIA)

Infrastructure for training, evaluating, and continuously improving a local AI model for the **OH** platform. The system fine-tunes an Apple-Silicon-native MLX model with LoRA and uses a self-play feedback loop (Groq-as-judge) to generate its own training data.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Pipeline                         │
│            (self_improve.py orchestrator)             │
│                                                      │
│  ┌────────┐  ┌───────────┐  ┌──────────┐  ┌───────┐│
│  │Logging │→ │  Dataset   │→ │FineTuner │→ │ Eval  ││
│  │Storage │  │  Builder   │  │  (LoRA)  │  │Harness││
│  └────────┘  └───────────┘  └──────────┘  └───────┘│
│       ↑                          ↑                   │
│  ┌────────────┐          ┌──────────────┐           │
│  │ Interaction │          │   Model      │           │
│  │ Middleware  │          │  Registry    │           │
│  └────────────┘          └──────────────┘           │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │           Provider Layer (Router)             │   │
│  │   MLXProvider ←──→ GroqProvider (fallback)    │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │          RAG Layer (Retrieval)                 │   │
│  │   ChromaDB ── Embeddings ── Chunker           │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

## Features

- **Provider Abstraction** — Unified interface (`BaseModelProvider`) for any LLM backend. Ships with `GroqProvider` (cloud) and `MLXProvider` (local Apple Silicon).
- **Intelligent Routing** — `ModelRouter` routes requests based on complexity analysis, quality mode, and automatic fallback.
- **Structured Logging** — Every interaction is captured (`InteractionLog`) with full metadata for training signal extraction.
- **Evaluation Harness** — Golden-test evaluation with pluggable rubrics and scoring to gate model promotions.
- **RAG Layer** — ChromaDB vector store + sentence-transformers embeddings + recursive text chunker for knowledge retrieval.
- **LoRA Fine-Tuning** — MLX-native LoRA training with gradient checkpointing, configurable rank/alpha, and adapter persistence.
- **Model Registry** — Version tracking with status lifecycle (candidate → staging → production → retired).
- **Self-Improvement Pipeline** — Orchestrates the full loop: aggregate logs → build dataset → train → evaluate → promote.
- **Self-Play Conversations** — Local model generates multi-turn conversations; Groq (70B) judges quality and only high-scoring conversations become training data.

## Project Structure

```
self-improving-agent/
├── src/self_improving_agent/
│   ├── providers/          # Model provider abstraction
│   │   ├── base.py         # BaseModelProvider, GenerationConfig, ProviderResponse
│   │   ├── groq_provider.py
│   │   ├── mlx_provider.py
│   │   └── router.py       # ModelRouter with complexity analysis
│   ├── logging/            # Interaction logging
│   │   ├── interaction_log.py
│   │   ├── middleware.py
│   │   └── storage.py      # JSONL-based log storage
│   ├── evaluation/         # Model evaluation
│   │   ├── harness.py      # EvaluationHarness
│   │   ├── models.py       # EvaluationReport, TestCase, etc.
│   │   └── golden_tests/   # Golden test YAML files
│   ├── retrieval/          # RAG layer
│   │   ├── chunker.py      # RecursiveTextChunker
│   │   ├── embeddings.py   # SentenceTransformer wrapper
│   │   ├── retriever.py    # RAGRetriever
│   │   └── store.py        # ChromaVectorStore
│   ├── training/           # Fine-tuning infrastructure
│   │   ├── dataset_builder.py
│   │   ├── finetune.py     # FineTuner, LoRAConfig
│   │   └── registry.py     # ModelRegistry, ModelVersion
│   ├── pipeline/           # Orchestration
│   │   └── self_improve.py # SelfImprovementPipeline
│   └── cli.py              # CLI entry point
├── scripts/                # Standalone operational scripts
│   ├── ingest_oh_docs.py           # Ingest OH docs into ChromaDB
│   ├── generate_training_data.py   # Generate single-turn SFT pairs via Groq
│   ├── train_oh_model.py           # Train v1.0-oh (initial LoRA fine-tune)
│   ├── generate_conversations.py   # Self-play multi-turn conversation generator
│   └── train_oh_model_v11.py       # Train v1.1-oh (SFT + conversations)
├── tests/                  # 291+ tests, 92%+ coverage
├── data/                   # (gitignored) Training data and knowledge store
│   ├── oh_knowledge/       # ChromaDB persistent store
│   └── training/           # JSONL training datasets
├── models/                 # (gitignored) Adapters and registry
│   ├── finetuned/          # LoRA adapters per version
│   └── registry.json       # Model version registry
├── .github/workflows/ci.yml
├── pyproject.toml
└── .env.example
```

## Requirements

- **Python** ≥ 3.12
- **macOS** with Apple Silicon (for MLX local inference/training)
- **Groq API key** (Dev tier or above) for cloud inference and data generation

## Installation

```bash
# Clone
git clone https://github.com/oh-systems/self-improving-agent.git
cd self-improving-agent

# Create environment (conda recommended)
conda create -n sia python=3.12 -y
conda activate sia

# Install with all extras
pip install -e ".[dev,rag,mlx]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

### Dependency Groups

| Extra | Packages | Purpose |
|-------|----------|---------|
| *(core)* | groq, aiofiles, pyyaml, pydantic, httpx | Base functionality |
| `[mlx]` | mlx, mlx-lm | Local Apple Silicon inference & training |
| `[rag]` | chromadb, sentence-transformers | Knowledge retrieval |
| `[dev]` | pytest, pytest-asyncio, pytest-cov, ruff, mypy | Development & testing |

## Quick Start

### 1. Ingest Knowledge Base

```bash
# Ingest OH documentation into ChromaDB
python scripts/ingest_oh_docs.py
# → 1,211 chunks from 34 files into data/oh_knowledge/
```

### 2. Generate Training Data

```bash
# Generate single-turn Q&A pairs via Groq
python scripts/generate_training_data.py --pairs 2500
# → data/training/oh_sft_v1.jsonl
```

### 3. Train v1.0

```bash
# Fine-tune the local model with LoRA
python scripts/train_oh_model.py
# → models/finetuned/v1.0-oh/adapters/
# ~7h on M3 Max, 6.7 GB peak memory, final loss ~0.09
```

### 4. Generate Conversations (Self-Play)

```bash
# Local model plays O-OH, Groq simulates users & judges quality
python scripts/generate_conversations.py --rounds 200 --resume
# → data/training/oh_conversations_v1.jsonl
# Only conversations scoring ≥ 7/10 are kept
```

### 5. Train v1.1

```bash
# Merge SFT + conversation data and train next version
python scripts/train_oh_model_v11.py
# → models/finetuned/v1.1-oh/adapters/
```

## Development

### Running Tests

```bash
# Full test suite
pytest

# With coverage
pytest --cov=src/self_improving_agent --cov-fail-under=80

# Skip slow/integration tests (CI default)
pytest -m "not slow and not integration"
```

### Linting & Formatting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### CI

GitHub Actions runs on every push/PR to `main`:
- Ruff lint + format check
- Pytest with 80% coverage gate
- Python 3.12 on ubuntu-latest

## Training Details

### v1.0-oh (Initial Fine-Tune)

| Parameter | Value |
|-----------|-------|
| Base model | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| Method | LoRA (rank 16, alpha 32) |
| Target modules | `self_attn.{q,k,v,o}_proj` |
| Training data | 2,496 single-turn SFT pairs |
| Batch size | 1 (grad accumulation 8) |
| Learning rate | 1e-4 (cosine decay) |
| Iterations | 7,488 (3 epochs) |
| Final loss | 0.091 |
| Peak memory | 6.7 GB |

### v1.1-oh (Planned)

Combines v1.0 SFT data with self-play conversations filtered by Groq-as-judge (threshold ≥ 7/10).

### Self-Play Pipeline

1. Sample a knowledge chunk from ChromaDB
2. Groq generates a user persona and opening message
3. Local model (O-OH) responds with adapter-loaded inference
4. Groq generates follow-up as the user persona
5. Repeat for 3-5 turns
6. Groq judges conversation quality on a 1-10 scale
7. Keep only conversations scoring ≥ 7

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model for inference |
| `MLX_MODEL` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | Local MLX model |
| `SIA_ROUTING_MODE` | `auto` | Routing: `auto`, `local-only`, `remote-only` |
| `SIA_LOG_DIR` | `logs` | Interaction log directory |
| `SIA_LOG_LEVEL` | `INFO` | Logging level |
| `SIA_EVAL_TESTS_DIR` | `evaluation/golden_tests` | Golden test directory |

## License

MIT
