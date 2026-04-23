# LLM-GQP: LLM-Graph Query Processing

A comprehensive benchmarking pipeline designed to evaluate the reasoning and algorithmic capabilities of Large Language Models (LLMs) on graph theory problems. This project translates standard NetworkX benchmark graphs into various Graph Description Languages (GDLs) and tests LLM accuracy across multiple topological domains (shortest path, connectivity, clustering, density, etc.).

## Key Features

- **Multi-Provider Architecture**: Native, auto-detecting support for **OpenAI**, **Gemini**, **Groq**, and **Ollama** (local) via LangChain.
- **Matrix Benchmarking**: Automated evaluation sweep across directed, undirected, weighted, and unweighted graphs using `benchmark.sh`.
- **Robust XML Output Parsing**: Strict `<answer>` tag extraction with built-in fallbacks and prose-stripping to handle verbose or malformed LLM responses.
- **Precision-Tolerant Evaluation**: Advanced metrics engine (`eval.py`) that handles LLM rounding behaviors (e.g., matching float predictions with a `1e-2` tolerance).
- **7 Graph Description Languages (GDL)**:
  - Natural Language (Text)
  - Adjacency List
  - Edge List
  - GML
  - GraphML
  - Random Walk
  - L2SP Paths

## Project Structure

- `orchestrator.py`: The CLI entry point. Generates graphs, extracts GDL descriptors, and orchestrates the LLM benchmarking.
- `llm.py`: Provider-agnostic LLM interface. Handles API connections, prompt injection, dynamic token budgeting (2048 limits), and execution state (resume capability).
- `prompts.py`: Defines the strict role, formatting rules, and robust response parsers.
- `eval.py`: The metrics engine. Evaluates predicted answers against ground truth.
- `benchmark.sh`: Shell script to run a full Cartesian product sweep (Graph Types × Models).
- `corpus.py`: Manages the graph dataset serialization with atomic JSON disk writes to prevent corruption.
- `graphs.py` & `descriptors.py`: Graph generation logic and GDL formatting functions.
- `config.py`: Centralized configuration for graph problems and parameters.

## Installation

```bash
# Clone the repository
git clone https://github.com/bendjafer/llm-gqp.git
cd llm-gqp

# Set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root to configure your cloud providers. The pipeline automatically detects the appropriate provider based on the model name.

```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

*Note: Local Ollama models run offline and do not require an API key.*

## Usage

### 1. Single Model Run

You can use `orchestrator.py` to generate a graph and immediately test a specific model:

```bash
# Example: Run GPT-4o-mini on a large, directed, unweighted graph
python orchestrator.py --generate --gen-directed --gen-size large --model gpt-4o-mini
```

**Available models include (but are not limited to):**

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o1-mini`
- **Gemini**: `gemini-2.0-flash`, `gemini-1.5-pro`
- **Groq**: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768` *(Set `LLM_PROVIDER=groq` in `.env` for ambiguous names like llama3.1)*
- **Ollama**: `llama3.1:8b`, `qwen2.5:7b`

### 2. Full Evaluation Matrix

To run a comprehensive benchmark across 4 graph configurations (combinations of directed/undirected and weighted/unweighted) and all configured providers in cloud-first order, run:

```bash
bash benchmark.sh
```

*This script features smart-skipping (bypassing existing CSV results and skipping providers with missing API keys).*

### 3. Analyzing Results

Raw LLM responses and evaluations are saved incrementally into CSV files under the `results/` directory. You can use the provided Jupyter notebook (`main.ipynb`) to explore the data, calculate parse rates, and generate accuracy heatmaps and confusion matrices.

## License

MIT
