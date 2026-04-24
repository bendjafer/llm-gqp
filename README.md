# 🕸️ LLM-GQP: LLM-Graph Query Processing

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Evaluation](https://img.shields.io/badge/LLM-Evaluation-purple.svg)

**LLM-GQP** is a comprehensive benchmarking pipeline designed to evaluate the reasoning and algorithmic capabilities of Large Language Models (LLMs) on graph theory problems. 

This project translates standard NetworkX benchmark graphs into various **Graph Description Languages (GDLs)** and tests LLM accuracy across multiple topological domains including shortest paths, connectivity, clustering coefficients, density, and cycle detection.

---

## 🚀 Key Features

- **Multi-Provider Architecture**: Native, auto-detecting support for **OpenAI**, **Gemini**, **Groq**, and **Ollama** (local) via LangChain.
- **Matrix Benchmarking**: Automated evaluation sweep across directed, undirected, weighted, and unweighted graphs using `benchmark.sh`.
- **Robust XML Output Parsing**: Strict `<answer>` tag extraction with built-in fallbacks and prose-stripping to handle verbose or malformed LLM responses without relying on error-prone reasoning blocks.
- **Precision-Tolerant Evaluation**: Advanced metrics engine (`eval.py`) that handles LLM rounding behaviors (e.g., matching float predictions with a flexible `1e-2` tolerance).
- **Few-Shot Learning Injection**: Automatically fetch and inject pristine, strictly-formatted few-shot examples into the LLM prompt using `--few-shot-k` to dramatically improve in-context learning.
- **7 Graph Description Languages (GDL)**:
  - Natural Language (Text)
  - Adjacency List
  - Edge List
  - GML
  - GraphML
  - Random Walk
  - L2SP Paths

---

## 📁 Project Structure

*   **`orchestrator.py`**: The main CLI entry point. Generates graphs, extracts GDL descriptors, and orchestrates the LLM benchmarking tasks.
*   **`llm.py`**: The provider-agnostic LLM interface. Handles API connections, dynamic prompt injection, memory-safe token budgeting (2048 limit), and robust execution state (resume capability).
*   **`prompts.py`**: Defines the strict expert persona, prompt formatting rules, and the robust `_clean_answer` regex parser.
*   **`eval.py`**: The metrics engine. Evaluates predicted answers against ground truth while accommodating data types (floats vs integers vs booleans).
*   **`benchmark.sh`**: Shell script running an optimized evaluation sweep targeting specific LLMs (`gpt-4o-mini`, `llama3.1:8b`, `graphwalker:latest`) across undirected and weighted graphs.
*   **`corpus.py`**: Manages the graph dataset serialization utilizing atomic JSON disk writes to prevent data corruption.
*   **`graphs.py` & `descriptors.py`**: NetworkX graph generation logic and GDL formatting functions.
*   **`config.py`**: Centralized configuration and mappings for graph problems and difficulty metrics.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/bendjafer/llm-gqp.git
cd llm-gqp

# Set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install strictly curated dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the project root to configure your cloud providers. The pipeline will automatically detect the appropriate provider based on the model name prefix.

```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

> **Note:** Local Ollama models run completely offline and do not require any API keys. However, you must install the models locally before benchmarking.

### Local Model Setup (Ollama)

To run the local benchmarks, ensure the Ollama service is running (`ollama serve`) and pull the standard models:
```bash
ollama pull llama3.1:8b
```

**For custom/fine-tuned models (e.g., `graphwalker:latest`):**
If you are using raw `.gguf` weights for a custom model, you must build it into Ollama manually so the pipeline can access it:
1. Create a text file named `Modelfile` next to your `.gguf` file containing the line: `FROM ./graphwalker.gguf`
2. Run `ollama create graphwalker:latest -f Modelfile`
3. Verify both models are ready by running `ollama list`.

---

## 📋 Usage

### 1. CLI Orchestrator (Single Run & Debugging)

You can use `orchestrator.py` to generate graphs, inject few-shot examples, inspect prompts safely, and test specific models:

```bash
# Example: Run GPT-4o-mini on the Petersen graph with 2 few-shot examples
python orchestrator.py --graphs Petersen --model gpt-4o-mini --few-shot-k 2

# Example: Print the exact rendered prompt template to the console before running
python orchestrator.py --graphs Petersen --model gpt-4o-mini --show-prompt
```

**Available models include (but are not limited to):**

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o1-mini`
- **Gemini**: `gemini-2.0-flash`, `gemini-1.5-pro`
- **Groq**: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768` *(Set `LLM_PROVIDER=groq` in `.env` for ambiguous names)*
- **Ollama**: `llama3.1:8b`, `graphwalker:latest`, `qwen2.5:7b`

### 2. Streamlined Benchmark Sweep

To run a targeted benchmark across 2 graph configurations (**Undirected/Unweighted** and **Undirected/Weighted**) using 5 optimized GDL formats, simply run:

```bash
bash benchmark.sh
```

*This script features **live prompt printing**, **smart-skipping** (resuming exact rows natively from CSV), and specifically pits OpenAI (`gpt-4o-mini`) against local Ollama models (`llama3.1:8b` and `graphwalker:latest`).*

### 3. Analyzing Results

Raw LLM responses and evaluations are saved incrementally into CSV files under the `results/` directory. You can use the provided Jupyter notebook (`main.ipynb`) to dynamically explore the data, calculate parse rates, and generate accuracy heatmaps and confusion matrices.

---

## ⚖️ License

Distributed under the MIT License.
