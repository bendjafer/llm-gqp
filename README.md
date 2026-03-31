# LLM-GQP: LLM-Graph Query Processing

A benchmarking pipeline designed to evaluate the reasoning capabilities of Large Language Models (LLMs) on graph data. This project converts standard NetworkX benchmark graphs into various Graph Description Language (GDL) formats and queries LLMs to solve common graph problems.

## 🚀 Features

-   **8 Graph Decription Languages (GDL)**:
    -   Natural Language (Text)
    -   Adjacency List
    -   Edge List
    -   Cypher (Neo4j style)
    -   GML (Graph Modelling Language)
    -   GraphML (XML based)
    -   Story Encoding
    -   Random Walk (GDL4LLM style)
-   **Multi-Provider Support**: 
    -   **Gemini** (Google)
    -   **Groq** (Llama-3, etc.)
    -   **Ollama** (Local models)
-   **Automated Evaluation**: Integrated `evaluate.py` script to compare LLM responses against NetworkX ground truth for metrics like node count, shortest paths, and diameter.

## 📁 Project Structure

-   `graphs.py`: Contains the core benchmark graphs and GDL descriptor functions.
-   `llm.py`: Implements client wrappers for Gemini, Groq, and Ollama with structured prompting.
-   `evaluate.py`: A metrics engine that parses LLM responses and calculates accuracy.
-   `main.ipynb`: An interactive notebook showcasing the end-to-end pipeline.
-   `results/`: Storage for generated corpora, LLM raw responses, and final evaluations.

## 🛠️ Installation

```bash
# Clone the repository (once created)
git clone https://github.com/bendjafer/llm-gqp.git
cd llm-gqp

# Install dependencies
pip install -r requirements.txt
```

## 📋 Usage

1.  Set up your `.env` file with `GEMINI_API_KEY` and `GROQ_API_KEY`.
2.  Run the `main.ipynb` notebook to generate graphs and descriptors.
3.  Query the LLMs using the interface in `llm.py`.
4.  Run evaluation:
    ```bash
    python3 evaluate.py
    ```

## ⚖️ License
MIT
