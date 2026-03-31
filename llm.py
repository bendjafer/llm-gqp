import os
import time
import google.genai as genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Prompts ───────────────────────────────────────────────────────────────────

QUERIES = {
    "node_count":    "How many nodes does this graph have?",
    "edge_count":    "How many edges does this graph have?",
    "neighbors":     "What are the neighbors of the first node?",
    "degree":        "What is the degree of the first node?",
    "shortest_path": "What is the shortest path from the first node to the fourth node?",
    "clustering":    "What is the clustering coefficient of the first node?",
    "diameter":      "What is the diameter of this graph?",
    "density":       "What is the density of this graph?",
    "is_connected":  "Is this graph connected?",
    "cycle":         "Does this graph contain a cycle?",
}

PROMPT_TEMPLATE = """{description}

Let's construct a graph with the nodes and edges first.
Q: {question}

Think step by step, then give your final answer strictly after the separator.
Format:
[your reasoning]
###ANSWER###
[your answer]

Answer format rules:
- node_count / edge_count / degree / diameter: integer only (e.g. 34)
- neighbors / shortest_path: Python list using the exact node names/IDs as they appear in the graph description (e.g. ['Napoleon', 'Myriel'] or [0, 3])
- clustering / density: float rounded to 4 decimal places (e.g. 0.1500)
- is_connected / cycle: True or False
- NEVER use integer indices to replace node names — use the actual node identifiers"""

# ── LLMs ──────────────────────────────────────────────────────────────────────
class GroqLLM:
    def __init__(self, model="llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY2"))
        self.model  = model

    def ask(self, description, question, first_node=None, fourth_node=None):
        node_hint = ""
        if first_node is not None:
            node_hint = f"The first node is: '{first_node}'\nThe fourth node is: '{fourth_node}'\n\n"
        prompt = PROMPT_TEMPLATE.format(
            description=node_hint + description,
            question=question
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
        
class GeminiLLM:
    def __init__(self, model="gemini-2.0-flash", rpm=10):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model  = model
        self.delay  = 60 / rpm

    def ask(self, description, question, retries=3):
        prompt = PROMPT_TEMPLATE.format(description=description, question=question)
        for attempt in range(retries):
            try:
                time.sleep(self.delay)
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    wait = 60 * (attempt + 1)
                    print(f"  ⏳ Rate limit, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise e





class OllamaLLM:
    def __init__(self, model="mistral"):
        self.model = model

    def ask(self, description, question):
        import ollama
        prompt = PROMPT_TEMPLATE.format(description=description, question=question)
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()