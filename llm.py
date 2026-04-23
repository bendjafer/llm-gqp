import os
import re

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from config import OLLAMA_MODEL, P_PROBLEMS
from eval import add_correctness, add_ground_truth
from prompts import (
    PROMPT_TEMPLATE,
    ROLE_AND_RULES,
    build_prompt_vars,
    get_question,
    is_valid_problem,
    parse_response,
)

load_dotenv()


# ── Provider registry ──────────────────────────────────────────────────────────
# Each entry: provider_name → (factory(model_str) → chat_model, required_env_key | None)
# Imports are lazy so missing optional packages don't crash the whole module.

def _build_ollama(model: str):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=0, num_predict=2048, client_kwargs={"timeout": 120})

def _build_openai(model: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0, max_tokens=2048)

def _build_groq(model: str):
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, temperature=0, max_tokens=2048)

def _build_gemini(model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model=model, temperature=0, max_output_tokens=2048, google_api_key=api_key)


_REGISTRY: dict[str, tuple] = {
    "ollama": (_build_ollama, None),
    "openai": (_build_openai, "OPENAI_API_KEY"),
    "groq":   (_build_groq,   "GROQ_API_KEY"),
    "gemini": (_build_gemini, "GOOGLE_API_KEY"),
}

# Model-name patterns for automatic provider detection (matched top-to-bottom).
# Groq shares model names with Ollama — use LLM_PROVIDER=groq in .env instead.
_INFERENCE_RULES = [
    (r"^gpt-|^o[1-9]-|^o3",  "openai"),
    (r"^gemini-",              "gemini"),
]


def _detect_provider(model: str) -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in _REGISTRY:
            raise ValueError(
                f"LLM_PROVIDER='{explicit}' is not supported. "
                f"Supported providers: {sorted(_REGISTRY)}"
            )
        return explicit
    for pattern, provider in _INFERENCE_RULES:
        if re.match(pattern, model, re.IGNORECASE):
            return provider
    return "ollama"


# ── LLM ───────────────────────────────────────────────────────────────────────

class LLM:
    """Provider-agnostic LLM wrapper for graph-QA evaluation.

    Provider is resolved automatically from the model name or the LLM_PROVIDER
    environment variable. Supported: ollama, openai, groq, gemini.
    Adding a new provider requires only a factory function and a registry entry.
    """

    def __init__(self, model: str | None = None, run_dir: str | None = None):
        raw_model        = model or OLLAMA_MODEL
        provider         = _detect_provider(raw_model)
        factory, env_key = _REGISTRY[provider]

        if env_key and not os.getenv(env_key):
            raise EnvironmentError(
                f"Provider '{provider}' requires {env_key} to be set in .env"
            )

        self.model_name = raw_model
        self.provider   = provider
        self.chat_model = factory(raw_model)

        safe_model    = raw_model.replace(":", "_").replace("/", "_")
        self.prompt   = ChatPromptTemplate([
            ("system", ROLE_AND_RULES),
            ("human",  PROMPT_TEMPLATE),
        ])
        self.pipeline = self.prompt | self.chat_model | StrOutputParser()

        if run_dir is None:
            run_dir = os.path.join("results", safe_model)
        os.makedirs(run_dir, exist_ok=True)

        self.run_dir   = run_dir
        self.save_path = os.path.join(run_dir, f"{safe_model}.csv")

    def generate_answers(
        self,
        corpus,
        graphs: dict,
        selected_graphs: list | None = None,
        selected_descriptors: list | None = None,
    ) -> pd.DataFrame:
        selected_graphs      = set(selected_graphs)      if selected_graphs      else None
        selected_descriptors = set(selected_descriptors) if selected_descriptors else None

        done, prior = set(), None
        if os.path.exists(self.save_path):
            try:
                prior = pd.read_csv(self.save_path)
                if "answer" in prior.columns:
                    done = {
                        (str(r.graph_name), str(r.format), str(r.problem))
                        for r in prior.itertuples()
                    }
            except Exception:
                prior = None

        tasks = []
        for graph_name, formats in corpus:
            if selected_graphs and graph_name not in selected_graphs:
                continue
            graph_obj = graphs.get(graph_name)
            directed  = graph_obj.is_directed() if graph_obj else False
            for gdl_format, gdl_content in formats.items():
                if selected_descriptors and gdl_format not in selected_descriptors:
                    continue
                for p_problem in P_PROBLEMS:
                    if not is_valid_problem(p_problem, graph_obj):
                        continue
                    question, fmt = get_question(p_problem, directed)
                    if (graph_name, gdl_format, p_problem) not in done:
                        tasks.append((graph_name, gdl_format, gdl_content, p_problem, question, fmt))

        if done:
            print(f"[INFO] Resuming — {len(done)} task(s) already completed, {len(tasks)} remaining.")

        if not tasks:
            if os.path.exists(self.save_path):
                print("[INFO] All tasks already completed. Loading existing results.")
                df = pd.read_csv(self.save_path)
                return add_correctness(add_ground_truth(df, graphs))
            raise RuntimeError(
                f"No tasks to run and no results file found at '{self.save_path}'. "
                "Check that graph names and descriptors match the corpus."
            )

        new_results = []
        progress = tqdm(tasks, desc="Running graph QA", unit="task")
        for graph_name, gdl_format, gdl_content, p_problem, question, fmt in progress:
            progress.set_postfix(graph=graph_name, fmt=gdl_format, problem=p_problem)

            graph_obj   = graphs.get(graph_name)
            prompt_vars = build_prompt_vars(graph_obj)

            try:
                llm_response = self.pipeline.invoke({
                    "description":  gdl_content,
                    "format_label": gdl_format,
                    "question":     question,
                    "format_rule":  fmt,
                    **prompt_vars,
                })
                answer = parse_response(llm_response)
                if not answer:
                    print(f"\n[WARN] Empty answer for {graph_name} | {gdl_format} | {p_problem}")
                    answer = "No answer found"

            except Exception as e:
                print(f"\n[WARN] Skipping {graph_name} | {gdl_format} | {p_problem}: {type(e).__name__}: {e}")
                answer = "No answer found"

            new_results.append({
                "graph_name": graph_name,
                "format":     gdl_format,
                "problem":    p_problem,
                "answer":     answer,
            })

        results_df = pd.DataFrame(new_results)
        if prior is not None:
            raw_cols  = ["graph_name", "format", "problem", "answer"]
            prior_raw = prior[[c for c in raw_cols if c in prior.columns]]
            results_df = pd.concat([prior_raw, results_df], ignore_index=True)

        results_df = add_correctness(add_ground_truth(results_df, graphs))
        results_df.to_csv(self.save_path, index=False)
        print(f"[INFO] Results saved → {self.save_path}")
        return results_df

    def view_prompt(self):
        self.prompt.pretty_print()
