import os
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import OLLAMA_MODEL, P_PROBLEMS_PROMPTS
from prompts import ROLE_AND_RULES, PROMPT_TEMPLATE
from eval import add_ground_truth, add_correctness

load_dotenv()


class OllamaLLM:
    def __init__(self, model=None):
        raw_model = model or OLLAMA_MODEL
        model_name = raw_model.replace(":", "_").replace("/", "_")

        self.model_name = raw_model
        self.chat_model = ChatOllama(
            model=raw_model,
            temperature=0,
            num_predict=512,
            client_kwargs={"timeout": 90},
        )
        self.prompt = ChatPromptTemplate([
            ("system", ROLE_AND_RULES),
            ("human", PROMPT_TEMPLATE),
        ])
        self.pipeline = self.prompt | self.chat_model | StrOutputParser()

        os.makedirs("results", exist_ok=True)
        self.save_path = os.path.join("results", f"{model_name}_results.csv")

        # Intentionally remove any prior partial run so results are always fresh.
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def generate_answers(self, corpus, graphs, selected_graphs=None, selected_descriptors=None):
        results = []
        selected_graphs = set(selected_graphs) if selected_graphs else None
        selected_descriptors = set(selected_descriptors) if selected_descriptors else None

        tasks = []
        for graph_name, formats in corpus:
            if selected_graphs and graph_name not in selected_graphs:
                continue
            for gdl_format, gdl_content in formats.items():
                if selected_descriptors and gdl_format not in selected_descriptors:
                    continue
                for p_problem, question in P_PROBLEMS_PROMPTS.items():
                    tasks.append((graph_name, gdl_format, gdl_content, p_problem, question))

        progress = tqdm(tasks, desc="Running graph QA", unit="task")
        for graph_name, gdl_format, gdl_content, p_problem, question in progress:
            progress.set_postfix(graph=graph_name, fmt=gdl_format, problem=p_problem)

            try:
                llm_response = self.pipeline.invoke({
                    "description": gdl_content,
                    "question": question,
                })

                parts = llm_response.split("###ANSWER###")
                reasoning = parts[0].strip() if parts else ""
                answer = parts[1].strip() if len(parts) > 1 else ""

                if not answer:
                    print(f"\n[WARN] No ###ANSWER### found for {graph_name} | {gdl_format} | {p_problem}")
                    answer = "No answer found"
                    reasoning = reasoning or "No reasoning detected"

                row = {
                    "graph_name": graph_name,
                    "format": gdl_format,
                    "problem": p_problem,
                    "reasoning": reasoning,
                    "answer": answer,
                }

            except Exception as e:
                print(f"\n[WARN] Skipping failed row {graph_name} | {gdl_format} | {p_problem}: {type(e).__name__}: {e}")
                row = {
                    "graph_name": graph_name,
                    "format": gdl_format,
                    "problem": p_problem,
                    "reasoning": f"ERROR: {type(e).__name__}: {e}",
                    "answer": "No answer found",
                }

            results.append(row)

            pd.DataFrame([row]).to_csv(
                self.save_path,
                mode="a",
                header=not os.path.exists(self.save_path),
                index=False,
            )

        df = pd.DataFrame(results)

        df = add_ground_truth(df, graphs)
        df = add_correctness(df)

        df.to_csv(self.save_path, index=False)
        print(f"[INFO] Enriched results saved -> {self.save_path}")
        return df

    def view_prompt(self):
        self.prompt.pretty_print()