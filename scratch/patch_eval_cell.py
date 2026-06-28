import json
from pathlib import Path

notebook_path = Path("/Users/sendils/work/repo/kinetic-v/kinegraph-v/notebooks/rag_evaluation.ipynb")

print(f"Loading notebook from {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find cell 32 (evaluator.evaluate_batch)
found = False
for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        source_text = "".join(source)
        if "evaluator.evaluate_batch(EVAL_DATASET)" in source_text and "pd.read_csv" not in source_text:
            print("Found evaluation cell. Patching it...")
            new_source = [
                "import os\n",
                "import pandas as pd\n",
                "\n",
                "evaluator = RAGASEvaluator(\n",
                "    openai_api_key=os.getenv('OPENAI_API_KEY'),\n",
                "    metrics=['faithfulness', 'answer_relevancy', 'context_precision',\n",
                "             'context_recall', 'answer_correctness'],\n",
                ")\n",
                "\n",
                "# OPTION: Load pre-computed results from eval_results.csv to avoid running live evaluation\n",
                "csv_path = 'eval_results.csv'\n",
                "if os.path.exists(csv_path):\n",
                "    print(f\"Found existing results in {csv_path}. Loading them...\")\n",
                "    results_df = pd.read_csv(csv_path)\n",
                "else:\n",
                "    print(\"No pre-computed results found. Running live evaluation (this may take several minutes)...\")\n",
                "    results_df = evaluator.evaluate_batch(EVAL_DATASET)\n",
                "    results_df.to_csv(csv_path, index=False)\n",
                "\n",
                "print(results_df.shape)\n",
                "results_df.head(3)"
            ]
            cell["source"] = new_source
            found = True
            break

if found:
    print(f"Saving patched notebook to {notebook_path}...")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("Notebook patched successfully!")
else:
    print("Could not find the evaluation cell to patch!")
