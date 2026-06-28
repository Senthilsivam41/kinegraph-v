import json
import os
from pathlib import Path

notebook_path = Path("notebooks/rag_evaluation.ipynb")
if not notebook_path.exists():
    print(f"Error: Notebook not found at {notebook_path}")
    exit(1)

print(f"Loading notebook from {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# 1. Patch Cell 1 (Setup)
print("Patching Cell 1 (Setup)...")
setup_cell = notebook["cells"][1]
setup_cell["source"] = [
    "# ── Setup ─────────────────────────────────────────────────────────────\n",
    "import sys, os\n",
    "from pathlib import Path\n",
    "cwd = Path(os.getcwd())\n",
    "project_root = None\n",
    "for parent in [cwd] + list(cwd.parents):\n",
    "    if (parent / '.env').exists() or (parent / 'requirements.txt').exists():\n",
    "        project_root = parent\n",
    "        break\n",
    "if project_root is None:\n",
    "    project_root = Path('..').resolve()\n",
    "sys.path.insert(0, str(project_root))\n",
    "\n",
    "from dotenv import load_dotenv\n",
    "load_dotenv(project_root / '.env')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "import nest_asyncio\n",
    "nest_asyncio.apply()\n",
    "\n",
    "PALETTE = ['#7c3aed', '#06b6d4', '#4f46e5', '#10b981', '#f59e0b']\n",
    "\n",
    "from eval.ragas_evaluator import RAGASEvaluator, ALL_METRICS, METRIC_DESCRIPTIONS\n"
]

# 2. Patch Cell 8 (Radar Chart)
print("Patching Cell 8 (Radar Chart)...")
radar_cell = notebook["cells"][8]
radar_cell["source"] = [
    "# ── Radar chart — average scores ──────────────────────────────────────\n",
    "import matplotlib.patches as mpatches\n",
    "from math import pi\n",
    "\n",
    "means = [report['per_metric'].get(m, {}).get('mean', 0) for m in metric_cols]\n",
    "N = len(metric_cols)\n",
    "angles = [n / float(N) * 2 * pi for n in range(N)] + [2 * pi]\n",
    "means_plot = means + [means[0]]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))\n",
    "ax.plot(angles, means_plot, 'o-', linewidth=2, color='#7c3aed')\n",
    "ax.fill(angles, means_plot, alpha=0.25, color='#7c3aed')\n",
    "ax.set_xticks(angles[:-1])\n",
    "ax.set_xticklabels(metric_cols, fontsize=10)\n",
    "ax.set_ylim(0, 1)\n",
    "ax.set_title('Average RAGAS Scores (Radar)', size=13, pad=20)\n",
    "plt.tight_layout()\n",
    "\n",
    "# Save spider graph to reports directory\n",
    "reports_dir = project_root / 'reports'\n",
    "os.makedirs(reports_dir, exist_ok=True)\n",
    "plt.savefig(reports_dir / 'spider_graph_ragas_score.png', dpi=300)\n",
    "print(f\"Spider graph saved to {reports_dir / 'spider_graph_ragas_score.png'}\")\n",
    "plt.show()\n"
]

# 3. Patch Cell 10 (Ingestion & Live Mode Evaluation)
print("Patching Cell 10 (Live Mode Evaluation)...")
live_eval_cell = notebook["cells"][10]
live_eval_cell["source"] = [
    "# ── Ingest System Documentation & Execute Live RAG ──────────────────\n",
    "import asyncio\n",
    "from pathlib import Path\n",
    "from backend.core.langgraph_workflow import HybridRAGWorkflow\n",
    "from backend.app.models import QueryMode\n",
    "from backend.services.chroma_service import ChromaService\n",
    "from backend.services.neo4j_service import Neo4jService\n",
    "from backend.workers.document_processor import chunk_text\n",
    "\n",
    "chroma = ChromaService()\n",
    "neo4j = Neo4jService()\n",
    "workflow = HybridRAGWorkflow(chroma_service=chroma, neo4j_service=neo4j)\n",
    "\n",
    "# 1. Automatically index project documentation if the vector store is empty\n",
    "try:\n",
    "    collection_count = chroma.get_or_create_collection().count()\n",
    "except Exception as e:\n",
    "    collection_count = 0\n",
    "\n",
    "if collection_count == 0:\n",
    "    print(\"📂 Vector store empty. Ingesting project docs to populate evaluation database...\")\n",
    "    doc_files = list(Path(\"../docs\").glob(\"*.md\")) + [Path(\"../README.md\")]\n",
    "    for file_path in doc_files:\n",
    "        if not file_path.exists():\n",
    "            continue\n",
    "        print(f\"  Ingesting {file_path.name}...\")\n",
    "        with open(file_path, \"r\", encoding=\"utf-8\") as f:\n",
    "            text = f.read()\n",
    "        \n",
    "        chunks = chunk_text(text)\n",
    "        doc_id = f\"doc_{file_path.stem.lower()}\"\n",
    "        chunk_ids = [f\"{doc_id}_chunk_{i}\" for i in range(len(chunks))]\n",
    "        metadatas = [{\"document_id\": doc_id, \"file_name\": file_path.name} for _ in chunks]\n",
    "        \n",
    "        # Load into Chroma\n",
    "        asyncio.run(chroma.add_documents(chunks, metadatas, chunk_ids))\n",
    "        \n",
    "        # Load basic metadata into Neo4j\n",
    "        asyncio.run(neo4j.add_document_graph(\n",
    "            doc_id=doc_id, \n",
    "            content=text[:2000], \n",
    "            metadata={\"file_name\": file_path.name}, \n",
    "            entities=[{\"name\": file_path.stem, \"type\": \"Document\"}], \n",
    "            relationships=[]\n",
    "        ))\n",
    "    print(\"✅ Ingestion complete!\")\n",
    "else:\n",
    "    print(f\"✅ Vector store already contains {collection_count} chunks. Skipping auto-ingestion.\")\n",
    "\n",
    "# 2. Helper to run live RAG queries\n",
    "def run_live_rag(question: str, mode: QueryMode):\n",
    "    try:\n",
    "        res = asyncio.run(workflow.execute_with_answer(question, mode=mode, max_results=5))\n",
    "        return {\n",
    "            \"answer\": res[\"answer\"],\n",
    "            \"contexts\": [chunk.content for chunk in res[\"chunks\"]]\n",
    "        }\n",
    "    except Exception as e:\n",
    "        print(f\"  Error running live query: {e}\")\n",
    "        return {\"answer\": \"Error retrieving answer.\", \"contexts\": []}\n",
    "\n",
    "# 3. Execute batch evaluations over all modes (or load from cache)\n",
    "mode_csv_path = 'mode_comparison_results.csv'\n",
    "if os.path.exists(mode_csv_path):\n",
    "    print(f\"Found existing mode comparison results in {mode_csv_path}. Loading them...\")\n",
    "    df_all_modes = pd.read_csv(mode_csv_path)\n",
    "else:\n",
    "    mode_results = {}\n",
    "    for mode_name, query_mode in [\n",
    "        ('vector', QueryMode.VECTOR),\n",
    "        ('graph',  QueryMode.GRAPH),\n",
    "        ('hybrid', QueryMode.HYBRID),\n",
    "    ]:\n",
    "        print(f\"🚀 Running live RAGAS evaluation for {mode_name.upper()} mode...\")\n",
    "        samples = []\n",
    "        for idx, s in enumerate(EVAL_DATASET[:3]):\n",
    "            print(f\"  [{idx+1}/3] Querying: {s['question']}\")\n",
    "            rag_output = run_live_rag(s['question'], query_mode)\n",
    "            samples.append({\n",
    "                'question':     s['question'],\n",
    "                'answer':       rag_output['answer'],\n",
    "                'contexts':     rag_output['contexts'],\n",
    "                'ground_truth': s.get('ground_truth'),\n",
    "            })\n",
    "        df_mode = evaluator.evaluate_batch(samples, show_progress=False)\n",
    "        df_mode['mode'] = mode_name\n",
    "        mode_results[mode_name] = df_mode\n",
    "\n",
    "    df_all_modes = pd.concat(mode_results.values(), ignore_index=True)\n",
    "    df_all_modes.to_csv(mode_csv_path, index=False)\n",
    "\n",
    "print('🎉 Live Mode comparison ready:', df_all_modes['mode'].value_counts().to_dict())\n"
]

# 4. Patch Cell 18 (Save Results & Export Markdown Report)
print("Patching Cell 18 (Save Results)...")
save_cell = notebook["cells"][18]
save_cell["source"] = [
    "# ── Export results ─────────────────────────────────────────────────────\n",
    "out_path = 'eval_results.csv'\n",
    "results_df.to_csv(out_path, index=False)\n",
    "print(f'Results saved to {out_path}')\n",
    "\n",
    "# ── Export Markdown Report ─────────────────────────────────────────────\n",
    "report = evaluator.generate_report(results_df)\n",
    "report_lines = []\n",
    "report_lines.append('# RAGAS Evaluation Report\\n')\n",
    "report_lines.append('## Per-Metric Average Scores\\n')\n",
    "for metric, stats in report['per_metric'].items():\n",
    "    report_lines.append(f'- **{metric}**: {stats[\"mean\"]:.4f}')\n",
    "report_lines.append('\\n## Actionable Recommendations\\n')\n",
    "for i, rec in enumerate(report['recommendations'], 1):\n",
    "    report_lines.append(f'{i}. {rec}')\n",
    "report_lines.append('\\n## Quality Tier Distribution\\n')\n",
    "for tier, count in report['summary']['quality_distribution'].items():\n",
    "    pct = count / report['summary']['total_samples'] * 100\n",
    "    report_lines.append(f'- **{tier}**: {count} ({pct:.1f}%)')\n",
    "report_lines.append(f'\\n**Overall Composite Score**: {report[\"summary\"][\"overall_composite_score\"]:.4f}')\n",
    "report_md = '\\n'.join(report_lines) + '\\n'\n",
    "\n",
    "reports_dir = project_root / 'reports'\n",
    "os.makedirs(reports_dir, exist_ok=True)\n",
    "with open(reports_dir / 'evaluation_report.md', 'w') as f:\n",
    "    f.write(report_md)\n",
    "print(f\"Markdown report saved to {reports_dir / 'evaluation_report.md'}\")\n"
]

print("Saving patched notebook...")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook patched successfully!")
