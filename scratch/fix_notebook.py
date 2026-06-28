import json
from pathlib import Path

notebook_path = Path("/Users/sendils/work/repo/kinetic-v/kinegraph-v/notebooks/rag_evaluation.ipynb")

print(f"Loading notebook from {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find the setup cell (cell 2)
# The setup cell contains "from dotenv import load_dotenv"
found = False
for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        source_text = "".join(source)
        if "load_dotenv(os.path.abspath('../.env'))" in source_text:
            print("Found setup cell. Patching it...")
            new_source = [
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
                "from eval.ragas_evaluator import RAGASEvaluator, ALL_METRICS, METRIC_DESCRIPTIONS\n",
                "from eval.metrics_collector import MetricsCollector\n"
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
    print("Could not find the setup cell to patch!")
