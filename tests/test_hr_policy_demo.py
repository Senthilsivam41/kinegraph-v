import json
import hashlib
from pathlib import Path


DEMO_DIR = Path(__file__).parents[1] / "use-cases" / "hr-policy-assistant"


def test_hr_policy_demo_contract_is_complete():
    corpus = DEMO_DIR / "data" / "northwind-analytics-employee-handbook.md"
    contract = json.loads((DEMO_DIR / "contract" / "prompts.json").read_text())

    assert corpus.is_file()
    assert (DEMO_DIR / "client.py").is_file()
    assert (DEMO_DIR.parents[0] / "README.md").is_file()
    assert contract["schema_version"] == "kinegraph.demo.hr-policy.v1"
    assert (DEMO_DIR / "contract" / contract["corpus"]).resolve() == corpus.resolve()
    assert contract["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert [case["id"] for case in contract["cases"]] == [f"HR-{n:03d}" for n in range(1, 7)]
    assert all(case["expected_facts"] and case["source_sections"] for case in contract["cases"])
    assert contract["cases"][-1]["outcome"] == "partial_refusal"
