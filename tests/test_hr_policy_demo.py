import json
import hashlib
import importlib.util
from pathlib import Path


DEMO_DIR = Path(__file__).parents[1] / "use-cases" / "hr-policy-assistant"


def load_client():
    spec = importlib.util.spec_from_file_location("hr_policy_client", DEMO_DIR / "client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_hr_policy_client_preserves_reproducibility_evidence():
    client = load_client()
    artifact = client.build_artifact(
        client.CASES[0],
        "hybrid",
        {"effective_mode": "hybrid", "execution_time_ms": 12.5},
        "abc123",
        "2026-09-24T00:00:00+00:00",
    )

    assert artifact["kinegraph_code_revision"] == "abc123"
    assert artifact["source_corpus_sha256"] == client.CONTRACT["corpus_sha256"]
    assert artifact["requested_mode"] == artifact["effective_mode"] == "hybrid"
    assert artifact["response"]["execution_time_ms"] == 12.5


def test_hr_policy_client_builds_explicit_mode_request():
    client = load_client()
    captured = {}

    class Response:
        def read(self, *_args):
            return b'{"effective_mode":"vector"}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    result = client.query_case("http://localhost:8000/", client.CASES[0], "vector", opener=opener)

    assert captured["url"] == "http://localhost:8000/api/v1/query/"
    assert captured["body"]["mode"] == "vector"
    assert captured["body"]["allow_mode_downgrade"] is False
    assert captured["timeout"] == 60
    assert result["effective_mode"] == "vector"
