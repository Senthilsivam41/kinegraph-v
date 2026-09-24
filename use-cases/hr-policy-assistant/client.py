"""Small standalone HTTP client for the synthetic HR policy demo."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
CONTRACT = json.loads((ROOT / "contract" / "prompts.json").read_text())
CASES = CONTRACT["cases"]


def build_artifact(case, requested_mode, result, code_revision, captured_at=None):
    """Wrap an API response with the inputs needed to reproduce the run."""
    return {
        "schema_version": "kinegraph.demo.evidence.v1",
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
        "kinegraph_code_revision": code_revision,
        "source_corpus_sha256": CONTRACT["corpus_sha256"],
        "case": case,
        "requested_mode": requested_mode,
        "effective_mode": result.get("effective_mode"),
        "query_id": result.get("query_id"),
        "response": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case["id"] for case in CASES], required=True)
    parser.add_argument("--mode", choices=("hybrid", "vector"), default="hybrid")
    parser.add_argument("--base-url", default=os.getenv("KINEGRAPH_API_URL", "http://localhost:8000"))
    parser.add_argument("--code-revision", default=os.getenv("KINEGRAPH_CODE_REVISION"))
    args = parser.parse_args()
    if not args.code_revision:
        parser.error("--code-revision is required to make the evidence reproducible")
    case = next(case for case in CASES if case["id"] == args.case)
    body = json.dumps({"query": case["question"], "mode": args.mode, "allow_mode_downgrade": False}).encode()
    request = Request(f"{args.base_url.rstrip('/')}/api/v1/query/", data=body, headers={"Content-Type": "application/json"})

    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except (HTTPError, URLError) as error:
        raise SystemExit(f"Kinegraph API request failed: {error}") from error

    print(json.dumps(build_artifact(case, args.mode, result, args.code_revision), indent=2))


if __name__ == "__main__":
    main()
