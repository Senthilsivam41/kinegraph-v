"""Small standalone HTTP client for the synthetic HR policy demo."""

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
CASES = json.loads((ROOT / "contract" / "prompts.json").read_text())["cases"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case["id"] for case in CASES], required=True)
    parser.add_argument("--mode", choices=("hybrid", "vector"), default="hybrid")
    parser.add_argument("--base-url", default=os.getenv("KINEGRAPH_API_URL", "http://localhost:8000"))
    args = parser.parse_args()
    case = next(case for case in CASES if case["id"] == args.case)
    body = json.dumps({"query": case["question"], "mode": args.mode, "allow_mode_downgrade": False}).encode()
    request = Request(f"{args.base_url.rstrip('/')}/api/v1/query/", data=body, headers={"Content-Type": "application/json"})

    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except (HTTPError, URLError) as error:
        raise SystemExit(f"Kinegraph API request failed: {error}") from error

    print(json.dumps({"case": case, "response": result}, indent=2))


if __name__ == "__main__":
    main()
