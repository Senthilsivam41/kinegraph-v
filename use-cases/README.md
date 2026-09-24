# Use cases

Each directory here is an independent client of Kinegraph, never an extension
of its core service. Keep this layout:

```text
use-cases/<use-case>/
  README.md        # setup and evidence required to make a claim
  client.py        # API or MCP client; stdlib where practical
  data/            # synthetic or redistributable source material only
  contract/        # prompts and expected grounded facts
```

Use cases may add a focused test under `tests/`. They must not change core
retrieval defaults, production configuration, or service implementation.

MCP clients follow the versioned [use-case MCP contract](MCP_CONTRACT.md).
