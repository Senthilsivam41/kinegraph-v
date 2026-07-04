import yaml
from pathlib import Path
from typing import List, Tuple, Set

class OntologySchema:
    def __init__(self, yaml_path: str = "config/ontology_schema.yaml"):
        # Make path resolution robust (absolute path support relative to workspace root)
        resolved_path = Path(yaml_path)
        if not resolved_path.is_absolute():
            # If relative, resolve against workspace root (we can check standard location)
            possible_roots = [Path("."), Path(__file__).resolve().parents[2]]
            for root in possible_roots:
                test_path = root / yaml_path
                if test_path.exists():
                    resolved_path = test_path
                    break

        with open(resolved_path, "r") as f:
            data = yaml.safe_load(f)
        self.version = data.get("version", "1.0.0")
        self.entity_types = data.get("entity_types", [])
        self.relation_types = data.get("relation_types", [])
        self.valid_triples = [tuple(t) for t in data.get("valid_triples", [])]
        self.valid_triples_set = set(self.valid_triples)

    def validate_triple(self, source_type: str, relation: str, target_type: str) -> bool:
        return (source_type, relation, target_type) in self.valid_triples_set
