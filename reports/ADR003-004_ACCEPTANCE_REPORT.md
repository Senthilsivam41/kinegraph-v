# ADR-003/004 Acceptance Report

Date: 2026-09-07
Branch: `codex/adr003-004-acceptance-calibration`

## Decision

ADR-003 and ADR-004 remain experimental. No retrieval configuration or
Kinetic Score policy is promoted by this branch.

## Accepted gates

- ADR-003 rejects missing profiles, rejected RAGAS evidence, missing metrics,
  non-finite values, out-of-range normalized metrics, negative latency, and
  incomplete candidate provenance.
- Controlled cross-encoder experiments remain one-lever comparisons and must
  show a slice-level precision or nDCG gain within the latency policy.
- ADR-004 refuses answers when citation validation is unavailable, preserves
  partial status for explicit retrieval conflicts, and keeps the score in
  shadow evidence-confidence mode.
- Calibration requires finite scores, boolean human labels, the minimum
  sample count, and both positive and negative classes. Promotion remains
  disabled even after calibration.

## Rejected or deferred experiments

- Local Docker Model Runner model `docker.io/ai/qwen3.5:9b-q8_0` was visible
  through the local `/engines/v1/models` endpoint but produced no completion
  within 180 seconds. It is not qualified as a judge.
- No new RAGAS benchmark was run or compared with the accepted Fireworks
  baseline.

## Remaining evidence gaps

- Run accepted Hybrid, Hybrid+BM25, and Vectorless benchmark slices when a
  usable judge model is available.
- Collect the required labeled calibration set with both acceptable and
  unacceptable examples.
- Obtain governance approval before exposing or promoting the Kinetic Score.

## Verification

Focused ADR-003/004 tests pass. The full repository suite is required before
merge; any missing optional dependency remains an environment limitation and
must be reported rather than converted into benchmark evidence.
