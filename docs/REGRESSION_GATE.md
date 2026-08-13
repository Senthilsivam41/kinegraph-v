# RAGAS regression gate

Kinegraph's RAGAS evaluator remains the only source of Faithfulness,
ContextPrecision, ContextRecall, AnswerRelevancy, and Correctness scores.  A
successful accepted run writes `reports/run_output.json` in the
`kinegraph.regression-gate.v1` format.  It contains the already-computed
aggregate composite plus per-query answer/context cases.

The root `regression_gate.py` is a thin DeepEval/pytest harness:

1. `RagasCompositeGate` reads the persisted composite and performs no LLM
   call.  It requires both the configured absolute floor (0.75 by default) and
   the last accepted `baseline_ref.json` score.
2. `CitationConstraintAdherence` is the only judge-backed metric.  It checks
   citation-to-retrieved-chunk adherence, which is not part of the RAGAS
   composite.
3. The baseline is atomically promoted only after the composite and every
   citation case pass.

Run it with:

```bash
deepeval test run regression_gate.py
```

Override paths without changing code:

```bash
KINEGRAPH_RUN_OUTPUT=reports/my_run_output.json \
KINEGRAPH_BASELINE_REF=reports/baseline_ref.json \
deepeval test run regression_gate.py
```

The evaluator writes the output only after its existing all-row
`ragas_failed == false` acceptance gate passes.  Failed or diagnostic runs
cannot advance the regression baseline.
