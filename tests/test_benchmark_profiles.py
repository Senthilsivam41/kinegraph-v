from eval.benchmark_profiles import build_profile_dataset, get_profile


ROWS = [{
    "user_input": "How do I query document.pdf?",
    "reference": "Use the query endpoint.",
    "reference_contexts": "['first source chunk', 'second source chunk']",
    "query_style": "PERFECT_GRAMMAR",
    "synthesizer_name": "multi_hop_specific_query_synthesizer",
}]


def test_profiles_are_explicit_and_do_not_change_production_defaults():
    hybrid = get_profile("hybrid")
    lexical = get_profile("hybrid_lexical")
    vectorless = get_profile("vectorless")
    adaptive = get_profile("adaptive_hybrid")

    assert hybrid.requested_mode.value == "hybrid"
    assert hybrid.enable_lexical_fusion is False
    assert hybrid.allow_mode_downgrade is False
    assert lexical.enable_lexical_fusion is True
    assert lexical.name != vectorless.name
    assert vectorless.requested_mode.value == "vectorless"
    assert adaptive.allow_mode_downgrade is True
    assert set(adaptive.declared_effective_modes) == {"hybrid", "vector", "graph"}


def test_vectorless_profile_uses_same_frozen_question_reference_and_sources():
    hybrid_sample = build_profile_dataset(ROWS, get_profile("hybrid"))[0]
    vectorless_sample = build_profile_dataset(ROWS, get_profile("vectorless"))[0]

    assert hybrid_sample["sample_id"] == vectorless_sample["sample_id"]
    assert hybrid_sample["question"] == vectorless_sample["question"]
    assert hybrid_sample["ground_truth"] == vectorless_sample["ground_truth"]
    assert "attachment_content" not in hybrid_sample
    assert vectorless_sample["attachment_content"] == (
        "first source chunk\n\n--- SOURCE CONTEXT ---\n\nsecond source chunk"
    )
    assert vectorless_sample["attachment_name"] == "kinegraph-benchmark-corpus.txt"
    assert len(vectorless_sample["source_corpus_sha256"]) == 64
    assert set(vectorless_sample["categories"]) >= {"multi_hop", "exact_token"}


def test_vectorless_does_not_receive_a_per_query_oracle_context():
    rows = [
        *ROWS,
        {
            "user_input": "What is graph traversal?",
            "reference": "It follows graph relationships.",
            "reference_contexts": "['graph traversal source']",
            "query_style": "PERFECT_GRAMMAR",
            "synthesizer_name": "single_hop_specific_query_synthesizer",
        },
    ]

    samples = build_profile_dataset(rows, get_profile("vectorless"))

    assert samples[0]["attachment_content"] == samples[1]["attachment_content"]
    assert "first source chunk" in samples[1]["attachment_content"]
    assert "graph traversal source" in samples[0]["attachment_content"]
    assert samples[0]["source_corpus_sha256"] == samples[1]["source_corpus_sha256"]
