---
format: 1920x1080
duration: 60s
message: "KineticGraph is four stacked layers — API → orchestration → stores → fusion"
arc: concept-explainer
audience: engineers
mode: collaborative
music: none
captions: yes
---

## Frame 1 — Hook

- status: built
- duration: 5s
- poster: 3s
- transition_in: cut
- type: hook
- persuasion: Counterintuitive claim + progressive disclosure
- beat: Curiosity + tension
- scene: Full-bleed title field; a single retrieval box fades, then cracks into a question — "one search?" — before the stack thesis takes over
- voiceover: "Most RAG systems pick one way to search. KineticGraph stacks four."
- blueprint: kinetic-type-beats
- src: compositions/frames/01-hook.html

narrativeRole: Opens the gap — single-retrieval RAG vs a layered stack.
keyMessage: One search path is the default; this architecture refuses that tradeoff.

## Frame 2 — The stack

- status: built
- duration: 7s
- poster: 4s
- transition_in: crossfade
- type: product_intro
- persuasion: Frame-then-fill + Rule of three (extended to four)
- beat: Orientation + anticipation
- scene: Four empty glass slab outlines appear as a vertical stack silhouette — labels still ghosted; the stage that every later frame will grow on
- voiceover: "API. Orchestration. Stores. Fusion. One stack."
- blueprint: titlecard-reveal
- src: compositions/frames/02-the-stack.html

narrativeRole: Names the protagonist — the four-layer stack — before any layer fills in.
keyMessage: The whole system is four layers stacked from request to ranked answer.

## Frame 3 — API layer

- status: built
- duration: 10s
- poster: 6s
- transition_in: push-slide DOWN
- type: feature_showcase
- persuasion: Progressive disclosure + Concretization
- beat: Clarity + focus
- scene: Top glass slab docks into place with a soft lock; FastAPI routes glow as thin rails — ingest, query, health — on the shared stack stage
- voiceover: "On top — FastAPI. Ingest documents. Ask questions. Stay healthy under load."
- blueprint: titlecard-reveal
- src: compositions/frames/03-api-layer.html

narrativeRole: Docks the API layer as the system's entry surface.
keyMessage: The API is the thin, scalable door for ingest and query.

## Frame 4 — Orchestration

- status: built
- duration: 12s
- poster: 7s
- transition_in: push-slide DOWN
- type: feature_showcase
- persuasion: Causal chain + Signposting
- beat: Comprehension + momentum
- scene: Second slab docks beneath API; inside it, a compact LangGraph route diagram draws — Vector / Graph / Hybrid / Vectorless agents feeding forward
- voiceover: "Under it — LangGraph. It routes each query to vector, graph, hybrid, or keyword search."
- blueprint: spatial-pan-stations
- src: compositions/frames/04-orchestration.html

narrativeRole: Docks orchestration as the adaptive router between API and stores.
keyMessage: LangGraph chooses how to retrieve — not every query uses the same path.

## Frame 5 — Stores

- status: built
- duration: 14s
- poster: 8s
- transition_in: push-slide DOWN
- type: feature_showcase
- persuasion: Numbered enumeration + Comparison of options
- beat: Mastery + fascination
- scene: Third slab docks; three store chips assemble inside — ChromaDB (vectors), Neo4j (relationships), BM25 cache (keywords) — side by side on the same stack stage
- voiceover: "Then three stores. Chroma for meaning. Neo4j for relationships. BM25 for exact words."
- blueprint: grid-card-assemble
- src: compositions/frames/05-stores.html

narrativeRole: Docks the data plane as three complementary retrieval stores.
keyMessage: Semantic, relational, and lexical retrieval each earn a dedicated store.

## Frame 6 — Fusion and land

- status: built
- duration: 12s
- poster: 8s
- transition_in: crossfade
- type: branding
- persuasion: Distillation + Callback
- beat: Clarity + "now I get it"
- scene: Bottom slab docks — Reciprocal Rank Fusion merges three result streams into one ranked rail; the full four-layer stack holds as the closing lockup
- voiceover: "At the base — Reciprocal Rank Fusion. Three result lists become one ranking. Four layers. One answer."
- blueprint: titlecard-reveal
- src: compositions/frames/06-fusion-land.html

narrativeRole: Docks fusion and lands the thesis — four layers, one answer.
keyMessage: RRF merges the stores so the user gets one ranked answer from the whole stack.
