import re

def parse_research_synthesis_output(text: str) -> tuple[str, float]:
    parts = re.split(r"###?\s*FINAL\s*OUTPUT\s*FORMAT", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        final_section = parts[1].strip()
    else:
        parts = re.split(r"###?\s*TASK\s*2(?::|and|synthesis|answer|generation|\s)*", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            task2_section = parts[1].strip()
            task2_parts = re.split(r"###?\s*TASK\s*3", task2_section, flags=re.IGNORECASE)
            final_section = task2_parts[0].strip()
        else:
            final_section = text.strip()

    score_match = re.search(r"(?:confidence|score)\s*(?:score)?\s*[:\-\s]\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+))?", final_section, re.IGNORECASE)
    
    confidence = 0.85
    clean_answer = final_section
    
    if score_match:
        val_str = score_match.group(1)
        max_str = score_match.group(2)
        try:
            val = float(val_str)
            if max_str:
                max_val = float(max_str)
                if max_val > 0:
                    confidence = round(val / max_val, 2)
            else:
                if 0 <= val <= 1:
                    confidence = val
                elif 0 <= val <= 10:
                    confidence = round(val / 10.0, 2)
                elif 0 <= val <= 5:
                    confidence = round(val / 5.0, 2)
        except ValueError:
            pass
        
        # Remove the confidence score line
        lines = clean_answer.split("\n")
        clean_lines = []
        for line in lines:
            if re.search(r"(?:confidence|score)\s*(?:score)?\s*[:\-\s]\s*\d+", line, re.IGNORECASE):
                continue
            clean_lines.append(line)
        clean_answer = "\n".join(clean_lines).strip()

    # Clean leading colons or headers
    clean_answer = re.sub(r"^(?::|and|synthesis|answer|generation|\s)*", "", clean_answer, flags=re.IGNORECASE).strip()
    clean_answer = re.sub(r"^(?:Provide only the synthesized answer from Task 2, followed by a concise confidence score based on your assessment in Task 3\.?|Synthesized Answer:)\s*", "", clean_answer, flags=re.IGNORECASE).strip()
    return clean_answer, confidence

# Test cases representing potential LLM outputs
test_outputs = [
    # Test case 1: Standard structured output
    """### TASK 1: EVIDENCE EXTRACTION
- Reciprocal Rank Fusion (RRF) combines ranked lists from multiple retrieval systems. [1]

### TASK 2: SYNTHESIS AND ANSWER GENERATION
Reciprocal Rank Fusion (RRF) is a method that merges document rankings from heterogeneous retrieval systems by summing reciprocal ranks.

### TASK 3: CRITICAL REVIEW
1. Faithfulness Check: Yes.
2. Completeness Check: Yes.

### FINAL OUTPUT FORMAT
Reciprocal Rank Fusion (RRF) is a rank fusion algorithm that combines multiple ranked lists by summing reciprocal ranks.

Confidence Score: 5/5""",

    # Test case 2: Task 2 only fallback
    """### TASK 1: EVIDENCE
- ChromaDB stores embeddings as dense vectors. [1]

### TASK 2: SYNTHESIS AND ANSWER GENERATION
ChromaDB stores embeddings as dense vectors alongside document text and metadata in named collections.

Confidence Score: 0.9""",

    # Test case 3: Fractional confidence out of 10
    """### FINAL OUTPUT FORMAT
LangGraph is a library used to orchestrate stateful multi-step LLM workflows as directed graphs.

Confidence: 9/10""",

    # Test case 4: Decimals
    """### FINAL OUTPUT FORMAT
Neo4j is a graph database optimized for traversing relationships.

Confidence Score: 0.95"""
]

for idx, test_text in enumerate(test_outputs, 1):
    answer, conf = parse_research_synthesis_output(test_text)
    print(f"=== TEST CASE {idx} ===")
    print(f"Parsed Answer:\n{answer}")
    print(f"Parsed Confidence: {conf}")
    print("-" * 50)
