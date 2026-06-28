import os
import argparse
import pandas as pd
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv(os.path.abspath('.env'))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer
)

def load_docs(directory: str) -> list:
    documents = []
    print(f"Loading markdown files from '{directory}'...")
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return []
        
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append(Document(page_content=content, metadata={"source": filename}))
            print(f"  Loaded: {filename} ({len(content)} chars)")
    return documents

def main():
    parser = argparse.ArgumentParser(description="Generate RAGAS testset from markdown docs.")
    parser.add_argument("--docs-dir", type=str, default="docs", help="Directory containing markdown files")
    parser.add_argument("--size", type=int, default=12, help="Number of questions to generate")
    parser.add_argument("--output", type=str, default="eval/kinegraph_benchmark_v1.csv", help="Path to save generated CSV")
    args = parser.parse_args()

    # Load and chunk docs
    docs = load_docs(args.docs_dir)
    if not docs:
        print("No documents loaded. Exiting.")
        return

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # Initialize LLM and Embeddings
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not found.")
        return

    print("Initializing RAGAS TestsetGenerator...")
    from openai import OpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings

    base_url = "https://openrouter.ai/api/v1" if (api_key.startswith("sk-or-") or "openrouter" in api_key) else None
    openai_client = OpenAI(api_key=api_key, base_url=base_url)
    ragas_llm = llm_factory(model="gpt-4o-mini", client=openai_client)
    ragas_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings
    )

    # Setup the custom query distribution
    # 40% simple factual lookup, 40% multi-hop reasoning, 20% abstract synthesis/comparison
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 0.4),
        (MultiHopSpecificQuerySynthesizer(llm=ragas_llm), 0.4),
        (MultiHopAbstractQuerySynthesizer(llm=ragas_llm), 0.2)
    ]

    print(f"Generating synthetic test set of size {args.size}...")
    try:
        # Generate testset
        testset = generator.generate_with_langchain_docs(
            documents=chunks,
            testset_size=args.size,
            query_distribution=query_distribution,
            raise_exceptions=True
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Export to CSV
        df = testset.to_pandas()
        df.to_csv(args.output, index=False)
        print(f"\nSuccess! Saved {len(df)} generated benchmark samples to '{args.output}'.")
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
