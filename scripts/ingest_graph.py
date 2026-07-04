import os
import sys
import asyncio
from dotenv import load_dotenv

# Load env variables
load_dotenv(os.path.abspath('.env'))

sys.path.insert(0, os.path.abspath('.'))

from backend.graph_ingestion.ingest import IdempotentGraphIngester

def main():
    ingester = IdempotentGraphIngester()
    try:
        print("Starting ingestion of 'docs' directory into the PropertyGraphIndex...")
        results = ingester.ingest_directory("docs")
        print("\nIngestion results:")
        for res in results:
            print(f"- {res.get('file_name')}: {res.get('status')} (Ingested: {res.get('ingested_chunks')}, Skipped: {res.get('skipped_chunks')})")
    finally:
        ingester.close()

if __name__ == "__main__":
    main()
