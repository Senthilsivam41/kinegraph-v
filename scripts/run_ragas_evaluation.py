#!/usr/bin/env python3
"""
Run RAGAS evaluation against your KineticGraph-Vectra pipeline.

Usage:
    # Offline mode (use pre-computed answers)
    python scripts/run_ragas_evaluation.py \
        --dataset eval/kinegraph_benchmark_v1.csv \
        --mode offline
    
    # Live mode (run through actual RAG workflow)
    python scripts/run_ragas_evaluation.py \
        --dataset eval/kinegraph_benchmark_v1.csv \
        --mode live \
        --max-hops 3 \
        --model gpt-4o-mini

Requirements:
    pip install ragas openai pandas datasets langchain-openai langchain-anthropic
"""
import os
import sys
import argparse
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")


class RAGASEvaluationRunner:
    """High-level wrapper for running RAGAS evaluations."""

    def __init__(self, mode="offline", model="gpt-4o-mini"):
        self.mode = mode
        self.model = model
        self.start_time = datetime.now()

    @staticmethod
    def load_dataset(csv_path: str) -> list[dict]:
        """Load benchmark dataset from CSV (user_input, reference columns)."""
        import pandas as pd
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = {"user_input", "reference"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Benchmark CSV missing columns: {missing}")

        print(f"✓ Loaded dataset from '{csv_path}'")
        return [
            {"question": row["user_input"], "ground_truth": row["reference"]}
            for _, row in df.iterrows()
        ]

    @staticmethod
    def prepare_offline_dataset(raw_data: list[dict], answers: dict[str, str]) -> list[dict]:
        """Prepare offline evaluation dataset with pre-computed answers."""
        import pandas as pd
        records = []
        for item in raw_data:
            question = item["question"]
            ground_truth = item.get("ground_truth", "")

            if question not in answers or not answers[question]:
                print(f"  ⚠️ No pre-computed answer found for query: '{question[:50]}...'")
                continue

            records.append({
                "question": question,
                "answer": answers[question],
                "contexts": [],  # Would need retrieval pipeline to populate this
                "ground_truth": ground_truth,
                "has_ground_truth": bool(ground_truth),
            })

        print(f"✓ Prepared {len(records)} samples for offline evaluation")
        return records

    @staticmethod
    def run_offline_evaluation(dataset: list[dict], output_dir="reports"):
        """Run RAGAS metric computation on pre-computed answers."""
        from eval.ragas_evaluator import RAGASEvaluator
        os.makedirs(output_dir, exist_ok=True)

        print("\n📊 Running offline RAGAS evaluation...")
        evaluator = RAGASEvaluator(model=evaluator.model if hasattr(evaluator, 'model') else "gpt-4o-mini")
        
        records = []
        for idx, sample in enumerate(dataset):
            print(f"  [{idx+1}/{len(dataset)}] Evaluating: {sample['question'][:60]}...")
            
            try:
                scores = evaluator.evaluate_single(
                    question=sample["question"],
                    answer=sample["answer"],
                    contexts=sample.get("contexts", []),
                    ground_truth=sample.get("ground_truth"),
                )
                
                records.append({
                    "question": sample["question"][:80],  # Truncate for readability
                    "eval_latency_ms": scores.get("eval_latency_ms", 0),
                    **scores,
                })
            except Exception as e:
                print(f"    ⚠️ Evaluation failed: {e}")

        if not records:
            print("❌ No samples were evaluated successfully.")
            return None
        
        results_df = __import__("pandas").DataFrame(records)
        
        # Generate report
        report = evaluator.generate_report(results_df)
        
        # Save outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_label = f"v2_{timestamp}"

        results_path = os.path.join(output_dir, f"{run_label}_results.csv")
        report_path = os.path.join(output_dir, f"{run_label}_report.json")

        results_df.to_csv(results_path, index=False)
        
        with open(report_path, "w", encoding="utf-8") as f:
            import json
            json.dump(report, f, indent=2)

        print(f"\n✓ Results saved to {results_path}")
        print(f"✓ Report saved to {report_path}")
        
        return report

    @staticmethod
    def run_live_evaluation(dataset: list[dict], output_dir="reports"):
        """Run live evaluation through the actual RAG workflow."""
        from backend.services.chroma_service import ChromaService
        from backend.services.neo4j_service import Neo4jService
        from backend.core.langgraph_workflow import HybridRAGWorkflow

        os.makedirs(output_dir, exist_ok=True)

        print("\n🚀 Running live evaluation through RAG workflow...")
        
        # Initialize services (must be done before accessing settings)
        chroma = ChromaService()
        neo4j = Neo4jService()

        try:
            from backend.core.config import settings
            workflow = HybridRAGWorkflow(
                chroma_service=chroma,
                neo4j_service=neo4j,
            )

            evaluator = RAGASEvaluator(model=settings.LLM_MODEL)

            print("\n⏳ Running concurrent live workflow and RAGAS evaluation...")
            
            import asyncio
            from backend.app.models import QueryMode
            
            results_df = asyncio.run(evaluator.evaluate_live_workflow(
                workflow=workflow,
                dataset=dataset,
                mode=QueryMode.HYBRID,
                concurrency_limit=3,
            ))

            # Generate report and save
            report = evaluator.generate_report(results_df)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_label = f"v2_{timestamp}"

            results_path = os.path.join(output_dir, f"{run_label}_results.csv")
            report_path = os.path.join(output_dir, f"{run_label}_report.json")

            results_df.to_csv(results_path, index=False)
            
            with open(report_path, "w", encoding="utf-8") as f:
                import json
                json.dump(report, f, indent=2)

            print(f"\n✓ Live evaluation complete!")
            print(f"  Results saved to {results_path}")
            print(f"  Report saved to {report_path}")
            
        except Exception as e:
            print(f"\n❌ Live workflow failed: {e}")
            raise
        
        finally:
            try:
                neo4j.close()
            except Exception:
                pass

    def run(self, dataset_path: str | None = None, output_dir: str = "reports") -> dict | None:
        """Main entry point for evaluation."""
        from eval.ragas_evaluator import RAGASEvaluator
        
        print("=" * 70)
        print(f"📈 Kinegraph-Vectra RAGAS Evaluation ({self.mode} mode)")
        print("=" * 70)

        # Load dataset
        raw_data = self.load_dataset(dataset_path or os.path.join("..", "eval", "kinegraph_benchmark_v1.csv"))
        
        if not raw_data:
            print("❌ No data loaded. Exiting.")
            return None

        # Handle offline vs live mode
        if self.mode == "offline":
            # For offline, we'd need pre-computed answers (usually from a previous run)
            answers_path = os.path.join("..", "eval", "pre_computed_answers.json")
            if os.path.exists(answers_path):
                import json
                with open(answers_path) as f:
                    answers = json.load(f)
                records = self.prepare_offline_dataset(raw_data, answers)
                
                if not records:
                    print("❌ No valid samples prepared.")
                    return None
                
                report = self.run_offline_evaluation(records, output_dir)
            else:
                print("\n⚠️ Pre-computed answers file not found. Running in live mode instead...")
                dataset = raw_data  # Fall back to live evaluation
                report = self.run_live_evaluation(dataset, output_dir)
        else:
            # Live mode - run through actual RAG pipeline
            report = self.run_live_evaluation(raw_data, output_dir)

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation against KineticGraph-Vectra pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run live evaluation (default)
  python scripts/run_ragas_evaluation.py

  # With specific parameters
  python scripts/run_ragas_evaluation.py --max-hops 3 --model gpt-4o-mini

  # Offline mode with pre-computed answers
  python scripts/run_ragas_evaluation.py --mode offline --dataset eval/benchmark.csv

  # Custom output directory
  python scripts/run_ragas_evaluation.py -o reports/my_run
        """,
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Path to benchmark CSV file (default: eval/kinegraph_benchmark_v1.csv)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["live", "offline"],
        default="live",
        help="Evaluation mode: 'live' runs through RAG pipeline, 'offline' uses pre-computed answers"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for evaluation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--max-hops", "-h",
        type=int,
        default=None,
        help="Maximum graph traversal depth (1-5). Defaults to config setting."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="reports",
        help="Directory for saving results and reports"
    )

    args = parser.parse_args()

    # Validate max-hops if provided
    if args.max_hops is not None and not (1 <= args.max_hops <= 5):
        print("Error: --max-hops must be between 1 and 5")
        sys.exit(2)

    runner = RAGASEvaluationRunner(mode=args.mode, model=args.model)

    try:
        report = runner.run(dataset_path=args.dataset, output_dir=args.output_dir)
        
        if report:
            print("\n📊 Evaluation Summary:")
            for metric_name, stats in report.get("per_metric", {}).items():
                status = "✅" if stats["mean"] >= 0.7 else ("⚠️" if stats["mean"] >= 0.5 else "❌")
                print(f"  {status} {metric_name:25s}: {stats['mean']:.4f}")

            composite = report.get("summary", {}).get("overall_composite_score", 0)
            print(f"\n  Overall Composite Score: {composite:.4f}")
            
        else:
            print("\n❌ Evaluation completed with no results.")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
