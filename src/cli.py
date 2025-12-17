"""
CLI interface for Research Agent
"""
import asyncio
import argparse
import sys
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research_agent.agents.modern_agent import create_research_agent
from research_agent.utils.cost_tracker import init_tracker, get_tracker
from research_agent.templates.report_templates import ReportContext


async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Predli Research Agent v2.0 - Advanced research with parallel search and RAG"
    )
    
    parser.add_argument(
        "question",
        type=str,
        help="Research question"
    )
    
    parser.add_argument(
        "-c", "--context",
        type=str,
        default="general",
        choices=[ctx.value for ctx in ReportContext],
        help="Research context type"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="report.md",
        help="Output file path"
    )
    
    parser.add_argument(
        "-b", "--budget",
        type=float,
        default=5.0,
        help="Maximum budget in USD"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model for RAG"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum search retries"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    news_key = os.getenv("NEWS_API_KEY")
    
    if not openai_key or not tavily_key:
        print("Error: OPENAI_API_KEY and TAVILY_API_KEY must be set in .env file")
        return 1
    
    # Initialize cost tracker
    init_tracker(max_budget=args.budget, log_file="cost_log.json")
    
    print(f"\n{'='*80}")
    print(f"PREDLI RESEARCH AGENT v2.0")
    print(f"{'='*80}")
    print(f"Question: {args.question}")
    print(f"Context: {args.context}")
    print(f"Budget: ${args.budget:.2f}")
    print(f"Model: {args.model}")
    print(f"{'='*80}\n")
    
    try:
        # Create agent
        agent = create_research_agent(
            openai_api_key=openai_key,
            tavily_api_key=tavily_key,
            news_api_key=news_key,
            model=args.model,
            embedding_model=args.embedding_model,
            max_retries=args.max_retries
        )
        
        # Execute research
        result = await agent.research(
            question=args.question,
            context=args.context
        )
        
        # Save report
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"# {args.question}\n\n")
            f.write(result["report"])
        
        print(f"\n{'='*80}")
        print(f"RESEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Report saved: {args.output}")
        print(f"Sources used: {result['sources_count']}")
        print(f"Citations: {len(result['citations'])}")
        print(f"{'='*80}\n")
        
        # Print cost summary
        tracker = get_tracker()
        tracker.print_summary()
        tracker.save_log()
        
        return 0
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
