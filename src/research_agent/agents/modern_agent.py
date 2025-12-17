"""
Modern LangGraph Research Agent with parallel search, RAG, and template-based generation
"""
from typing import TypedDict, List, Dict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio
import os

from ..tools.search_tools import ParallelSearchOrchestrator, SearchResult
from ..rag.rag_system import RAGSystem
from ..templates.report_templates import get_template, ReportContext
from ..utils.cost_tracker import get_tracker


# Define the agent state
class ResearchState(TypedDict):
    """State for the modern research agent"""
    # Input
    question: str
    context: str  # academic, business, product, investment, technical, general
    
    # Refined questions
    refined_questions: List[str]
    
    # Search results
    search_results: Dict[str, List[SearchResult]]
    all_results_flat: List[SearchResult]
    
    # RAG
    rag_system: RAGSystem
    retrieved_docs: str
    citations: List[Dict]
    
    # Generation
    template_sections: Dict[str, str]
    final_report: str
    
    # Feedback loop
    feedback: str
    iteration: int
    max_iterations: int
    
    # Metadata
    retry_count: int
    max_retries: int
    sources_found: bool


class ModernResearchAgent:
    """
    Modern research agent with:
    - Context-aware question refinement
    - Parallel multi-source search
    - RAG with embeddings
    - Template-based generation
    - Feedback loop
    """
    
    def __init__(
        self,
        openai_api_key: str,
        tavily_api_key: str,
        news_api_key: str = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_retries: int = 1,
        max_iterations: int = 2
    ):
        """
        Initialize modern research agent
        
        Args:
            openai_api_key: OpenAI API key
            tavily_api_key: Tavily API key
            news_api_key: News API key (optional)
            model: OpenAI model name
            embedding_model: Sentence transformer model
            max_retries: Max retries for failed searches
            max_iterations: Max feedback iterations
        """
        # LLM setup
        self.llm = ChatOpenAI(
            model=model,
            api_key=openai_api_key,
            temperature=0.7
        )
        
        # Smaller model for question refinement (cheaper)
        self.small_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.8
        )
        
        # Search orchestrator
        self.search_orchestrator = ParallelSearchOrchestrator(
            tavily_key=tavily_api_key,
            news_api_key=news_api_key,
            max_results_per_source=5
        )
        
        # Config
        self.embedding_model = embedding_model
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.cost_tracker = get_tracker()
        
        # Session storage
        self._last_rag_system = None
        self._last_retrieved_docs = ""
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("refine_questions", self.refine_questions)
        workflow.add_node("parallel_search", self.parallel_search)
        workflow.add_node("check_results", self.check_results)
        workflow.add_node("retry_search", self.retry_search)
        workflow.add_node("build_rag", self.build_rag)
        workflow.add_node("retrieve_docs", self.retrieve_docs)
        workflow.add_node("generate_report", self.generate_report)
        workflow.add_node("process_feedback", self.process_feedback)
        
        # Define edges
        workflow.set_entry_point("refine_questions")
        workflow.add_edge("refine_questions", "parallel_search")
        
        # Check if we got results
        workflow.add_conditional_edges(
            "parallel_search",
            self._should_retry,
            {
                "retry": "retry_search",
                "proceed": "check_results"
            }
        )
        
        workflow.add_conditional_edges(
            "retry_search",
            self._should_retry,
            {
                "retry": "check_results",  # After retry, always proceed
                "proceed": "check_results"
            }
        )
        
        workflow.add_edge("check_results", "build_rag")
        workflow.add_edge("build_rag", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "generate_report")
        
        # Feedback loop
        workflow.add_conditional_edges(
            "generate_report",
            self._should_iterate,
            {
                "feedback": "process_feedback",
                "end": END
            }
        )
        
        workflow.add_edge("process_feedback", "retrieve_docs")
        
        return workflow.compile()
    
    async def refine_questions(self, state: ResearchState) -> ResearchState:
        """Refine questions based on context"""
        print(f"\n🎯 Refining questions for context: {state['context']}")
        
        context_descriptions = {
            "academic": "scholarly research with rigorous analysis",
            "business": "business strategy and market analysis",
            "product": "product features, comparisons, and reviews",
            "investment": "investment analysis and financial evaluation",
            "technical": "technical implementation and architecture",
            "general": "comprehensive general information"
        }
        
        context_desc = context_descriptions.get(state['context'], "general information")
        
        prompt = f"""You are a research question specialist. Given a research question and context, generate 3 additional refined questions that will help gather comprehensive information.

Original Question: {state['question']}
Context: {state['context']} ({context_desc})

Generate 3 specific, focused questions that:
1. Cover different aspects of the main question
2. Are appropriate for the {state['context']} context
3. Will help gather diverse, relevant information
4. Are clear and searchable

Format: One question per line, numbered 1-3."""
        
        messages = [
            SystemMessage(content="You are an expert at formulating research questions."),
            HumanMessage(content=prompt)
        ]
        
        response = await asyncio.to_thread(self.small_llm.invoke, messages)
        
        # Track cost
        self.cost_tracker.log_call(
            model=self.small_llm.model_name,
            input_tokens=response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
            output_tokens=response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
            operation="refine_questions"
        )
        
        # Parse questions
        refined = [state['question']]  # Include original
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                question = line.lstrip('0123456789.-) ').strip()
                if question:
                    refined.append(question)
        
        state['refined_questions'] = refined[:4]  # Max 4 total questions
        
        print(f"   ✓ Generated {len(state['refined_questions'])} questions:")
        for i, q in enumerate(state['refined_questions'], 1):
            print(f"      {i}. {q[:80]}...")
        
        return state
    
    async def parallel_search(self, state: ResearchState) -> ResearchState:
        """Execute parallel searches across all sources"""
        print(f"\n🔍 Executing parallel searches across 4 sources")
        
        # Combine questions for search (use first 2 for broader coverage)
        combined_query = " ".join(state['refined_questions'][:2])
        
        # Search all sources in parallel
        results = await self.search_orchestrator.search_all(
            combined_query,
            retry=False  # We handle retry ourselves
        )
        
        state['search_results'] = results
        state['all_results_flat'] = self.search_orchestrator.get_all_results_flat(results)
        state['sources_found'] = len(state['all_results_flat']) > 0
        
        # Print results summary
        for source, source_results in results.items():
            status = "✓" if source_results else "✗"
            print(f"   {status} {source}: {len(source_results)} results")
        
        print(f"   Total results: {len(state['all_results_flat'])}")
        
        return state
    
    async def retry_search(self, state: ResearchState) -> ResearchState:
        """Retry search if no results found"""
        print(f"\n🔄 Retrying search (attempt {state['retry_count'] + 1})")
        
        state['retry_count'] += 1
        
        # Try with different query formulation
        alt_query = state['refined_questions'][0] if state['refined_questions'] else state['question']
        
        results = await self.search_orchestrator.search_all(
            alt_query,
            retry=True  # Enable built-in retry
        )
        
        state['search_results'] = results
        state['all_results_flat'] = self.search_orchestrator.get_all_results_flat(results)
        state['sources_found'] = len(state['all_results_flat']) > 0
        
        print(f"   Retry results: {len(state['all_results_flat'])} total")
        
        return state
    
    async def check_results(self, state: ResearchState) -> ResearchState:
        """Check if we have sufficient results"""
        if state['sources_found']:
            print(f"   ✓ Sufficient results found")
        else:
            print(f"   ⚠ Limited results, proceeding with available data")
        
        return state
    
    async def build_rag(self, state: ResearchState) -> ResearchState:
        """Build RAG system from search results"""
        print(f"\n📚 Building RAG system with embeddings")
        
        # Create RAG system
        rag = RAGSystem(embedding_model=self.embedding_model)
        
        # Add all search results
        if state['all_results_flat']:
            await asyncio.to_thread(rag.add_search_results, state['all_results_flat'])
            print(f"   ✓ Embedded {len(state['all_results_flat'])} documents")
        else:
            print(f"   ⚠ No documents to embed")
        
        state['rag_system'] = rag
        
        # Store in agent for session access
        self._last_rag_system = rag
        
        return state
    
    async def retrieve_docs(self, state: ResearchState) -> ResearchState:
        """Retrieve relevant documents using RAG"""
        print(f"\n📖 Retrieving relevant documents")
        
        # Use all refined questions for retrieval
        context, citations = await asyncio.to_thread(
            state['rag_system'].get_context_for_generation,
            state['refined_questions'],
            max_tokens=6000
        )
        
        state['retrieved_docs'] = context
        state['citations'] = citations
        
        # Store in agent for session access
        self._last_retrieved_docs = context
        
        print(f"   ✓ Retrieved context ({len(context)} chars)")
        print(f"   ✓ Found {len(citations)} unique sources")
        
        return state
    
    async def generate_report(self, state: ResearchState) -> ResearchState:
        """Generate report using template-based approach"""
        print(f"\n📝 Generating {state['context']} report")
        
        # Get template
        template = get_template(state['context'])
        
        # Generate each section
        sections_content = {}
        
        for i, section in enumerate(template.sections, 1):
            print(f"   Generating: {section.name} ({i}/{len(template.sections)})")
            
            # Create prompt
            prompt = template.get_generation_prompt(
                state['question'],
                state['retrieved_docs'],
                section
            )
            
            messages = [
                SystemMessage(content=f"You are an expert {state['context']} writer."),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            
            # Track cost
            self.cost_tracker.log_call(
                model=self.llm.model_name,
                input_tokens=response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                output_tokens=response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
                operation=f"generate_{section.name}"
            )
            
            sections_content[section.name] = response.content
            print(f"      ✓ {section.name} complete")
        
        # Format final report
        report = template.format_report(sections_content)
        
        # Add citations
        if state['citations']:
            report += "\n\n## References\n\n"
            for citation in state['citations']:
                report += f"{citation['id']}. {citation['title']} - {citation['source']}\n"
                report += f"   {citation['url']}\n\n"
        
        state['template_sections'] = sections_content
        state['final_report'] = report
        
        print(f"   ✓ Report complete ({len(report)} chars)")
        
        return state
    
    async def process_feedback(self, state: ResearchState) -> ResearchState:
        """Process user feedback with intelligent information need detection"""
        print(f"\n🔄 Processing feedback (iteration {state['iteration']})")
        
        # Check if feedback requires new information
        needs_new_info = await self._check_needs_new_info(state['feedback'])
        
        if needs_new_info:
            print("   → Feedback requires new information")
            # Do a single targeted Tavily search
            new_search_query = await self._generate_feedback_query(
                state['question'],
                state['feedback']
            )
            
            print(f"   → Searching: {new_search_query}")
            
            # Single Tavily search
            tavily_search = self.search_orchestrator.sources.get("tavily")
            if tavily_search:
                new_results = await tavily_search.search_with_retry(new_search_query)
                
                if new_results:
                    # Add to existing RAG system
                    await asyncio.to_thread(
                        state['rag_system'].add_search_results,
                        new_results
                    )
                    print(f"   ✓ Added {len(new_results)} new results to RAG")
        else:
            print("   → Feedback can be addressed with existing information")
        
        # Re-retrieve with updated context
        # This will automatically use any new embeddings if added
        state['iteration'] += 1
        
        return state
    
    async def _check_needs_new_info(self, feedback: str) -> bool:
        """
        Check if feedback requires new information
        
        Args:
            feedback: User feedback
            
        Returns:
            True if needs new information, False otherwise
        """
        prompt = f"""Analyze this user feedback and determine if it requires NEW INFORMATION from external sources, or if it can be addressed by reorganizing/rephrasing existing information.

User Feedback: {feedback}

Respond with ONLY "YES" if new information is needed, or "NO" if existing information is sufficient.

Examples:
- "Add more details about the economic impact" → NO (rephrasing needed)
- "What about developments in 2024?" → YES (new time-specific information)
- "Include information about quantum entanglement" → YES (new topic)
- "Make the conclusion more concise" → NO (restructuring needed)
- "Can you explain this in simpler terms?" → NO (rephrasing needed)

Response (YES or NO):"""
        
        messages = [
            SystemMessage(content="You are an expert at analyzing information needs."),
            HumanMessage(content=prompt)
        ]
        
        response = await asyncio.to_thread(self.small_llm.invoke, messages)
        
        # Track cost
        self.cost_tracker.log_call(
            model=self.small_llm.model_name,
            input_tokens=response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
            output_tokens=response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
            operation="check_needs_new_info"
        )
        
        return "YES" in response.content.upper()
    
    async def _generate_feedback_query(self, original_question: str, feedback: str) -> str:
        """
        Generate search query based on feedback
        
        Args:
            original_question: Original research question
            feedback: User feedback
            
        Returns:
            Search query string
        """
        prompt = f"""Generate a concise search query (5-10 words) to find information that addresses this feedback.

Original Question: {original_question}
User Feedback: {feedback}

Generate ONE specific search query that will find the missing information:"""
        
        messages = [
            SystemMessage(content="You are an expert at creating search queries."),
            HumanMessage(content=prompt)
        ]
        
        response = await asyncio.to_thread(self.small_llm.invoke, messages)
        
        # Track cost
        self.cost_tracker.log_call(
            model=self.small_llm.model_name,
            input_tokens=response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
            output_tokens=response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
            operation="generate_feedback_query"
        )
        
        return response.content.strip()
    
    def _should_retry(self, state: ResearchState) -> Literal["retry", "proceed"]:
        """Decide whether to retry search"""
        if not state['sources_found'] and state['retry_count'] < state['max_retries']:
            return "retry"
        return "proceed"
    
    def _should_iterate(self, state: ResearchState) -> Literal["feedback", "end"]:
        """Decide whether to process feedback"""
        if state.get('feedback') and state['iteration'] < state['max_iterations']:
            return "feedback"
        return "end"
    
    async def research(
        self,
        question: str,
        context: str = "general",
        feedback: str = None
    ) -> Dict:
        """
        Execute full research workflow
        
        Args:
            question: Research question
            context: Context type (academic, business, etc.)
            feedback: Optional feedback for refinement
            
        Returns:
            Dict with final report and metadata
        """
        print(f"\n{'='*80}")
        print(f"RESEARCH AGENT v2.0")
        print(f"{'='*80}")
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"{'='*80}")
        
        # Initialize state
        initial_state = ResearchState(
            question=question,
            context=context.lower(),
            refined_questions=[],
            search_results={},
            all_results_flat=[],
            rag_system=None,
            retrieved_docs="",
            citations=[],
            template_sections={},
            final_report="",
            feedback=feedback or "",
            iteration=0,
            max_iterations=self.max_iterations,
            retry_count=0,
            max_retries=self.max_retries,
            sources_found=False
        )
        
        # Run graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Store RAG system and retrieved docs for session management
        self._last_rag_system = final_state.get('rag_system')
        self._last_retrieved_docs = final_state.get('retrieved_docs', '')
        
        print(f"\n{'='*80}")
        print(f"RESEARCH COMPLETE")
        print(f"{'='*80}\n")
        
        return {
            "report": final_state['final_report'],
            "sections": final_state['template_sections'],
            "citations": final_state['citations'],
            "questions": final_state['refined_questions'],
            "sources_count": len(final_state['all_results_flat']),
            "context": context,
            "rag_system": final_state.get('rag_system'),  # Include for session
            "retrieved_docs": final_state.get('retrieved_docs', '')  # Include for session
        }


# Factory function
def create_research_agent(
    openai_api_key: str,
    tavily_api_key: str,
    news_api_key: str = None,
    **kwargs
) -> ModernResearchAgent:
    """
    Create research agent instance
    
    Args:
        openai_api_key: OpenAI API key
        tavily_api_key: Tavily API key  
        news_api_key: News API key (optional)
        **kwargs: Additional agent config
        
    Returns:
        ModernResearchAgent instance
    """
    return ModernResearchAgent(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        news_api_key=news_api_key,
        **kwargs
    )
