"""
Smart Feedback Processor
Analyzes feedback to determine if new information is needed or existing RAG data is sufficient
"""
from typing import Dict, Tuple, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio

from ..utils.cost_tracker import get_tracker
from ..utils.session_manager import SessionData


class FeedbackAnalyzer:
    """Analyzes user feedback to determine action needed"""
    
    def __init__(self, openai_api_key: str):
        """
        Initialize feedback analyzer
        
        Args:
            openai_api_key: OpenAI API key
        """
        # Use smallest model for cost efficiency
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.3  # Lower temperature for more consistent analysis
        )
        self.cost_tracker = get_tracker()
    
    async def analyze_feedback(
        self,
        original_question: str,
        feedback: str,
        current_report: str,
        context: str
    ) -> Tuple[Literal["search", "refine"], str]:
        """
        Analyze feedback to determine if new search is needed
        
        Args:
            original_question: Original research question
            feedback: User feedback
            current_report: Current report content
            context: Research context
            
        Returns:
            Tuple of (action, reasoning)
            action: "search" (need new info) or "refine" (use existing)
            reasoning: Explanation of decision
        """
        
        prompt = f"""Analyze this user feedback to determine if NEW information is needed or if the existing research is sufficient.

Original Question: {original_question}
Context: {context}

Current Report Preview:
{current_report[:800]}...

User Feedback:
{feedback}

Determine if the feedback requires:
1. "SEARCH" - New information that's not in the current research (e.g., "add information about X", "what about Y?", "include recent developments")
2. "REFINE" - Just rewriting/reorganizing existing information (e.g., "make it shorter", "focus more on conclusion", "change tone", "add more details from existing sources")

Respond with ONLY one of these formats:
SEARCH: [brief reason why new search is needed]
or
REFINE: [brief reason why existing info is sufficient]

Be conservative - only choose SEARCH if truly new information is needed."""

        messages = [
            SystemMessage(content="You are an expert at analyzing research feedback."),
            HumanMessage(content=prompt)
        ]
        
        response = await asyncio.to_thread(self.llm.invoke, messages)
        
        # Track cost
        self.cost_tracker.log_call(
            model=self.llm.model_name,
            input_tokens=response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
            output_tokens=response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
            operation="analyze_feedback"
        )
        
        # Parse response
        content = response.content.strip()
        
        if content.startswith("SEARCH:"):
            action = "search"
            reasoning = content.replace("SEARCH:", "").strip()
        elif content.startswith("REFINE:"):
            action = "refine"
            reasoning = content.replace("REFINE:", "").strip()
        else:
            # Default to refine if unclear
            action = "refine"
            reasoning = "Unable to determine - using existing information"
        
        return action, reasoning


class FeedbackProcessor:
    """Processes user feedback with smart decision making"""
    
    def __init__(
        self,
        openai_api_key: str,
        tavily_api_key: str
    ):
        """
        Initialize feedback processor
        
        Args:
            openai_api_key: OpenAI API key
            tavily_api_key: Tavily API key (for supplemental search)
        """
        self.analyzer = FeedbackAnalyzer(openai_api_key)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.7
        )
        self.tavily_api_key = tavily_api_key
        self.cost_tracker = get_tracker()
    
    async def process_feedback(
        self,
        session: SessionData,
        feedback: str
    ) -> Dict:
        """
        Process feedback and generate refined report
        
        Args:
            session: Session data with existing RAG
            feedback: User feedback
            
        Returns:
            Dict with refined report and metadata
        """
        print(f"\n🔄 Processing feedback: {feedback[:100]}...")
        
        # Analyze feedback
        action, reasoning = await self.analyzer.analyze_feedback(
            original_question=session.question,
            feedback=feedback,
            current_report=session.final_report,
            context=session.context
        )
        
        print(f"   Decision: {action.upper()}")
        print(f"   Reasoning: {reasoning}")
        
        if action == "search":
            # Need new information
            return await self._process_with_new_search(session, feedback, reasoning)
        else:
            # Use existing RAG data
            return await self._process_with_existing_data(session, feedback, reasoning)
    
    async def _process_with_new_search(
        self,
        session: SessionData,
        feedback: str,
        reasoning: str
    ) -> Dict:
        """Process feedback with new search"""
        from ..tools.search_tools import TavilySearch, SearchResult
        
        print(f"   🔍 Performing supplemental search...")
        
        # Extract search query from feedback
        query_prompt = f"""Based on this feedback, generate a concise search query (3-8 words) to find the missing information:

Feedback: {feedback}
Context: {session.context}

Return ONLY the search query, nothing else."""
        
        messages = [
            SystemMessage(content="You are a search query expert."),
            HumanMessage(content=query_prompt)
        ]
        
        response = await asyncio.to_thread(self.llm.invoke, messages)
        
        self.cost_tracker.log_call(
            model=self.llm.model_name,
            input_tokens=response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
            output_tokens=response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
            operation="extract_search_query"
        )
        
        search_query = response.content.strip()
        print(f"   Query: {search_query}")
        
        # Perform single Tavily search
        tavily = TavilySearch(api_key=self.tavily_api_key, max_results=3)
        new_results = await tavily.search(search_query)
        
        print(f"   ✓ Found {len(new_results)} new results")
        
        # Add to existing RAG
        if new_results:
            session.rag_system.add_search_results(new_results)
            session.search_results.extend(new_results)
        
        # Retrieve with original questions + feedback as query
        all_queries = session.refined_questions + [feedback]
        context, new_citations = await asyncio.to_thread(
            session.rag_system.get_context_for_generation,
            all_queries,
            max_tokens=6000
        )
        
        # Generate refined report
        refined_report = await self._generate_refined_report(
            session=session,
            context=context,
            feedback=feedback,
            reasoning=f"Added new information: {reasoning}"
        )
        
        # Update citations
        all_citations = session.citations + new_citations
        
        return {
            "report": refined_report,
            "citations": all_citations,
            "action_taken": "search",
            "reasoning": reasoning,
            "new_sources": len(new_results)
        }
    
    async def _process_with_existing_data(
        self,
        session: SessionData,
        feedback: str,
        reasoning: str
    ) -> Dict:
        """Process feedback using existing RAG data"""
        print(f"   📚 Using existing research data...")
        
        # Retrieve with all questions + feedback
        all_queries = session.refined_questions + [feedback]
        context, citations = await asyncio.to_thread(
            session.rag_system.get_context_for_generation,
            all_queries,
            max_tokens=6000
        )
        
        # Generate refined report
        refined_report = await self._generate_refined_report(
            session=session,
            context=context,
            feedback=feedback,
            reasoning=f"Refined from existing sources: {reasoning}"
        )
        
        return {
            "report": refined_report,
            "citations": citations,
            "action_taken": "refine",
            "reasoning": reasoning,
            "new_sources": 0
        }
    
    async def _generate_refined_report(
        self,
        session: SessionData,
        context: str,
        feedback: str,
        reasoning: str
    ) -> str:
        """Generate refined report based on feedback"""
        from ..templates.report_templates import get_template
        
        template = get_template(session.context)
        
        # For general context, generate complete report
        if session.context == "general":
            section = template.sections[0]  # Complete Report section
            
            prompt = f"""Refine this research report based on user feedback.

Original Question: {session.question}
Context: {session.context}

Previous Report:
{session.final_report}

User Feedback:
{feedback}

Available Research Context:
{context}

Generate an improved report that addresses the feedback while maintaining quality and structure.
Use the research context to support your points."""

            messages = [
                SystemMessage(content=f"You are an expert {session.context} writer."),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            
            self.cost_tracker.log_call(
                model=self.llm.model_name,
                input_tokens=response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                output_tokens=response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
                operation="refine_general_report"
            )
            
            return response.content
        
        else:
            # For structured contexts, regenerate key sections based on feedback
            sections_content = {}
            
            # Determine which sections to regenerate based on feedback
            priority_sections = self._identify_sections_to_update(feedback, template.sections)
            
            for section in priority_sections[:3]:  # Limit to 3 sections max
                print(f"   Regenerating: {section.name}")
                
                prompt = f"""Refine the "{section.name}" section based on feedback.

Original Question: {session.question}
User Feedback: {feedback}

Research Context:
{context}

Previous {section.name}:
{session.final_report[session.final_report.find(f"## {section.name}"):session.final_report.find(f"## {section.name}") + 500] if f"## {section.name}" in session.final_report else "N/A"}

Generate an improved "{section.name}" section that addresses the feedback."""

                messages = [
                    SystemMessage(content=f"You are an expert {session.context} writer."),
                    HumanMessage(content=prompt)
                ]
                
                response = await asyncio.to_thread(self.llm.invoke, messages)
                
                self.cost_tracker.log_call(
                    model=self.llm.model_name,
                    input_tokens=response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                    output_tokens=response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
                    operation=f"refine_{section.name}"
                )
                
                sections_content[section.name] = response.content
            
            # Merge with existing report
            refined_report = self._merge_sections(session.final_report, sections_content)
            
            return refined_report
    
    def _identify_sections_to_update(self, feedback: str, sections: list) -> list:
        """Identify which sections are most relevant to feedback"""
        # Simple keyword matching - could be enhanced with embeddings
        feedback_lower = feedback.lower()
        
        # Priority keywords for each section type
        keywords = {
            "abstract": ["abstract", "summary", "overview"],
            "introduction": ["introduction", "background", "context"],
            "methodology": ["method", "approach", "how"],
            "findings": ["findings", "results", "data"],
            "discussion": ["discussion", "implications", "significance"],
            "conclusion": ["conclusion", "summary", "takeaway"],
            "executive summary": ["summary", "overview", "key"],
            "recommendations": ["recommend", "suggest", "should"],
        }
        
        scored_sections = []
        for section in sections:
            score = 0
            section_name_lower = section.name.lower()
            
            # Check if section name mentioned
            if section_name_lower in feedback_lower:
                score += 10
            
            # Check keywords
            for key, kwords in keywords.items():
                if key in section_name_lower:
                    for kw in kwords:
                        if kw in feedback_lower:
                            score += 1
            
            scored_sections.append((score, section))
        
        # Sort by score
        scored_sections.sort(reverse=True, key=lambda x: x[0])
        
        return [section for score, section in scored_sections]
    
    def _merge_sections(self, original_report: str, updated_sections: Dict[str, str]) -> str:
        """Merge updated sections into original report"""
        result = original_report
        
        for section_name, new_content in updated_sections.items():
            # Find and replace section
            section_header = f"## {section_name}"
            if section_header in result:
                # Find start and end of section
                start_idx = result.find(section_header)
                # Find next section header
                next_section = result.find("\n## ", start_idx + len(section_header))
                
                if next_section == -1:
                    # Last section
                    result = result[:start_idx] + f"{section_header}\n{new_content}\n"
                else:
                    # Replace section
                    result = result[:start_idx] + f"{section_header}\n{new_content}\n\n" + result[next_section:]
        
        return result


# Factory function
def create_feedback_processor(
    openai_api_key: str,
    tavily_api_key: str
) -> FeedbackProcessor:
    """Create feedback processor instance"""
    return FeedbackProcessor(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key
    )
