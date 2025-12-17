"""
Enhanced FastAPI Application with Sessions, PDF, and Token Tracking
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import asyncio
from datetime import datetime
import tempfile
import numpy as np
import json

from ..agents.modern_agent import create_research_agent
from ..utils.cost_tracker import init_tracker, get_tracker
from ..utils.session_manager import get_session_manager, init_session_manager
from ..templates.report_templates import ReportContext
from ..utils.pdf_generator import PDFGenerator


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


# Load environment
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Predli Research Agent API v2.0",
    description="Advanced research agent with sessions, feedback, and PDF generation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize trackers
init_tracker(max_budget=float(os.getenv("MAX_BUDGET", "5.0")))
init_session_manager(session_ttl_hours=24)

# Global agent instance (lazy loaded)
_agent = None
_pdf_generator = None


def get_agent():
    """Get or create agent instance"""
    global _agent
    if _agent is None:
        openai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        news_key = os.getenv("NEWS_API_KEY")
        
        if not openai_key or not tavily_key:
            raise ValueError("OPENAI_API_KEY and TAVILY_API_KEY must be set")
        
        _agent = create_research_agent(
            openai_api_key=openai_key,
            tavily_api_key=tavily_key,
            news_api_key=news_key,
            model=os.getenv("MODEL", "gpt-4o-mini")
        )
    
    return _agent


def get_pdf_generator():
    """Get or create PDF generator"""
    global _pdf_generator
    if _pdf_generator is None:
        _pdf_generator = PDFGenerator(output_dir="/tmp/reports")
    return _pdf_generator


# Pydantic models
class ResearchRequest(BaseModel):
    """Research request model"""
    question: str = Field(..., description="Research question", min_length=5)
    context: str = Field(
        default="general",
        description="Research context: academic, business, product, investment, technical, or general"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the latest developments in quantum computing?",
                "context": "technical"
            }
        }


class FeedbackRequest(BaseModel):
    """Feedback request model"""
    session_id: str = Field(..., description="Session ID from initial research")
    feedback: str = Field(..., description="User feedback for refinement", min_length=5)
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc-123-def-456",
                "feedback": "Please add more information about recent 2024 developments"
            }
        }


class ResearchResponse(BaseModel):
    """Research response model with session and tokens"""
    session_id: str
    report: str
    sections: Dict[str, str]
    citations: List[Dict[str, Any]]  # More flexible for various citation formats
    questions: List[str]
    sources_count: int
    context: str
    cost: float
    tokens_used: Dict[str, int]
    
    class Config:
        # Allow extra fields and be more permissive
        extra = "allow"
        arbitrary_types_allowed = True


class FeedbackResponse(BaseModel):
    """Feedback response model"""
    session_id: str
    report: str
    sections: Dict[str, str]
    citations: List[Dict[str, Any]]  # More flexible
    needs_new_info: bool
    cost: float
    tokens_used: Dict[str, int]


class PDFRequest(BaseModel):
    """PDF generation request"""
    session_id: str = Field(..., description="Session ID")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    budget_remaining: float
    active_sessions: int


# API Endpoints

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Predli Research Agent API v2.0",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Session-based research",
            "Intelligent feedback",
            "PDF generation",
            "Token tracking",
            "4 parallel search sources"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    tracker = get_tracker()
    session_manager = get_session_manager()
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        budget_remaining=tracker.get_remaining_budget(),
        active_sessions=len(session_manager.sessions)
    )


@app.get("/contexts", response_model=List[str])
async def get_contexts():
    """Get available research contexts"""
    return [ctx.value for ctx in ReportContext]


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Execute research query with session creation
    
    Args:
        request: Research request with question and context
        
    Returns:
        Research response with session ID and token info
    """
    try:
        # Get agent and tracker
        agent = get_agent()
        tracker = get_tracker()
        session_manager = get_session_manager()
        
        # Track initial state
        start_cost = tracker.get_total_cost()
        start_tokens = tracker.get_total_tokens()
        
        # Execute research (this returns a dict with all results)
        print(f"\n🔬 Starting research: {request.question}")
        print(f"   Context: {request.context}")
        
        result = await agent.research(
            question=request.question,
            context=request.context
        )
        
        print(f"✅ Research complete")
        print(f"   Sources found: {result.get('sources_count', 0)}")
        
        # Calculate usage
        end_cost = tracker.get_total_cost()
        end_tokens = tracker.get_total_tokens()
        
        # Create session with RAG state from result
        session_id = session_manager.create_session(
            question=request.question,
            context=request.context,
            rag_system=result.get("rag_system"),
            retrieved_docs=result.get("retrieved_docs", ""),
            citations=result["citations"],
            refined_questions=result["questions"],
            report=result["report"],
            sections=result["sections"]
        )
        
        print(f"   Session created: {session_id}")
        
        # Debug: Print what we're about to return
        print(f"\n📊 Preparing response:")
        print(f"   - Report length: {len(result['report'])} chars")
        print(f"   - Sections: {len(result['sections'])} items")
        print(f"   - Citations: {len(result['citations'])} items")
        print(f"   - Questions: {len(result['questions'])} items")
        print(f"   - Sources: {result['sources_count']}")
        print(f"   - Cost: ${end_cost - start_cost:.4f}")
        
        tokens_used_calc = {
            "input": end_tokens["input"] - start_tokens["input"],
            "output": end_tokens["output"] - start_tokens["output"],
            "total": (end_tokens["input"] + end_tokens["output"]) - 
                    (start_tokens["input"] + start_tokens["output"])
        }
        print(f"   - Tokens: {tokens_used_calc}")
        
        # Clean and validate citations before creating response
        cleaned_citations = []
        for i, citation in enumerate(result["citations"]):
            try:
                cleaned_citations.append({
                    "id": str(citation.get("id", i)),
                    "title": str(citation.get("title", "Unknown")),
                    "url": str(citation.get("url", "")),
                    "source": str(citation.get("source", "Unknown"))
                })
            except Exception as e:
                print(f"   ⚠️  Warning: Skipping invalid citation {i}: {e}")
        
        print(f"   ✓ Cleaned {len(cleaned_citations)} citations")
        
        # Try to create response with explicit error catching
        try:
            response = ResearchResponse(
                session_id=session_id,
                report=result["report"],
                sections=result["sections"],
                citations=cleaned_citations,
                questions=result["questions"],
                sources_count=result["sources_count"],
                context=result["context"],
                cost=end_cost - start_cost,
                tokens_used=tokens_used_calc
            )
            print(f"✅ Response object created successfully")
            
            # Test JSON serialization
            import json
            try:
                test_json = json.dumps(response.model_dump())
                print(f"✅ Response is JSON serializable ({len(test_json)} chars)")
            except Exception as json_error:
                print(f"❌ JSON serialization failed: {json_error}")
                raise
            
            return response
        except Exception as validation_error:
            print(f"\n❌ VALIDATION ERROR:")
            print(f"   Type: {type(validation_error).__name__}")
            print(f"   Message: {str(validation_error)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Response validation failed: {str(validation_error)}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/stream")
async def research_stream(request: ResearchRequest):
    """
    Execute research with real-time progress streaming (SSE)
    
    Args:
        request: Research request with question and context
        
    Returns:
        Server-Sent Events stream with progress updates
    """
    async def generate_stream():
        try:
            # Get agent and tracker
            agent = get_agent()
            tracker = get_tracker()
            session_manager = get_session_manager()
            
            # Track initial state
            start_cost = tracker.get_total_cost()
            start_tokens = tracker.get_total_tokens()
            
            # Send start event
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'init', 'message': '🚀 Starting research...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Refining questions
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'refining', 'message': f'🎯 Refining questions for {request.context} context...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Start research (this will take time)
            research_task = asyncio.create_task(agent.research(
                question=request.question,
                context=request.context
            ))
            
            # Simulate progress updates while research runs
            stages = [
                ('refining', '✓ Questions refined, preparing searches...'),
                ('searching', '🔍 Searching across 4 parallel sources...'),
                ('searching', '✓ Tavily search complete'),
                ('searching', '✓ ArXiv search complete'),
                ('searching', '✓ Wikipedia search complete'),
                ('embedding', '📚 Building knowledge base with embeddings...'),
                ('embedding', '✓ Embedded documents into vector database'),
                ('retrieving', '📖 Retrieving most relevant information...'),
                ('generating', f'📝 Generating {request.context} report...')
            ]
            
            stage_delay = 2  # Send update every 2 seconds
            current_stage = 0
            
            while not research_task.done():
                await asyncio.sleep(stage_delay)
                if current_stage < len(stages):
                    stage, message = stages[current_stage]
                    yield f"data: {json.dumps({'type': 'progress', 'stage': stage, 'message': message})}\n\n"
                    current_stage += 1
            
            # Get result
            result = await research_task
            
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'finalizing', 'message': '✓ Report generation complete!'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Calculate usage
            end_cost = tracker.get_total_cost()
            end_tokens = tracker.get_total_tokens()
            
            # Clean citations
            cleaned_citations = []
            for i, citation in enumerate(result["citations"]):
                try:
                    cleaned_citations.append({
                        "id": str(citation.get("id", i)),
                        "title": str(citation.get("title", "Unknown")),
                        "url": str(citation.get("url", "")),
                        "source": str(citation.get("source", "Unknown"))
                    })
                except Exception as e:
                    print(f"   ⚠️  Warning: Skipping invalid citation {i}: {e}")
            
            tokens_used_calc = {
                "input": int(end_tokens["input"] - start_tokens["input"]),
                "output": int(end_tokens["output"] - start_tokens["output"]),
                "total": int((end_tokens["input"] + end_tokens["output"]) - 
                        (start_tokens["input"] + start_tokens["output"]))
            }
            
            # Create session
            session_id = session_manager.create_session(
                question=request.question,
                context=request.context,
                rag_system=result.get("rag_system"),
                retrieved_docs=result.get("retrieved_docs", ""),
                citations=cleaned_citations,
                refined_questions=result["questions"],
                report=result["report"],
                sections=result["sections"]
            )
            
            # Build completion message separately to avoid nested quote issues
            completion_message = f'✅ Research complete! Found {result["sources_count"]} sources'
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'complete', 'message': completion_message})}\n\n"
            
            # Send completion event with full data
            completion_data = {
                'type': 'complete',
                'session_id': session_id,
                'report': result["report"],
                'sections': result["sections"],
                'citations': cleaned_citations,
                'questions': result["questions"],
                'sources_count': result["sources_count"],
                'context': result["context"],
                'cost': float(end_cost - start_cost),
                'tokens_used': tokens_used_calc
            }
            
            yield f"data: {json.dumps(completion_data)}\n\n"
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"❌ Stream error: {e}")
            print(error_trace)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for report refinement with intelligent info detection
    
    Args:
        request: Feedback request with session ID and feedback
        
    Returns:
        Refined report with token usage
    """
    try:
        # Get session
        session_manager = get_session_manager()
        session = session_manager.get_session(request.session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        # Get agent and tracker
        agent = get_agent()
        tracker = get_tracker()
        
        # Track initial state
        start_cost = tracker.get_total_cost()
        start_tokens = tracker.get_total_tokens()
        
        # Check if new info needed
        needs_new_info = await agent._check_needs_new_info(request.feedback)
        
        session.add_feedback(request.feedback, needs_new_info)
        
        # If needs new info, do targeted search
        if needs_new_info:
            query = await agent._generate_feedback_query(
                session.question,
                request.feedback
            )
            
            # Single Tavily search
            tavily_search = agent.search_orchestrator.sources.get("tavily")
            if tavily_search:
                new_results = await tavily_search.search_with_retry(query)
                
                if new_results and session.rag_system:
                    await asyncio.to_thread(
                        session.rag_system.add_search_results,
                        new_results
                    )
        
        # Re-retrieve and regenerate
        if session.rag_system:
            # Get updated context
            retrieved_docs, citations = await asyncio.to_thread(
                session.rag_system.get_context_for_generation,
                session.refined_questions,
                max_tokens=6000
            )
            
            # Generate updated report (simplified - reuse template)
            from ..templates.report_templates import get_template
            template = get_template(session.context)
            
            # Regenerate with feedback
            feedback_prompt = f"""Original Question: {session.question}

User Feedback: {request.feedback}

Context:
{retrieved_docs}

Address the feedback while maintaining the report structure. Be specific and incorporate the feedback thoroughly."""
            
            # Generate updated content (simplified for demonstration)
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
            
            sections_content = {}
            for section in template.sections[:3]:  # Regenerate first 3 sections for speed
                prompt = template.get_generation_prompt(
                    f"{session.question}\n\nUser Feedback: {request.feedback}",
                    retrieved_docs,
                    section
                )
                
                response = await asyncio.to_thread(
                    llm.invoke,
                    [SystemMessage(content=f"You are an expert {session.context} writer."),
                     HumanMessage(content=prompt)]
                )
                
                tracker.log_call(
                    model="gpt-4o-mini",
                    input_tokens=response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                    output_tokens=response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
                    operation=f"feedback_generate_{section.name}"
                )
                
                sections_content[section.name] = response.content
            
            # Format report
            report = template.format_report(sections_content)
            
            # Add citations
            if citations:
                report += "\n\n## References\n\n"
                for citation in citations:
                    report += f"{citation['id']}. {citation['title']} - {citation['source']}\n"
                    report += f"   {citation['url']}\n\n"
            
            # Update session
            session_manager.update_session(
                request.session_id,
                report=report,
                sections=sections_content,
                retrieved_docs=retrieved_docs,
                citations=citations
            )
        else:
            report = session.report
            sections_content = session.sections
            citations = session.citations
        
        # Calculate usage
        end_cost = tracker.get_total_cost()
        end_tokens = tracker.get_total_tokens()
        
        # Convert all numpy types to native Python types
        cleaned_report = convert_numpy_types(report)
        cleaned_sections = convert_numpy_types(sections_content)
        cleaned_citations = convert_numpy_types(citations)
        
        tokens_used_calc = {
            "input": int(end_tokens["input"] - start_tokens["input"]),
            "output": int(end_tokens["output"] - start_tokens["output"]),
            "total": int((end_tokens["input"] + end_tokens["output"]) - 
                    (start_tokens["input"] + start_tokens["output"]))
        }
        
        return FeedbackResponse(
            session_id=request.session_id,
            report=cleaned_report,
            sections=cleaned_sections,
            citations=cleaned_citations,
            needs_new_info=needs_new_info,
            cost=float(end_cost - start_cost),
            tokens_used=tokens_used_calc
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pdf")
async def generate_pdf(request: PDFRequest):
    """
    Generate PDF from session with proper markdown formatting
    
    Args:
        request: PDF request with session ID
        
    Returns:
        PDF file
    """
    try:
        # Get session
        session_manager = get_session_manager()
        session = session_manager.get_session(request.session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        # Convert markdown to HTML
        import markdown2
        from weasyprint import HTML
        import tempfile
        
        # Convert report markdown to HTML with extras
        report_html = markdown2.markdown(
            session.report, 
            extras=['fenced-code-blocks', 'tables', 'break-on-newline']
        )
        
        # Build complete HTML with improved styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{session.question}</title>
            <style>
                @page {{
                    size: A4;
                    margin: 2.5cm;
                    @bottom-right {{
                        content: "Page " counter(page) " of " counter(pages);
                        font-size: 9pt;
                        color: #666;
                    }}
                }}
                body {{
                    font-family: 'Georgia', 'Times New Roman', serif;
                    font-size: 11pt;
                    line-height: 1.8;
                    color: #333;
                }}
                h1 {{
                    color: #cb4f2b;
                    font-size: 24pt;
                    border-bottom: 3px solid #cb4f2b;
                    padding-bottom: 15px;
                    margin-bottom: 30px;
                    font-weight: bold;
                }}
                h2 {{
                    color: #cb4f2b;
                    font-size: 18pt;
                    margin-top: 35px;
                    margin-bottom: 15px;
                    border-bottom: 2px solid #ddd;
                    padding-bottom: 8px;
                    font-weight: bold;
                }}
                h3 {{
                    color: #555;
                    font-size: 14pt;
                    margin-top: 25px;
                    margin-bottom: 12px;
                    font-weight: bold;
                }}
                p {{
                    margin-bottom: 12pt;
                    text-align: justify;
                }}
                .metadata {{
                    background: #f8f8f8;
                    padding: 20px;
                    border-left: 4px solid #cb4f2b;
                    margin-bottom: 30px;
                    font-size: 10pt;
                    color: #666;
                }}
                .metadata p {{
                    margin: 5px 0;
                }}
                .citations {{
                    margin-top: 40px;
                }}
                .citation {{
                    font-size: 9pt;
                    color: #555;
                    padding: 10px 0;
                    border-bottom: 1px solid #eee;
                    margin-bottom: 10px;
                }}
                .citation strong {{
                    color: #cb4f2b;
                }}
                .source {{
                    color: #888;
                    font-style: italic;
                }}
                .url {{
                    color: #0066cc;
                    text-decoration: none;
                    font-size: 8pt;
                    word-break: break-all;
                }}
                code {{
                    background: #f5f5f5;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', 'Consolas', monospace;
                    font-size: 10pt;
                    color: #d14;
                }}
                pre {{
                    background: #f8f8f8;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #cb4f2b;
                    overflow-x: auto;
                    font-size: 9pt;
                    line-height: 1.4;
                }}
                pre code {{
                    background: none;
                    padding: 0;
                    color: #333;
                }}
                ul, ol {{
                    margin-left: 20px;
                    margin-bottom: 15px;
                }}
                li {{
                    margin-bottom: 8px;
                }}
                strong {{
                    font-weight: bold;
                    color: #222;
                }}
                em {{
                    font-style: italic;
                }}
                blockquote {{
                    border-left: 4px solid #ddd;
                    margin-left: 0;
                    padding-left: 20px;
                    color: #666;
                    font-style: italic;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th {{
                    background: #cb4f2b;
                    color: white;
                    padding: 10px;
                    text-align: left;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
            </style>
        </head>
        <body>
            <h1>{session.question}</h1>
            <div class="metadata">
                <p><strong>Research Context:</strong> {session.context.title()}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
                <p><strong>Total Sources:</strong> {len(session.citations)} sources analyzed</p>
                <p><strong>Research Questions:</strong> {len(session.refined_questions)} queries used</p>
            </div>
            
            {report_html}
        </body>
        </html>
        """
        
        # Generate PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            HTML(string=html_content).write_pdf(tmp.name)
            pdf_path = tmp.name
        
        # Return PDF file
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"research_report_{session.context}.pdf",
            background=None  # Don't delete file in background
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "question": session.question,
        "context": session.context,
        "created_at": session.created_at.isoformat(),
        "last_accessed": session.last_accessed.isoformat(),
        "feedback_count": len(session.feedback_history),
        "sources_count": len(session.citations)
    }


@app.get("/cost", response_model=Dict)
async def get_cost_summary():
    """Get cost usage summary"""
    tracker = get_tracker()
    tokens = tracker.get_total_tokens()
    
    return {
        "total_cost": tracker.get_total_cost(),
        "remaining_budget": tracker.get_remaining_budget(),
        "max_budget": tracker.max_budget,
        "utilization_percent": (tracker.get_total_cost() / tracker.max_budget) * 100,
        "api_calls": len(tracker.current_session["calls"]),
        "tokens": {
            "input": tokens["input"],
            "output": tokens["output"],
            "total": tokens["input"] + tokens["output"]
        }
    }


# Error handlers
@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    import traceback
    print(f"\n❌ Pydantic ValidationError:")
    print(f"   Request: {request.url}")
    print(f"   Errors: {exc.errors()}")
    traceback.print_exc()
    return JSONResponse(
        status_code=422,
        content={"error": "Validation failed", "details": exc.errors()}
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    import traceback
    print(f"\n❌ ValueError caught by global handler:")
    print(f"   Error: {str(exc)}")
    print(f"   Request: {request.url}")
    traceback.print_exc()
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "type": "ValueError"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Starting Predli Research Agent API v2.0")
    print(f"   Max budget: ${get_tracker().max_budget:.2f}")
    print(f"   Model: {os.getenv('MODEL', 'gpt-4o-mini')}")
    print(f"   Docs: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n🛑 Shutting down Research Agent API")
    tracker = get_tracker()
    tracker.save_log()
    tracker.print_summary()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
