"""
Session Manager for storing retrieved documents and research state
"""
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import asyncio
from ..rag.rag_system import RAGSystem


@dataclass
class ResearchSession:
    """Research session with stored state"""
    session_id: str
    question: str
    context: str
    rag_system: Optional[RAGSystem]  # Make optional
    retrieved_docs: str
    citations: List[Dict]
    refined_questions: List[str]
    report: str
    sections: Dict[str, str]
    created_at: datetime
    last_accessed: datetime
    feedback_history: List[Dict] = field(default_factory=list)
    
    def update_access(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
    
    def add_feedback(self, feedback: str, needs_new_info: bool):
        """Add feedback to history"""
        self.feedback_history.append({
            "feedback": feedback,
            "needs_new_info": needs_new_info,
            "timestamp": datetime.now().isoformat()
        })


class SessionManager:
    """Manages research sessions with TTL"""
    
    def __init__(self, session_ttl_hours: int = 24):
        """
        Initialize session manager
        
        Args:
            session_ttl_hours: Time-to-live for sessions in hours
        """
        self.sessions: Dict[str, ResearchSession] = {}
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self._cleanup_task = None
    
    def create_session(
        self,
        question: str,
        context: str,
        rag_system: RAGSystem,
        retrieved_docs: str,
        citations: List[Dict],
        refined_questions: List[str],
        report: str,
        sections: Dict[str, str]
    ) -> str:
        """
        Create a new research session
        
        Args:
            question: Research question
            context: Research context
            rag_system: RAG system with embedded documents
            retrieved_docs: Retrieved document context
            citations: List of citations
            refined_questions: Refined questions used
            report: Generated report
            sections: Report sections
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        session = ResearchSession(
            session_id=session_id,
            question=question,
            context=context,
            rag_system=rag_system,
            retrieved_docs=retrieved_docs,
            citations=citations,
            refined_questions=refined_questions,
            report=report,
            sections=sections,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        self.sessions[session_id] = session
        
        # Start cleanup task if not running
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """
        Get session by ID
        
        Args:
            session_id: Session ID
            
        Returns:
            ResearchSession or None
        """
        session = self.sessions.get(session_id)
        if session:
            session.update_access()
        return session
    
    def update_session(
        self,
        session_id: str,
        report: str,
        sections: Dict[str, str],
        retrieved_docs: str = None,
        citations: List[Dict] = None
    ):
        """
        Update session with new report
        
        Args:
            session_id: Session ID
            report: Updated report
            sections: Updated sections
            retrieved_docs: Updated retrieved docs (optional)
            citations: Updated citations (optional)
        """
        session = self.get_session(session_id)
        if session:
            session.report = report
            session.sections = sections
            if retrieved_docs:
                session.retrieved_docs = retrieved_docs
            if citations:
                session.citations = citations
            session.update_access()
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_accessed > self.session_ttl
        ]
        
        for sid in expired:
            del self.sessions[sid]
        
        if expired:
            print(f"Cleaned up {len(expired)} expired sessions")
    
    async def _cleanup_loop(self):
        """Background task to cleanup expired sessions"""
        while True:
            await asyncio.sleep(3600)  # Run every hour
            self.cleanup_expired_sessions()
    
    def get_stats(self) -> Dict:
        """Get session statistics"""
        return {
            "total_sessions": len(self.sessions),
            "oldest_session": min(
                (s.created_at for s in self.sessions.values()),
                default=None
            ),
            "newest_session": max(
                (s.created_at for s in self.sessions.values()),
                default=None
            )
        }


# Global session manager
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get global session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(session_ttl_hours=24)
    return _session_manager


def init_session_manager(session_ttl_hours: int = 24) -> SessionManager:
    """Initialize global session manager"""
    global _session_manager
    _session_manager = SessionManager(session_ttl_hours=session_ttl_hours)
    return _session_manager
