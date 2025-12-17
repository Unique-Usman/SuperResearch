"""
Cost Tracker for OpenAI API Usage
Modern version with async support
"""
import json
from typing import Dict, Optional
from datetime import datetime
import asyncio


class CostTracker:
    """Tracks API costs and token usage with multiple model support"""
    
    # Pricing per 1M tokens (as of Dec 2024)
    PRICING = {
        "gpt-4o-mini": {
            "input": 0.150,
            "output": 0.600,
        },
        "gpt-4o": {
            "input": 2.50,
            "output": 10.00,
        },
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50,
        }
    }
    
    def __init__(self, max_budget: float = 5.0, log_file: str = "cost_log.json"):
        self.max_budget = max_budget
        self.log_file = log_file
        self.sessions = []
        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "calls": []
        }
        self._lock = asyncio.Lock()
        
    def log_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "unknown"
    ):
        """Log a single API call (sync)"""
        cost = self._calculate_call_cost(model, input_tokens, output_tokens)
        
        call_log = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        }
        
        self.current_session["calls"].append(call_log)
        
        # Check budget
        total_cost = self.get_total_cost()
        if total_cost > self.max_budget:
            raise BudgetExceededError(
                f"Budget exceeded! Total cost: ${total_cost:.4f}, "
                f"Max budget: ${self.max_budget:.2f}"
            )
        
        return cost
    
    async def log_call_async(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "unknown"
    ):
        """Log a single API call (async)"""
        async with self._lock:
            return self.log_call(model, input_tokens, output_tokens, operation)
    
    def _calculate_call_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for a single call"""
        if model not in self.PRICING:
            model = "gpt-4o-mini"
        
        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def get_total_cost(self) -> float:
        """Get total cost for current session"""
        return sum(call["cost"] for call in self.current_session["calls"])
    
    def get_total_tokens(self) -> Dict[str, int]:
        """Get total token counts"""
        return {
            "input": sum(call["input_tokens"] for call in self.current_session["calls"]),
            "output": sum(call["output_tokens"] for call in self.current_session["calls"]),
        }
    
    def can_proceed(self, estimated_tokens: int = 10000) -> bool:
        """Check if we can proceed with another call"""
        estimated_cost = (estimated_tokens / 1_000_000) * (
            self.PRICING["gpt-4o-mini"]["input"] + 
            self.PRICING["gpt-4o-mini"]["output"]
        )
        
        return (self.get_total_cost() + estimated_cost) < self.max_budget
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget"""
        return max(0, self.max_budget - self.get_total_cost())
    
    def save_log(self):
        """Save cost log to file"""
        self.current_session["end_time"] = datetime.now().isoformat()
        self.current_session["total_cost"] = self.get_total_cost()
        self.current_session["total_tokens"] = self.get_total_tokens()
        
        with open(self.log_file, "w") as f:
            json.dump(self.current_session, f, indent=2)
    
    def print_summary(self):
        """Print cost summary"""
        tokens = self.get_total_tokens()
        total_cost = self.get_total_cost()
        
        print("\n" + "="*60)
        print("COST SUMMARY")
        print("="*60)
        print(f"Total API Calls: {len(self.current_session['calls'])}")
        print(f"Input Tokens: {tokens['input']:,}")
        print(f"Output Tokens: {tokens['output']:,}")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Remaining Budget: ${self.get_remaining_budget():.4f}")
        print(f"Budget Utilization: {(total_cost/self.max_budget)*100:.1f}%")
        print("="*60 + "\n")


class BudgetExceededError(Exception):
    """Raised when API budget is exceeded"""
    pass


# Global cost tracker instance
_tracker: Optional[CostTracker] = None


def get_tracker() -> CostTracker:
    """Get global cost tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def init_tracker(max_budget: float = 5.0, log_file: str = "cost_log.json"):
    """Initialize global cost tracker"""
    global _tracker
    _tracker = CostTracker(max_budget, log_file)
    return _tracker
