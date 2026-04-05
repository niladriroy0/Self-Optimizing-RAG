"""
observability/cost_tracker.py

Tracks LLM token usage and estimated cost per request.
- Accumulates across the session in-memory.
- Persists totals to logs/cost_log.json on every update so data
  survives restarts.
- Supports per-model pricing tables (easily extensible).
"""

import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Per-model token pricing (USD per 1 000 tokens).
# These are approximate reference values; swap in your actual numbers.
# For local Ollama models the cost is $0 – they're still tracked for
# completeness (token counts are useful for latency/resource budgeting).
# ---------------------------------------------------------------------------
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # format: "model-name": {"input": $/1k tokens, "output": $/1k tokens}
    "gpt-4":            {"input": 0.03,   "output": 0.06},
    "gpt-4-turbo":      {"input": 0.01,   "output": 0.03},
    "gpt-3.5-turbo":    {"input": 0.0015, "output": 0.002},
    "claude-3-opus":    {"input": 0.015,  "output": 0.075},
    "claude-3-sonnet":  {"input": 0.003,  "output": 0.015},
    "claude-3-haiku":   {"input": 0.00025,"output": 0.00125},
    # Local Ollama models – zero cost
    "llama3":           {"input": 0.0,    "output": 0.0},
    "mistral":          {"input": 0.0,    "output": 0.0},
    "deepseek-r1":      {"input": 0.0,    "output": 0.0},
    "phi3":             {"input": 0.0,    "output": 0.0},
    # Fallback default (unknown model)
    "__default__":      {"input": 0.0,    "output": 0.0},
}

# Where totals are persisted between restarts
_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "cost_log.json"


class CostTracker:
    """Thread-safe, persistent cost / token tracker."""

    def __init__(self, log_path: Path = _LOG_PATH):
        self._lock = threading.Lock()
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Session-level accumulators (reset on every process start)
        self._session_requests: int = 0
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._session_cost_usd: float = 0.0

        # Per-model breakdown for this session
        self._by_model: Dict[str, Dict[str, Any]] = {}

        # Load historical totals from disk
        self._totals = self._load_totals()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a single LLM call.

        Parameters
        ----------
        model         : model name (used for pricing lookup)
        input_tokens  : prompt token count
        output_tokens : completion token count
        metadata      : optional dict with extra info (query_type, etc.)

        Returns
        -------
        dict with cost breakdown for this single call.
        """
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["__default__"])
        input_cost  = (input_tokens  / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost  = input_cost + output_cost

        entry = {
            "timestamp":     datetime.utcnow().isoformat() + "Z",
            "model":         model,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  input_tokens + output_tokens,
            "cost_usd":      round(total_cost, 8),
            "metadata":      metadata or {},
        }

        with self._lock:
            # Session accumulators
            self._session_requests       += 1
            self._session_input_tokens   += input_tokens
            self._session_output_tokens  += output_tokens
            self._session_cost_usd       += total_cost

            # Per-model session breakdown
            if model not in self._by_model:
                self._by_model[model] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            m = self._by_model[model]
            m["requests"]       += 1
            m["input_tokens"]   += input_tokens
            m["output_tokens"]  += output_tokens
            m["cost_usd"]       += total_cost

            # Persist updated totals
            self._totals["total_requests"]       += 1
            self._totals["total_input_tokens"]   += input_tokens
            self._totals["total_output_tokens"]  += output_tokens
            self._totals["total_cost_usd"]       += total_cost
            self._totals["last_updated"]          = entry["timestamp"]

            by_model_totals = self._totals.setdefault("by_model", {})
            if model not in by_model_totals:
                by_model_totals[model] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            t = by_model_totals[model]
            t["requests"]       += 1
            t["input_tokens"]   += input_tokens
            t["output_tokens"]  += output_tokens
            t["cost_usd"]       += total_cost

            self._save_totals()

        return entry

    def get_session_summary(self) -> Dict[str, Any]:
        """Return aggregated stats for the current process session."""
        with self._lock:
            return {
                "session_requests":      self._session_requests,
                "session_input_tokens":  self._session_input_tokens,
                "session_output_tokens": self._session_output_tokens,
                "session_total_tokens":  self._session_input_tokens + self._session_output_tokens,
                "session_cost_usd":      round(self._session_cost_usd, 6),
                "by_model":              dict(self._by_model),
            }

    def get_all_time_totals(self) -> Dict[str, Any]:
        """Return persisted all-time totals from cost_log.json."""
        with self._lock:
            return dict(self._totals)

    def reset_session(self) -> None:
        """Clear session-only counters (does not touch the persisted file)."""
        with self._lock:
            self._session_requests      = 0
            self._session_input_tokens  = 0
            self._session_output_tokens = 0
            self._session_cost_usd      = 0.0
            self._by_model              = {}

    # ------------------------------------------------------------------
    # CONVENIENCE — estimate tokens without calling the LLM
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough token estimate: ~4 chars per token (GPT-style rule of thumb).
        Good enough for budgeting; replace with tiktoken if precision matters.
        """
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def _load_totals(self) -> Dict[str, Any]:
        if self._log_path.exists():
            try:
                with open(self._log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Ensure all expected keys exist (handles schema additions)
                data.setdefault("total_requests", 0)
                data.setdefault("total_input_tokens", 0)
                data.setdefault("total_output_tokens", 0)
                data.setdefault("total_cost_usd", 0.0)
                data.setdefault("last_updated", None)
                data.setdefault("by_model", {})
                return data
            except (json.JSONDecodeError, OSError):
                pass  # Corrupt file – start fresh

        return {
            "total_requests":       0,
            "total_input_tokens":   0,
            "total_output_tokens":  0,
            "total_cost_usd":       0.0,
            "last_updated":         None,
            "by_model":             {},
        }

    def _save_totals(self) -> None:
        """Write totals to disk (must be called while holding self._lock)."""
        try:
            tmp = self._log_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._totals, f, indent=2)
            tmp.replace(self._log_path)  # atomic rename on most OSes
        except OSError:
            pass  # Non-fatal: next call will retry


# 🔥 GLOBAL INSTANCE
cost_tracker = CostTracker()
