"""
Adapters Module - External service integrations

This module provides adapters for external services like WorldQuant BRAIN.
All adapters implement protocols defined in backend.protocols for testability.

Usage:
    from backend.adapters import get_brain_adapter
    
    # In async context:
    brain = await get_brain_adapter()
    result = await brain.simulate_alpha("rank(close)")
"""

from backend.adapters.brain_adapter import (
    BrainAdapter,
    get_brain_adapter,
    get_brain_adapter_sync,
    brain_adapter,
    reset_brain_adapter,
)
from backend.adapters.mcp_brain_adapter import MCPBrainAdapter

__all__ = [
    "BrainAdapter",
    "MCPBrainAdapter",
    "get_brain_adapter",
    "get_brain_adapter_sync",
    "brain_adapter",
    "reset_brain_adapter",
]
