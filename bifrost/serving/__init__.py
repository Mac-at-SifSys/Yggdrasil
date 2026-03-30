"""
Bifrost serving — HTTP server for Clifford HLM inference.

Provides three endpoint families:
  1. Basic /generate and /embed endpoints
  2. OpenAI-compatible /v1/chat/completions
  3. Native Clifford /v1/clifford/* for downstream KL-speaking agents
"""

from .server import CliffordServer
