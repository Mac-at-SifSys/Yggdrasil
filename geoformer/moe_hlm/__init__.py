"""MoE-HLM: Mixture of Experts with Holographic Language Model experts.

Standard flat tensor router selects top-k experts per token.
Each expert is a full HLM-8^3 — three rounds of Clifford algebra
geometric products across 8 blades, with shared ToU bank lookup.
"""
