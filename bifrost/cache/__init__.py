"""
Bifrost cache — Clifford-aware KV caching for inference.

Standard KV caches store all components.  Because Clifford attention
reads only certain grades of the key (typically grade 0 + grade 2),
we can cache *only* those grades for keys while keeping full grades
for values — saving ~37.5 % memory in the default configuration.
"""

from .clifford_kv_cache import CliffordKVCache, GradeStratifiedCache
