"""
Memory Planning -- Determine buffer allocation for execution.

This decides:
- Which tensors can share the same memory buffer (if lifetimes don't overlap)
- The total memory needed
- Buffer assignments for each node

Algorithm:
1. Compute liveness intervals: for each node, [first_use, last_use]
2. Sort by interval start
3. Greedy allocation: for each node, find the smallest free buffer that fits,
   or allocate a new one if none available
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from rune.compiler.ir import (
    IRGraph, IRNode, OpCode,
    GRADE_FULL,
)


@dataclass
class BufferInfo:
    """Info about a memory buffer."""
    buffer_id: int
    size_bytes: int
    assigned_nodes: List[int] = field(default_factory=list)


@dataclass
class MemoryPlan:
    """Complete memory allocation plan."""
    # Node ID -> buffer ID
    node_to_buffer: Dict[int, int] = field(default_factory=dict)
    # Buffer ID -> BufferInfo
    buffers: Dict[int, BufferInfo] = field(default_factory=dict)
    # Constant/parameter nodes that need persistent storage
    persistent_buffers: Dict[int, int] = field(default_factory=dict)  # node_id -> buffer_id

    @property
    def total_bytes(self) -> int:
        return sum(b.size_bytes for b in self.buffers.values())

    @property
    def n_buffers(self) -> int:
        return len(self.buffers)

    @property
    def peak_bytes(self) -> int:
        """Peak memory = sum of all buffer sizes (they can coexist)."""
        return self.total_bytes

    def __repr__(self):
        return (f"MemoryPlan({self.n_buffers} buffers, "
                f"total={self.total_bytes / 1024 / 1024:.1f} MB)")


class MemoryPlanPass:
    """
    Compute memory allocation plan with buffer reuse.
    """

    def __init__(self, verbose: bool = False, reuse_buffers: bool = False):
        self.verbose = verbose
        self.reuse_buffers = reuse_buffers
        self.stats = {
            'total_nodes': 0,
            'buffers_allocated': 0,
            'buffers_reused': 0,
            'naive_bytes': 0,
            'planned_bytes': 0,
        }

    def run(self, graph: IRGraph) -> MemoryPlan:
        """
        Compute memory plan for the graph.

        Returns a MemoryPlan mapping each node to a buffer.
        """
        self.stats = {k: 0 for k in self.stats}
        plan = MemoryPlan()

        topo = graph.topological_order()
        live_nodes = [graph.get_node(nid) for nid in topo]
        self.stats['total_nodes'] = len(live_nodes)

        # Step 1: Compute liveness intervals
        # For each node: (first_produced, last_consumed)
        # first_produced = topological position of the node
        # last_consumed = max topological position of any consumer
        topo_pos = {nid: i for i, nid in enumerate(topo)}

        intervals: Dict[int, Tuple[int, int]] = {}
        for node in live_nodes:
            nid = node.id
            produced = topo_pos[nid]
            users = graph.get_users(nid)
            if users:
                last_use = max(topo_pos.get(u, produced) for u in users)
            else:
                last_use = produced  # Output or dead-end
            intervals[nid] = (produced, last_use)

        # Step 2: Separate persistent (CONSTANT / INPUT) and transient buffers
        # Constants live forever. Input buffers are also dedicated because the
        # runtime uploads them before launch and expects them to stay valid for
        # the entire compiled step.
        next_buffer_id = 0
        naive_total = 0

        for node in live_nodes:
            if node.output_type is None:
                continue
            size = node.output_type.memory_bytes
            naive_total += size

            if node.op in (OpCode.CONSTANT, OpCode.INPUT):
                buf = BufferInfo(
                    buffer_id=next_buffer_id,
                    size_bytes=size,
                    assigned_nodes=[node.id],
                )
                plan.buffers[next_buffer_id] = buf
                plan.node_to_buffer[node.id] = next_buffer_id
                if node.op == OpCode.CONSTANT:
                    plan.persistent_buffers[node.id] = next_buffer_id
                next_buffer_id += 1

        self.stats['naive_bytes'] = naive_total

        # Step 3: Allocate transient buffers.
        # The current persistent-engine runtime is only correct with dedicated
        # transient buffers; aggressive reuse can alias values that later ops
        # still depend on. Keep the reuse path available behind an explicit
        # opt-in until a fully alias-safe planner/runtime contract exists.
        if not self.reuse_buffers:
            for nid in topo:
                node = graph.get_node(nid)
                if node.is_dead or node.output_type is None:
                    continue
                if node.op in (OpCode.CONSTANT, OpCode.INPUT):
                    continue
                size = node.output_type.memory_bytes
                if size == 0:
                    continue

                buf = BufferInfo(
                    buffer_id=next_buffer_id,
                    size_bytes=size,
                    assigned_nodes=[nid],
                )
                plan.buffers[next_buffer_id] = buf
                plan.node_to_buffer[nid] = next_buffer_id
                next_buffer_id += 1
                self.stats['buffers_allocated'] += 1

            self.stats['planned_bytes'] = plan.total_bytes

            if self.verbose:
                naive_mb = self.stats['naive_bytes'] / 1024 / 1024
                planned_mb = self.stats['planned_bytes'] / 1024 / 1024
                savings = 1.0 - (planned_mb / naive_mb) if naive_mb > 0 else 0
                print(
                    f"Memory plan: {plan.n_buffers} buffers, "
                    f"{planned_mb:.1f} MB (naive: {naive_mb:.1f} MB, "
                    f"savings: {savings:.0%}), reuse disabled"
                )

            return plan

        # Greedy reuse mode.
        # Track free buffers: list of (buffer_id, size, free_after_pos)
        free_pool: List[Tuple[int, int, int]] = []  # (buf_id, size, free_after)

        for nid in topo:
            node = graph.get_node(nid)
            if node.is_dead or node.output_type is None:
                continue
            if node.op in (OpCode.CONSTANT, OpCode.INPUT):
                continue  # Already handled

            size = node.output_type.memory_bytes
            if size == 0:
                continue

            produced, last_use = intervals[nid]

            # Release buffers whose last use is before our production time
            still_busy = []
            newly_free = []
            for buf_id, buf_size, free_after in free_pool:
                if free_after < produced:
                    newly_free.append((buf_id, buf_size))
                else:
                    still_busy.append((buf_id, buf_size, free_after))
            free_pool = still_busy

            # Try to find a free buffer that fits
            best_match = None
            best_idx = -1
            for idx, (buf_id, buf_size) in enumerate(newly_free):
                if buf_size >= size:
                    if best_match is None or buf_size < best_match[1]:
                        best_match = (buf_id, buf_size)
                        best_idx = idx

            if best_match is not None:
                # Reuse buffer
                buf_id = best_match[0]
                plan.node_to_buffer[nid] = buf_id
                plan.buffers[buf_id].assigned_nodes.append(nid)
                # Put back remaining free buffers
                for idx, (bid, bsz) in enumerate(newly_free):
                    if idx != best_idx:
                        free_pool.append((bid, bsz, -1))  # free immediately
                # This buffer is busy until last_use
                free_pool.append((buf_id, best_match[1], last_use))
                self.stats['buffers_reused'] += 1
            else:
                # Allocate new buffer
                buf = BufferInfo(
                    buffer_id=next_buffer_id,
                    size_bytes=size,
                    assigned_nodes=[nid],
                )
                plan.buffers[next_buffer_id] = buf
                plan.node_to_buffer[nid] = next_buffer_id
                free_pool.append((next_buffer_id, size, last_use))
                # Put back unused free buffers
                for bid, bsz in newly_free:
                    free_pool.append((bid, bsz, -1))
                next_buffer_id += 1
                self.stats['buffers_allocated'] += 1

        self.stats['planned_bytes'] = plan.total_bytes

        if self.verbose:
            naive_mb = self.stats['naive_bytes'] / 1024 / 1024
            planned_mb = self.stats['planned_bytes'] / 1024 / 1024
            savings = 1.0 - (planned_mb / naive_mb) if naive_mb > 0 else 0
            print(f"Memory plan: {plan.n_buffers} buffers, "
                  f"{planned_mb:.1f} MB (naive: {naive_mb:.1f} MB, "
                  f"savings: {savings:.0%}), "
                  f"reused {self.stats['buffers_reused']} buffers")

        return plan
