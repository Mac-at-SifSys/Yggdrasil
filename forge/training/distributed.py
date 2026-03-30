"""
distributed.py -- Stub for distributed training with grade-aware sharding.

Strategy: shard multivector parameters by grade across devices.
  - Device 0: grade 0 (scalar) + grade 1 (vector)     -- 4 components
  - Device 1: grade 2 (bivector) + grade 3 (trivector) -- 4 components

This gives a natural 50/50 split of the 8 components per multivector, and
the even/odd split aligns with the algebraic sub-algebra decomposition.

Current status: placeholder implementation.  Real distributed training
requires MPI or a process-group backend.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class DistributedConfig:
    """Configuration for grade-aware distributed training.

    Attributes
    ----------
    world_size : int
        Total number of devices/processes.
    rank : int
        This process's rank (0-indexed).
    grade_sharding : dict
        Maps device rank -> list of grades it owns.
        Default: {0: [0,1], 1: [2,3]} for 2-device setup.
    backend : str
        Communication backend.  'stub' = no-op (single process).
    sync_interval : int
        Synchronize grade boundaries every N steps.
    """
    world_size: int = 1
    rank: int = 0
    grade_sharding: Dict[int, List[int]] = field(default_factory=lambda: {0: [0, 1], 1: [2, 3]})
    backend: str = "stub"
    sync_interval: int = 1


def get_local_grades(config: DistributedConfig) -> List[int]:
    """Return the list of grades owned by this rank."""
    return config.grade_sharding.get(config.rank, [0, 1, 2, 3])


def shard_parameter(data: np.ndarray, config: DistributedConfig) -> np.ndarray:
    """Extract this rank's grade shard from a full multivector parameter.

    Parameters
    ----------
    data : np.ndarray, shape (..., 8)
        Full multivector parameter.
    config : DistributedConfig

    Returns
    -------
    np.ndarray
        Shard containing only this rank's grade components.
        Other grades are zeroed out.
    """
    from rune.types.multivector import GRADE_SLICES

    local_grades = get_local_grades(config)
    result = np.zeros_like(data)
    for grade in local_grades:
        slc = GRADE_SLICES[grade]
        result[..., slc] = data[..., slc]
    return result


def all_reduce_grades(
    local_data: np.ndarray,
    config: DistributedConfig,
) -> np.ndarray:
    """Placeholder all-reduce: in stub mode just returns the local data.

    In a real implementation this would:
    1. Each rank sends its grade shard to all others
    2. Each rank fills in the received grades
    3. Result: every rank has the full multivector
    """
    if config.backend == "stub" or config.world_size <= 1:
        return local_data

    # Future: MPI_Allreduce / NCCL all-gather by grade
    raise NotImplementedError(
        f"Distributed backend '{config.backend}' not yet implemented. "
        f"Use backend='stub' for single-process training."
    )


def synchronize_boundaries(
    params: list,
    config: DistributedConfig,
):
    """Synchronize grade boundary values across ranks.

    At grade boundaries (e.g., where grade-1 meets grade-2) the geometric
    product couples adjacent grades.  Periodic synchronization ensures
    consistency when grades are sharded across devices.

    In stub mode this is a no-op.
    """
    if config.backend == "stub" or config.world_size <= 1:
        return

    raise NotImplementedError("Boundary sync requires a real distributed backend.")


def print_sharding_plan(config: DistributedConfig):
    """Print the grade sharding plan for debugging."""
    print(f"Distributed config: world_size={config.world_size}, "
          f"backend={config.backend}")
    for rank, grades in sorted(config.grade_sharding.items()):
        grade_names = {0: "scalar", 1: "vector", 2: "bivector", 3: "trivector"}
        names = [grade_names.get(g, f"grade{g}") for g in grades]
        components = sum(
            {0: 1, 1: 3, 2: 3, 3: 1}.get(g, 0) for g in grades
        )
        print(f"  Rank {rank}: grades {grades} ({', '.join(names)}) "
              f"= {components} components per MV")
