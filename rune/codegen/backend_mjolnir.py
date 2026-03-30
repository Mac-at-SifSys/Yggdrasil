"""
backend_mjolnir.py — Code generation backend targeting mjolnir kernels

Translates optimized trace graphs into sequences of mjolnir kernel calls.
Phase 1a: generates Python/NumPy calls.
Phase 1b (future): generates C/CUDA kernel call sequences.
"""

from typing import Dict, Any
from rune.codegen.tracer import TraceGraph, TraceOp


class MjolnirBackend:
    """Backend that generates mjolnir kernel call sequences."""

    def __init__(self):
        self._kernel_map = {
            'geometric_product': 'cl3_geometric_product',
            'outer_product': 'cl3_outer_product',
            'inner_product': 'cl3_inner_product',
            'grade_project': 'cl3_grade_project',
            'reverse': 'cl3_reverse',
            'sandwich': 'cl3_sandwich',
            'bivector_exp': 'cl3_bivector_exp',
            'normalize': 'cl3_normalize',
            'add': 'cl3_add',
            'scale': 'cl3_scale',
            'grade_projected_product': 'cl3_fused_grade_product',
        }

    def compile(self, graph: TraceGraph) -> str:
        """
        Generate C code that calls mjolnir kernels.
        Returns a string of C code.
        """
        lines = [
            '#include "cl3_types.h"',
            '#include "cl3_ops.h"',
            '',
        ]

        for op in graph.ops:
            kernel = self._kernel_map.get(op.op_name, f'/* unknown: {op.op_name} */')
            args = ', '.join([f'&{inp}' for inp in op.inputs])
            if op.kwargs:
                extra_args = ', '.join([f'{v}' for v in op.kwargs.values()])
                lines.append(f'Cl3Multivector {op.output} = {kernel}({args}, {extra_args});')
            else:
                lines.append(f'Cl3Multivector {op.output} = {kernel}({args});')

        return '\n'.join(lines)

    def emit_python(self, graph: TraceGraph) -> str:
        """
        Generate Python/NumPy code for the optimized graph.
        Used in Phase 1a for immediate execution.
        """
        lines = ['import numpy as np', 'from rune.ops import *', '']

        for op in graph.ops:
            if op.op_name == 'geometric_product':
                lines.append(f'{op.output} = geom_prod({", ".join(op.inputs)})')
            elif op.op_name == 'grade_project':
                grade = op.kwargs.get('grade', 0)
                lines.append(f'{op.output} = grade_project({op.inputs[0]}, {grade})')
            elif op.op_name == 'sandwich':
                lines.append(f'{op.output} = sandwich({", ".join(op.inputs)})')
            elif op.op_name == 'grade_projected_product':
                grade = op.kwargs.get('grade', 0)
                lines.append(
                    f'{op.output} = grade_project(geom_prod({", ".join(op.inputs)}), {grade})'
                )
            else:
                lines.append(f'{op.output} = {op.op_name}({", ".join(op.inputs)})')

        return '\n'.join(lines)
