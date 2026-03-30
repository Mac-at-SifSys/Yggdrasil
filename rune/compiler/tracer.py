"""
Rune Tracer -- Records HLM forward pass to build an IR graph.

Usage:
    tracer = Tracer()
    ir = tracer.trace(model, batch_size=16, seq_len=2048)
    # ir is an IRGraph ready for optimization

The tracer intercepts holograph layer structure. For each layer:
- Records the operation type, input shapes, output shapes
- Infers grade masks from the operation
- Builds IRNodes

The tracer operates on SHAPES ONLY, not actual data. It produces a static
computation graph that can be compiled once and reused for any input with
the same batch/seq dimensions.
"""

from typing import Dict, List, Optional, Tuple
from rune.compiler.ir import (
    IRGraph, IRType, OpCode,
    GRADE_SCALAR, GRADE_VECTOR, GRADE_BIVECTOR, GRADE_TRIVECTOR,
    GRADE_EVEN, GRADE_ODD, GRADE_FULL,
)


class Tracer:
    """
    Traces an HLM model to produce an IRGraph.

    Walks the model's layer structure and records operations with their
    shapes and grade annotations. Does NOT execute any computation --
    purely a shape/structure analysis.
    """

    def __init__(self):
        self._graph: Optional[IRGraph] = None
        self._param_map: Dict[int, int] = {}  # id(param) -> node_id
        self._include_memory: bool = False
        self._memory_state_id: Optional[int] = None
        self._memory_write_source_id: Optional[int] = None

    def trace(self, model, batch_size: int = 1, seq_len: int = 64,
              include_memory: bool = False) -> IRGraph:
        """
        Trace a full HLM forward pass and return the IR graph.

        Args:
            model: An HLM model instance.
            batch_size: Batch size for the traced graph.
            seq_len: Sequence length for the traced graph.

        Returns:
            IRGraph with all operations recorded.
        """
        self._graph = IRGraph()
        self._graph.batch_size = batch_size
        self._graph.seq_len = seq_len
        self._param_map = {}
        self._include_memory = include_memory
        self._memory_state_id = None
        self._memory_write_source_id = None

        config = model.config

        self._graph.config_attrs = {
            'vocab_size': config.vocab_size,
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'd_ff': config.d_ff,
            'd_head': config.d_head,
        }

        B, S = batch_size, seq_len
        D = config.d_model
        H = config.n_heads
        D_H = config.d_head
        D_FF = config.d_ff
        V = config.vocab_size

        # --- Input ---
        input_id = self._graph.add_node(
            op=OpCode.INPUT,
            output_type=IRType(shape=(B, S), grade_mask=0, dtype='int64'),
            name='token_ids',
        )

        # --- Embedding ---
        embed_out = self._trace_embedding(input_id, B, S, D, V, model.embedding)

        # --- Positional encoding ---
        if getattr(model, 'pos_encoding', None) is not None:
            pos_out = self._trace_pos_encoding(embed_out, B, S, D, model.pos_encoding)
        else:
            pos_out = embed_out

        # --- Transformer blocks ---
        x = pos_out
        for i, block in enumerate(model.blocks):
            x = self._trace_block(x, B, S, D, H, D_H, D_FF, block, layer_idx=i)
            if (
                self._include_memory
                and getattr(model, 'memory_layers', None)
                and i in model.memory_layers
            ):
                x = self._trace_memory_layer(
                    x, B, S, D, model.memory_layers[i], model, layer_idx=i
                )
            if (
                self._include_memory
                and getattr(model, 'memory_write_layer', None) is not None
                and i == model.memory_write_layer
            ):
                self._memory_write_source_id = x
            # ToU layers (skip for compiler -- they are optional geometry ops)
            if i in model.tou_layers:
                x = self._trace_tou_layer(x, B, S, D, model.tou_layers[i], layer_idx=i)

        # --- Final norm ---
        x = self._trace_layer_norm(x, B, S, D, model.final_norm, name='final_norm')

        # --- LM head / tied output projection ---
        if hasattr(model, 'lm_head'):
            head_out = self._trace_linear(x, B, S, D, V, model.lm_head, name='lm_head')
            logits_id = self._graph.add_node(
                op=OpCode.GRADE_PROJECT,
                inputs=[head_out],
                output_type=IRType(shape=(B, S, V), grade_mask=GRADE_SCALAR),
                attrs={'target_grade': 0},
                name='logits_scalar_proj',
            )
        else:
            logits_id = self._trace_tied_output_head(x, B, S, D, V, model.embedding)

        # --- Output ---
        self._graph.add_node(
            op=OpCode.OUTPUT,
            inputs=[logits_id],
            output_type=IRType(shape=(B, S, V), grade_mask=GRADE_SCALAR),
            name='output',
        )

        return self._graph

    def _add_constant(
        self,
        param,
        shape: Tuple,
        grade_mask: int = GRADE_FULL,
        name: str = '',
        *,
        dtype: str = 'float32',
        storage_components: Optional[int] = None,
    ) -> int:
        """Register a learnable parameter as a CONSTANT node."""
        pid = id(param)
        if pid in self._param_map:
            return self._param_map[pid]
        nid = self._graph.add_node(
            op=OpCode.CONSTANT,
            output_type=IRType(
                shape=shape,
                grade_mask=grade_mask,
                dtype=dtype,
                storage_components=storage_components,
            ),
            attrs={'param_id': pid},
            name=name,
        )
        self._param_map[pid] = nid
        return nid

    def _add_static_constant(
        self,
        data,
        shape: Tuple,
        grade_mask: int = GRADE_FULL,
        name: str = '',
        *,
        dtype: str = 'float32',
        storage_components: Optional[int] = None,
    ) -> int:
        """Register a compile-time constant payload."""
        import numpy as _np

        arr = _np.ascontiguousarray(data, dtype=_np.dtype(dtype))
        return self._graph.add_node(
            op=OpCode.CONSTANT,
            output_type=IRType(
                shape=shape,
                grade_mask=grade_mask,
                dtype=dtype,
                storage_components=storage_components,
            ),
            attrs={'const_data': arr},
            name=name,
        )

    def _trace_embedding(self, input_id: int, B: int, S: int, D: int, V: int,
                         embedding) -> int:
        """Trace CliffordEmbedding: lookup table."""
        weight_id = self._add_constant(
            embedding.weight, shape=(V, D), grade_mask=GRADE_FULL,
            name='embed.weight',
        )
        return self._graph.add_node(
            op=OpCode.EMBED_LOOKUP,
            inputs=[input_id, weight_id],
            output_type=IRType(shape=(B, S, D), grade_mask=GRADE_FULL),
            name='embedding',
        )

    def _trace_pos_encoding(self, x_id: int, B: int, S: int, D: int,
                            pos_enc) -> int:
        """Trace RotorPositionalEncoding as positions @ bivector_freqs -> bvexp -> GP."""
        import numpy as _np

        positions = _np.broadcast_to(
            _np.arange(S, dtype=_np.float32).reshape(1, S, 1),
            (B, S, 1),
        ).copy()
        pos_id = self._add_static_constant(
            positions,
            shape=(B, S, 1),
            grade_mask=GRADE_SCALAR,
            name='pos_enc.positions',
            dtype='float32',
            storage_components=1,
        )
        freq_id = self._add_constant(
            pos_enc.bivector_freqs,
            shape=(D, 3),
            grade_mask=GRADE_BIVECTOR,
            name='pos_enc.bivector_freqs',
            dtype='float32',
            storage_components=1,
        )
        scaled_bv_id = self._graph.add_node(
            op=OpCode.MATMUL_SCALAR,
            inputs=[pos_id, freq_id],
            output_type=IRType(
                shape=(B, S, D),
                grade_mask=GRADE_BIVECTOR,
                dtype='float32',
                storage_components=3,
            ),
            attrs={'m': B * S, 'k': 1, 'n': D * 3},
            name='pos_enc.scaled_bivectors',
        )
        bvexp_id = self._graph.add_node(
            op=OpCode.BIVECTOR_EXP,
            inputs=[scaled_bv_id],
            output_type=IRType(shape=(B, S, D), grade_mask=GRADE_EVEN),
            attrs={
                'max_seq_len': pos_enc.max_seq_len,
                'input_components': 3,
            },
            name='pos_enc.rotor_table',
        )
        return self._graph.add_node(
            op=OpCode.GEOMETRIC_PRODUCT,
            inputs=[bvexp_id, x_id],
            output_type=IRType(shape=(B, S, D), grade_mask=GRADE_FULL),
            name='pos_encoding',
        )

    def _trace_tied_output_head(self, x_id: int, B: int, S: int, D: int, V: int,
                                embedding) -> int:
        """
        Trace HLM125M-style tied output projection:
            logits = grade_0(x_final) @ grade_0(embed.weight)^T
        """
        x_scalar_id = self._graph.add_node(
            op=OpCode.GRADE_PROJECT,
            inputs=[x_id],
            output_type=IRType(shape=(B, S, D), grade_mask=GRADE_SCALAR),
            attrs={'target_grade': 0},
            name='final_hidden_scalar_proj',
        )
        embed_weight_id = self._add_constant(
            embedding.weight, shape=(V, D), grade_mask=GRADE_FULL,
            name='embed.weight',
        )
        embed_scalar_id = self._graph.add_node(
            op=OpCode.GRADE_PROJECT,
            inputs=[embed_weight_id],
            output_type=IRType(shape=(V, D), grade_mask=GRADE_SCALAR),
            attrs={'target_grade': 0},
            name='embed_scalar_proj',
        )
        return self._graph.add_node(
            op=OpCode.MATMUL_SCALAR,
            inputs=[x_scalar_id, embed_scalar_id],
            output_type=IRType(shape=(B, S, V), grade_mask=GRADE_SCALAR),
            attrs={'m': B * S, 'k': D, 'n': V},
            name='tied_output_head',
        )

    def _trace_block(self, x_id: int, B: int, S: int, D: int,
                     H: int, D_H: int, D_FF: int,
                     block, layer_idx: int) -> int:
        """Trace a single HLMBlock: norm->attn->residual->norm->ffn->residual."""
        prefix = f'block{layer_idx}'

        # --- Attention sub-layer ---
        # LayerNorm
        attn_norm_out = self._trace_layer_norm(
            x_id, B, S, D, block.attn_norm, name=f'{prefix}.attn_norm')

        # Attention
        attn_out = self._trace_attention(
            attn_norm_out, B, S, D, H, D_H, block.attn, name=f'{prefix}.attn')

        # Residual add
        residual1 = self._graph.add_node(
            op=OpCode.ADD,
            inputs=[x_id, attn_out],
            output_type=IRType(shape=(B, S, D), grade_mask=GRADE_FULL),
            name=f'{prefix}.attn_residual',
        )

        # --- FFN sub-layer ---
        ffn_norm_out = self._trace_layer_norm(
            residual1, B, S, D, block.ffn_norm, name=f'{prefix}.ffn_norm')

        if block.use_moe:
            # MoE: trace as single compound FFN op (not individually)
            ffn_out = self._graph.add_node(
                op=OpCode.FFN,
                inputs=[ffn_norm_out],
                output_type=IRType(shape=(B, S, D), grade_mask=GRADE_FULL),
                attrs={'d_ff': D_FF, 'moe': True,
                       'n_experts': len(block.experts) if block.experts else 4},
                name=f'{prefix}.moe_ffn',
            )
        else:
            ffn_out = self._trace_ffn(
                ffn_norm_out, B, S, D, D_FF, block.ffn, name=f'{prefix}.ffn')

        # Residual add
        residual2 = self._graph.add_node(
            op=OpCode.ADD,
            inputs=[residual1, ffn_out],
            output_type=IRType(shape=(B, S, D), grade_mask=GRADE_FULL),
            name=f'{prefix}.ffn_residual',
        )

        return residual2

    def _trace_attention(self, x_id: int, B: int, S: int, D: int,
                         H: int, D_H: int, attn, name: str = '') -> int:
        """
        Trace CliffordAttention.

        Per head:
          Q = proj_q(x)                      (B, S, D_H, 8)
          K = proj_k(x)                      (B, S, D_H, 8)
          V = proj_v(x)                      (B, S, D_H, 8)
          scores = matmul(Q_flat, K_flat^T)  (B, S, S) -- scalar product
          attn = softmax(scores)             (B, S, S) -- scalar
          out_h = einsum(attn, V)            (B, S, D_H, 8)
        concat heads -> proj_out -> (B, S, D, 8)
        """
        head_out_ids = []

        for h in range(H):
            # Q projection
            q_id = self._trace_linear(
                x_id, B, S, D, D_H, attn.proj_q[h],
                name=f'{name}.head{h}.proj_q', bias=False)

            # K projection
            k_id = self._trace_linear(
                x_id, B, S, D, D_H, attn.proj_k[h],
                name=f'{name}.head{h}.proj_k', bias=False)

            # V projection
            v_id = self._trace_linear(
                x_id, B, S, D, D_H, attn.proj_v[h],
                name=f'{name}.head{h}.proj_v', bias=False)

            # Attention score: grade_0(Q * ~K) via flatten + matmul
            # The reverse is implicit in the flat dot product
            # (scalar_part(Q * ~K) == Euclidean dot in Cl(3,0))
            score_id = self._graph.add_node(
                op=OpCode.ATTN_SCORE,
                inputs=[q_id, k_id],
                output_type=IRType(shape=(B, S, S), grade_mask=GRADE_SCALAR),
                attrs={'d_head': D_H, 'scale': 1.0 / (D_H ** 0.5)},
                name=f'{name}.head{h}.score',
            )

            # Softmax
            softmax_id = self._graph.add_node(
                op=OpCode.SOFTMAX,
                inputs=[score_id],
                output_type=IRType(shape=(B, S, S), grade_mask=GRADE_SCALAR),
                name=f'{name}.head{h}.softmax',
            )

            # Weighted sum of V
            weighted_id = self._graph.add_node(
                op=OpCode.WEIGHTED_SUM,
                inputs=[softmax_id, v_id],
                output_type=IRType(shape=(B, S, D_H), grade_mask=GRADE_FULL),
                attrs={'d_head': D_H},
                name=f'{name}.head{h}.weighted_sum',
            )

            head_out_ids.append(weighted_id)

        # Conceptually concatenate heads -- this is a COPY/reshape, no compute
        # We model it as a single node that takes all head outputs
        concat_id = self._graph.add_node(
            op=OpCode.COPY,
            inputs=head_out_ids,
            output_type=IRType(shape=(B, S, D), grade_mask=GRADE_FULL),
            attrs={'concat_axis': 2, 'n_heads': H},
            name=f'{name}.concat_heads',
        )

        # Output projection
        out_id = self._trace_linear(
            concat_id, B, S, D, D, attn.proj_out,
            name=f'{name}.proj_out', bias=True)

        return out_id

    def _trace_linear(self, x_id: int, B: int, S: int,
                      D_in: int, D_out: int, linear,
                      name: str = '', bias: bool = True) -> int:
        """Trace CliffordLinear: y = sum_i GP(W[j,i], x[i]) + b[j]."""
        weight_id = self._add_constant(
            linear.weight, shape=(D_out, D_in),
            grade_mask=GRADE_FULL, name=f'{name}.weight')

        inputs = [weight_id, x_id]
        attrs = {'d_in': D_in, 'd_out': D_out}

        if bias and linear.bias is not None:
            bias_id = self._add_constant(
                linear.bias, shape=(D_out,),
                grade_mask=GRADE_FULL, name=f'{name}.bias')
            inputs.append(bias_id)
            attrs['has_bias'] = True
        else:
            attrs['has_bias'] = False

        return self._graph.add_node(
            op=OpCode.LINEAR,
            inputs=inputs,
            output_type=IRType(shape=(B, S, D_out), grade_mask=GRADE_FULL),
            attrs=attrs,
            name=name,
        )

    def _trace_ffn(self, x_id: int, B: int, S: int, D: int, D_FF: int,
                   ffn, name: str = '') -> int:
        """Trace CliffordFFN: linear_up -> gelu -> linear_down."""
        # Up projection
        up_id = self._trace_linear(
            x_id, B, S, D, D_FF, ffn.up, name=f'{name}.up', bias=True)

        # GELU activation
        gelu_id = self._graph.add_node(
            op=OpCode.GELU,
            inputs=[up_id],
            output_type=IRType(shape=(B, S, D_FF), grade_mask=GRADE_FULL),
            name=f'{name}.gelu',
        )

        # Down projection
        down_id = self._trace_linear(
            gelu_id, B, S, D_FF, D, ffn.down, name=f'{name}.down', bias=True)

        return down_id

    def _trace_layer_norm(self, x_id: int, B: int, S: int, D: int,
                          norm, name: str = '') -> int:
        """Trace CliffordLayerNorm: norm -> scale(GP) -> shift."""
        gamma_id = self._add_constant(
            norm.gamma, shape=(D,), grade_mask=GRADE_FULL,
            name=f'{name}.gamma')
        beta_id = self._add_constant(
            norm.beta, shape=(D,), grade_mask=GRADE_FULL,
            name=f'{name}.beta')

        return self._graph.add_node(
            op=OpCode.LAYER_NORM,
            inputs=[x_id, gamma_id, beta_id],
            output_type=IRType(shape=(B, S, D), grade_mask=GRADE_FULL),
            attrs={'d_model': D, 'eps': norm.eps},
            name=name,
        )

    def _trace_tou_layer(self, x_id: int, B: int, S: int, D: int,
                         tou_layer, layer_idx: int) -> int:
        """
        Trace ToU layer as a compound sandwich-product operation.
        ToU applies geometric transformations via blade primitives.
        We model it as a single opaque node for now.
        """
        return self._graph.add_node(
            op=OpCode.SANDWICH,
            inputs=[x_id],
            output_type=IRType(shape=(B, S, D), grade_mask=GRADE_FULL),
            attrs={'tou_layer_idx': layer_idx},
            name=f'tou_layer{layer_idx}',
        )

    def trace_with_loss(self, model, batch_size: int = 1, seq_len: int = 64,
                        include_memory: bool = False) -> IRGraph:
        """
        Trace model forward pass plus cross-entropy loss.
        Returns graph with a loss OUTPUT node.
        """
        graph = self.trace(model, batch_size, seq_len, include_memory=include_memory)

        # Find the output node
        output_nodes = graph.find_nodes_by_op(OpCode.OUTPUT)
        assert len(output_nodes) == 1, f"Expected 1 output, got {len(output_nodes)}"
        output_node = output_nodes[0]
        logits_id = output_node.inputs[0]

        # Remove the old output
        output_node.is_dead = True

        B, S = batch_size, seq_len
        V = model.config.vocab_size

        # Target input
        target_id = graph.add_node(
            op=OpCode.INPUT,
            output_type=IRType(shape=(B, S), grade_mask=0, dtype='int64'),
            name='target_ids',
        )

        # Cross entropy loss
        loss_id = graph.add_node(
            op=OpCode.CROSS_ENTROPY,
            inputs=[logits_id, target_id],
            output_type=IRType(shape=(1,), grade_mask=GRADE_SCALAR),
            name='loss',
        )

        # New output
        graph.add_node(
            op=OpCode.OUTPUT,
            inputs=[loss_id],
            output_type=IRType(shape=(1,), grade_mask=GRADE_SCALAR),
            name='output_loss',
        )

        return graph

    def trace_training_step(self, model, batch_size: int = 1, seq_len: int = 64,
                            optimizer: str = 'adam',
                            include_memory: bool = False) -> IRGraph:
        """
        Trace a full training step: forward + loss + backward + optimizer update.

        Produces a complete IR graph that includes:
        - Forward graph (existing trace_with_loss)
        - Backward graph (mirror of forward, reversed, with gradient ops)
        - Optimizer update graph (Adam state + update for each parameter)

        Args:
            model: An HLM model instance.
            batch_size: Batch size for the traced graph.
            seq_len: Sequence length for the traced graph.
            optimizer: Optimizer type ('adam' or 'sgd').
            include_memory: If True, include memory bank nodes.

        Returns:
            IRGraph with forward + backward + update.
        """
        # Start with forward + loss graph
        graph = self.trace_with_loss(model, batch_size, seq_len, include_memory=include_memory)

        config = model.config
        B, S = batch_size, seq_len
        D = config.d_model
        V = config.vocab_size

        # --- Backward graph ---
        # Find the loss node
        loss_nodes = graph.find_nodes_by_op(OpCode.CROSS_ENTROPY)
        assert len(loss_nodes) == 1
        loss_node = loss_nodes[0]
        logits_id = loss_node.inputs[0]
        target_id = loss_node.inputs[1]

        # Backward through CE loss: produce grad_logits
        grad_logits_id = graph.add_node(
            op=OpCode.BACKWARD_CE,
            inputs=[logits_id, target_id, loss_node.id],
            output_type=IRType(shape=(B, S, V), grade_mask=GRADE_SCALAR),
            name='backward_ce',
        )

        # Walk the forward graph in reverse to trace backward ops
        # Collect the forward nodes in reverse topological order
        topo = graph.topological_order()
        forward_nodes = []
        for nid in topo:
            node = graph.get_node(nid)
            if node.is_dead or node.fused_into >= 0:
                continue
            # Skip loss/output nodes (already handled)
            if node.op in (OpCode.CROSS_ENTROPY, OpCode.OUTPUT, OpCode.INPUT):
                continue
            forward_nodes.append(node)

        # Build maps from forward values / parameters to their gradient nodes.
        grad_map = {logits_id: grad_logits_id}
        param_grad_map = {}

        # Trace backward through remaining forward ops in reverse
        for node in reversed(forward_nodes):
            if node.op == OpCode.CONSTANT:
                continue  # parameters are handled by update nodes later

            grad_input_id = self._find_grad_for_node(node, grad_map, graph)
            if grad_input_id is None:
                continue

            if node.op == OpCode.GRADE_PROJECT:
                bwd_id = graph.add_node(
                    op=OpCode.BACKWARD_GRADE_PROJECT,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(node.inputs[0]).output_type,
                    attrs=node.attrs,
                    name=f'backward_{node.name}',
                )
                if node.inputs:
                    self._merge_input_grad(graph, grad_map, param_grad_map, node.inputs[0], bwd_id)

            elif node.op == OpCode.LAYER_NORM:
                x_id, gamma_id, beta_id = node.inputs
                grad_x_id = graph.add_node(
                    op=OpCode.BACKWARD_NORM,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(x_id).output_type,
                    attrs={**node.attrs, 'mode': 'input'},
                    name=f'backward_{node.name}_input',
                )
                grad_gamma_id = graph.add_node(
                    op=OpCode.BACKWARD_NORM,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(gamma_id).output_type,
                    attrs={**node.attrs, 'mode': 'gamma'},
                    name=f'backward_{node.name}_gamma',
                )
                grad_beta_id = graph.add_node(
                    op=OpCode.BACKWARD_NORM,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(beta_id).output_type,
                    attrs={**node.attrs, 'mode': 'beta'},
                    name=f'backward_{node.name}_beta',
                )
                self._merge_grad(graph, grad_map, x_id, grad_x_id)
                self._merge_grad(graph, param_grad_map, gamma_id, grad_gamma_id)
                self._merge_grad(graph, param_grad_map, beta_id, grad_beta_id)

            elif node.op == OpCode.ADD:
                bwd_id = graph.add_node(
                    op=OpCode.BACKWARD_ADD,
                    inputs=[grad_input_id],
                    output_type=graph.get_node(node.inputs[0]).output_type,
                    attrs={'n_inputs': len(node.inputs)},
                    name=f'backward_{node.name}',
                )
                # Gradient flows to both inputs for ADD
                for inp_id in node.inputs:
                    self._merge_grad(graph, grad_map, inp_id, bwd_id)

            elif node.op == OpCode.LINEAR:
                weight_id, x_id = node.inputs[:2]
                grad_x_id = graph.add_node(
                    op=OpCode.BACKWARD_LINEAR,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(x_id).output_type,
                    attrs={**node.attrs, 'mode': 'input'},
                    name=f'backward_{node.name}_input',
                )
                grad_w_id = graph.add_node(
                    op=OpCode.BACKWARD_LINEAR,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(weight_id).output_type,
                    attrs={**node.attrs, 'mode': 'weight'},
                    name=f'backward_{node.name}_weight',
                )
                self._merge_input_grad(graph, grad_map, param_grad_map, x_id, grad_x_id)
                self._merge_grad(graph, param_grad_map, weight_id, grad_w_id)
                if len(node.inputs) > 2:
                    bias_id = node.inputs[2]
                    grad_b_id = graph.add_node(
                        op=OpCode.BACKWARD_LINEAR,
                        inputs=[grad_input_id] + node.inputs,
                        output_type=graph.get_node(bias_id).output_type,
                        attrs={**node.attrs, 'mode': 'bias'},
                        name=f'backward_{node.name}_bias',
                    )
                    self._merge_grad(graph, param_grad_map, bias_id, grad_b_id)

            elif node.op == OpCode.GELU:
                bwd_id = graph.add_node(
                    op=OpCode.BACKWARD_GELU,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(node.inputs[0]).output_type,
                    name=f'backward_{node.name}',
                )
                if node.inputs:
                    self._merge_input_grad(graph, grad_map, param_grad_map, node.inputs[0], bwd_id)

            elif node.op == OpCode.ATTENTION:
                bwd_id = graph.add_node(
                    op=OpCode.BACKWARD_ATTENTION,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=node.output_type,
                    attrs=node.attrs,
                    name=f'backward_{node.name}',
                )
                if node.inputs:
                    self._merge_grad(graph, grad_map, node.inputs[0], bwd_id)

            elif node.op == OpCode.FFN:
                bwd_id = graph.add_node(
                    op=OpCode.BACKWARD_FFN,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=node.output_type,
                    attrs=node.attrs,
                    name=f'backward_{node.name}',
                )
                if node.inputs:
                    self._merge_grad(graph, grad_map, node.inputs[0], bwd_id)

            elif node.op == OpCode.SOFTMAX:
                bwd_id = graph.add_node(
                    op=OpCode.BACKWARD_SOFTMAX,
                    inputs=[grad_input_id, node.id],
                    output_type=graph.get_node(node.inputs[0]).output_type,
                    name=f'backward_{node.name}',
                )
                if node.inputs:
                    self._merge_input_grad(graph, grad_map, param_grad_map, node.inputs[0], bwd_id)

            elif node.op == OpCode.WEIGHTED_SUM:
                weights_id, value_id = node.inputs
                grad_attn_id = graph.add_node(
                    op=OpCode.BACKWARD_WEIGHTED_SUM,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(weights_id).output_type,
                    attrs={**node.attrs, 'mode': 'weights'},
                    name=f'backward_{node.name}_weights',
                )
                grad_value_id = graph.add_node(
                    op=OpCode.BACKWARD_WEIGHTED_SUM,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(value_id).output_type,
                    attrs={**node.attrs, 'mode': 'values'},
                    name=f'backward_{node.name}_values',
                )
                self._merge_input_grad(graph, grad_map, param_grad_map, weights_id, grad_attn_id)
                self._merge_input_grad(graph, grad_map, param_grad_map, value_id, grad_value_id)

            elif node.op == OpCode.MEMORY_GATE:
                x_id, context_id, gate_id = node.inputs
                gate_attrs = {
                    'batch': B,
                    'seq': S,
                    'd_model': D,
                }
                grad_x_id = graph.add_node(
                    op=OpCode.BACKWARD_MEMORY_GATE,
                    inputs=[grad_input_id, context_id, gate_id],
                    output_type=graph.get_node(x_id).output_type,
                    attrs={**gate_attrs, 'mode': 'input'},
                    name=f'backward_{node.name}_input',
                )
                grad_context_id = graph.add_node(
                    op=OpCode.BACKWARD_MEMORY_GATE,
                    inputs=[grad_input_id, context_id, gate_id],
                    output_type=graph.get_node(context_id).output_type,
                    attrs={**gate_attrs, 'mode': 'context'},
                    name=f'backward_{node.name}_context',
                )
                grad_gate_id = graph.add_node(
                    op=OpCode.BACKWARD_MEMORY_GATE,
                    inputs=[grad_input_id, context_id, gate_id],
                    output_type=graph.get_node(gate_id).output_type,
                    attrs={**gate_attrs, 'mode': 'gate'},
                    name=f'backward_{node.name}_gate',
                )
                self._merge_input_grad(graph, grad_map, param_grad_map, x_id, grad_x_id)
                self._merge_input_grad(graph, grad_map, param_grad_map, context_id, grad_context_id)
                self._merge_input_grad(graph, grad_map, param_grad_map, gate_id, grad_gate_id)

            elif node.op in (OpCode.MEMORY_READ, OpCode.MEMORY_WRITE, OpCode.MEAN_POOL_SEQ):
                continue

            elif node.op == OpCode.ATTN_SCORE:
                q_id, k_id = node.inputs
                grad_q_id = graph.add_node(
                    op=OpCode.BACKWARD_GP,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(q_id).output_type,
                    attrs={**node.attrs, 'mode': 'left'},
                    name=f'backward_{node.name}_q',
                )
                grad_k_id = graph.add_node(
                    op=OpCode.BACKWARD_GP,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(k_id).output_type,
                    attrs={**node.attrs, 'mode': 'right'},
                    name=f'backward_{node.name}_k',
                )
                self._merge_input_grad(graph, grad_map, param_grad_map, q_id, grad_q_id)
                self._merge_input_grad(graph, grad_map, param_grad_map, k_id, grad_k_id)

            elif node.op == OpCode.COPY:
                for input_index, inp_id in enumerate(node.inputs):
                    bwd_id = graph.add_node(
                        op=OpCode.BACKWARD_COPY,
                        inputs=[grad_input_id],
                        output_type=graph.get_node(inp_id).output_type,
                        attrs={**node.attrs, 'input_index': input_index},
                        name=f'backward_{node.name}_{input_index}',
                    )
                    self._merge_input_grad(graph, grad_map, param_grad_map, inp_id, bwd_id)

            elif node.op == OpCode.EMBED_LOOKUP:
                bwd_id = graph.add_node(
                    op=OpCode.BACKWARD_EMBED,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(node.inputs[1]).output_type,
                    name=f'backward_{node.name}',
                )
                self._merge_grad(graph, param_grad_map, node.inputs[1], bwd_id)

            elif node.op == OpCode.GEOMETRIC_PRODUCT:
                left_id, right_id = node.inputs
                grad_left_id = graph.add_node(
                    op=OpCode.BACKWARD_GP,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(left_id).output_type,
                    attrs={'mode': 'left'},
                    name=f'backward_{node.name}_left',
                )
                grad_right_id = graph.add_node(
                    op=OpCode.BACKWARD_GP,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(right_id).output_type,
                    attrs={'mode': 'right'},
                    name=f'backward_{node.name}_right',
                )
                self._merge_input_grad(graph, grad_map, param_grad_map, left_id, grad_left_id)
                self._merge_input_grad(graph, grad_map, param_grad_map, right_id, grad_right_id)

            elif node.op == OpCode.MATMUL_SCALAR:
                left_id, right_id = node.inputs
                grad_left_id = graph.add_node(
                    op=OpCode.BACKWARD_MATMUL_SCALAR,
                    inputs=[grad_input_id, left_id, right_id],
                    output_type=graph.get_node(left_id).output_type,
                    attrs={'mode': 'input'},
                    name=f'backward_{node.name}_input',
                )
                grad_right_id = graph.add_node(
                    op=OpCode.BACKWARD_MATMUL_SCALAR,
                    inputs=[grad_input_id, left_id, right_id],
                    output_type=graph.get_node(right_id).output_type,
                    attrs={'mode': 'weight'},
                    name=f'backward_{node.name}_weight',
                )
                self._merge_input_grad(graph, grad_map, param_grad_map, left_id, grad_left_id)
                self._merge_input_grad(graph, grad_map, param_grad_map, right_id, grad_right_id)

            elif node.op == OpCode.BIVECTOR_EXP:
                bwd_id = graph.add_node(
                    op=OpCode.BACKWARD_BVEXP,
                    inputs=[grad_input_id] + node.inputs,
                    output_type=graph.get_node(node.inputs[0]).output_type,
                    attrs=dict(node.attrs),
                    name=f'backward_{node.name}',
                )
                if node.inputs:
                    self._merge_input_grad(graph, grad_map, param_grad_map, node.inputs[0], bwd_id)

        # --- Optimizer update graph ---
        # For each CONSTANT (parameter) node, emit Adam update
        constant_nodes = graph.find_nodes_by_op(OpCode.CONSTANT)
        for const_node in constant_nodes:
            if 'const_data' in const_node.attrs or 'param_id' not in const_node.attrs:
                continue
            if optimizer == 'adam':
                grad_node_id = param_grad_map.get(const_node.id)
                if grad_node_id is None:
                    continue
                graph.add_node(
                    op=OpCode.ADAM_FULL_UPDATE,
                    inputs=[const_node.id, grad_node_id],
                    output_type=const_node.output_type,
                    attrs={
                        'param_id': const_node.attrs.get('param_id', const_node.id),
                        'param_name': const_node.name,
                        'beta1': 0.9,
                        'beta2': 0.999,
                        'eps': 1e-8,
                    },
                    name=f'adam_update_{const_node.name}',
                )

        if (
            include_memory
            and getattr(config, 'memory_enabled', False)
            and self._memory_write_source_id is not None
            and self._memory_state_id is not None
        ):
            adam_updates = graph.find_nodes_by_op(OpCode.ADAM_FULL_UPDATE)
            dependency_id = adam_updates[-1].id if adam_updates else loss_node.id
            graph.add_node(
                op=OpCode.MEMORY_WRITE,
                inputs=[self._memory_write_source_id, self._memory_state_id, dependency_id],
                output_type=IRType(shape=(1,), grade_mask=GRADE_SCALAR),
                attrs={
                    'batch': B,
                    'seq': S,
                    'd_model': D,
                    'n_slots': getattr(config, 'memory_n_slots', 0),
                },
                name='memory_write',
            )

        return graph

    def _find_grad_for_node(self, node, grad_map, graph):
        """Find the gradient node for a forward node's output."""
        # Prefer the direct gradient for this node's output.
        if node.id in grad_map:
            return grad_map[node.id]
        # Check if any user of this node has a gradient
        users = graph.get_users(node.id)
        for user_id in users:
            if user_id in grad_map:
                return grad_map[user_id]
        return None

    def _merge_grad(self, graph: IRGraph, grad_map: Dict[int, int], target_id: int, grad_id: int):
        """Accumulate multiple gradient contributions for the same target."""
        existing = grad_map.get(target_id)
        if existing is None:
            grad_map[target_id] = grad_id
            return
        add_id = graph.add_node(
            op=OpCode.ADD,
            inputs=[existing, grad_id],
            output_type=graph.get_node(target_id).output_type,
            name=f'grad_accum_{target_id}',
        )
        grad_map[target_id] = add_id

    def _merge_input_grad(
        self,
        graph: IRGraph,
        grad_map: Dict[int, int],
        param_grad_map: Dict[int, int],
        target_id: int,
        grad_id: int,
    ):
        """Route gradients for parameter constants into param_grad_map."""
        target_node = graph.get_node(target_id)
        if (
            target_node.op == OpCode.CONSTANT
            and 'param_id' in target_node.attrs
            and 'const_data' not in target_node.attrs
        ):
            self._merge_grad(graph, param_grad_map, target_id, grad_id)
            return
        self._merge_grad(graph, grad_map, target_id, grad_id)

    def _ensure_memory_bank_state(self, model) -> int:
        """Create or reuse the mutable bank-state buffer backing compiled memory ops."""
        if self._memory_state_id is not None:
            return self._memory_state_id

        import numpy as _np
        from rune.backend import to_numpy

        n_slots = int(getattr(model.config, 'memory_n_slots', 0))
        state = _np.zeros((n_slots + 1, 8), dtype=_np.float32)
        bank = getattr(model, 'memory_bank', None)
        if bank is not None:
            bank_data = to_numpy(bank.bank)
            state[1:1 + bank_data.shape[0]] = bank_data
            state[0, 0] = float(getattr(bank, 'n_valid', 0))
            state[0, 1] = float(getattr(bank, 'write_head', 0))

        self._memory_state_id = self._add_static_constant(
            state,
            shape=(n_slots + 1,),
            grade_mask=GRADE_FULL,
            name='memory.bank_state',
            dtype='float32',
            storage_components=8,
        )
        return self._memory_state_id

    def _trace_mean_pool_seq(self, x_id: int, B: int, S: int, D: int, name: str) -> int:
        return self._graph.add_node(
            op=OpCode.MEAN_POOL_SEQ,
            inputs=[x_id],
            output_type=IRType(shape=(B, 1, D), grade_mask=GRADE_FULL),
            attrs={'batch': B, 'seq': S, 'd_model': D},
            name=name,
        )

    def _trace_memory_layer(self, x_id: int, B: int, S: int, D: int,
                            mem_layer, model, layer_idx: int) -> int:
        """Trace MemoryAttentionLayer using explicit query/read/gate nodes."""
        prefix = f'memory_layer{layer_idx}'
        pooled_id = self._trace_mean_pool_seq(
            x_id, B, S, D, name=f'{prefix}.mean_pool'
        )
        query_id = self._trace_linear(
            pooled_id, B, 1, D, 1, mem_layer.query_proj,
            name=f'{prefix}.query_proj', bias=True,
        )
        bank_state_id = self._ensure_memory_bank_state(model)
        read_id = self._graph.add_node(
            op=OpCode.MEMORY_READ,
            inputs=[query_id, bank_state_id],
            output_type=IRType(shape=(B, 1, 1), grade_mask=GRADE_FULL),
            attrs={
                'batch': B,
                'top_k': int(getattr(mem_layer, 'top_k', 1)),
                'n_slots': int(getattr(model.config, 'memory_n_slots', 0)),
            },
            name=f'{prefix}.memory_read',
        )
        context_id = self._trace_linear(
            read_id, B, 1, 1, D, mem_layer.memory_gate_proj,
            name=f'{prefix}.memory_gate_proj', bias=True,
        )
        gate_id = self._add_constant(
            mem_layer.gate_scalar,
            shape=(1,),
            grade_mask=GRADE_SCALAR,
            name=f'{prefix}.gate_scalar',
            dtype='float32',
            storage_components=1,
        )
        return self._graph.add_node(
            op=OpCode.MEMORY_GATE,
            inputs=[x_id, context_id, gate_id],
            output_type=IRType(shape=(B, S, D), grade_mask=GRADE_FULL),
            attrs={'batch': B, 'seq': S, 'd_model': D},
            name=f'{prefix}.memory_gate',
        )
