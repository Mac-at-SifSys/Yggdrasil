/**
 * persistent_engine.cu — Persistent CUDA Training Engine
 *
 * A single cooperative kernel that executes a full forward-backward-update
 * cycle on GPU without returning to Python. Uses grid-wide synchronization
 * (cooperative groups) between operations via a command buffer.
 *
 * YGGDRASIL Clifford Algebra Stack — mjolnir L0
 */

#include <cooperative_groups.h>
#include "engine_ops.cuh"

namespace cg = cooperative_groups;

// ===== Operation codes =====
enum EngineOp {
    OP_NOP = 0,
    OP_BATCH_GP,           // 1: Batched geometric product
    OP_BATCH_REVERSE,      // 2: Batched reverse
    OP_BATCH_SANDWICH,     // 3: Batched sandwich product
    OP_BATCH_BVEXP,        // 4: Batched bivector exp
    OP_BATCH_ADD,          // 5: Batched add
    OP_BATCH_SCALE,        // 6: Batched scale by scalar
    OP_BATCH_GRADE_PROJ,   // 7: Batched grade projection
    OP_BATCH_SCALAR_PROD,  // 8: Batched scalar product (grade-0 of GP)
    OP_BATCH_GELU,         // 9: Batched Clifford GELU activation
    OP_BATCH_NORM_SCALE,   // 10: RMS norm + scale (CliffordLayerNorm)
    OP_EMBED_LOOKUP,       // 11: Embedding table lookup
    OP_LINEAR_FWD,         // 12: CliffordLinear forward
    OP_ATTN_SCORE,         // 13: Attention scoring
    OP_SOFTMAX,            // 14: Softmax along last axis
    OP_WEIGHTED_SUM,       // 15: Scalar-weighted sum of multivectors
    OP_MATMUL_SCALAR,      // 16: Standard float matmul (logits)
    OP_CE_LOSS_FWD,        // 17: Cross-entropy forward + gradient
    OP_ADAM_STEP,          // 18: Adam optimizer update
    OP_GRAD_CLIP,          // 19: Gradient norm clipping
    OP_COPY,               // 20: memcpy device-to-device
    OP_ZERO,               // 21: zero out a buffer
    OP_ACCUMULATE,         // 22: out += input
    OP_BARRIER,            // 23: Grid-wide sync (implicit between ops)
    OP_DONE,               // 24: Exit kernel
    OP_TIED_LM_HEAD,       // 25: logits = x_scalar @ embed_scalar^T
    OP_TIED_LM_HEAD_BWD,   // 26: grad for tied embedding head
    OP_ADAM_FULL,          // 27: full Adam update on flat float buffers
    OP_BACKWARD_CE,        // 28: CE backward only
    OP_BACKWARD_LINEAR,    // 29: linear backward (input / weight / bias)
    OP_BACKWARD_GP,        // 30: GP backward (or attn-score backward)
    OP_BACKWARD_NORM,      // 31: layer-norm backward
    OP_BACKWARD_GELU,      // 32: grade-aware GELU backward
    OP_BACKWARD_EMBED,     // 33: embedding backward
    OP_BACKWARD_ADD,       // 34: residual-add backward
    OP_BACKWARD_MATMUL,    // 35: scalar matmul backward
    OP_BACKWARD_GRADE_PROJECT, // 36: grade-project backward
    OP_BACKWARD_BVEXP,     // 37: bivector-exp backward
    OP_BACKWARD_SOFTMAX,   // 38: softmax backward
    OP_BACKWARD_WEIGHTED_SUM, // 39: weighted-sum backward
    OP_BACKWARD_COPY,      // 40: structured copy backward
    OP_BACKWARD_ATTENTION, // 41: reserved
    OP_BACKWARD_FFN,       // 42: reserved
    OP_MEAN_POOL_SEQ,      // 43: mean-pool sequence dimension
    OP_MEMORY_READ,        // 44: read from persistent holographic memory bank
    OP_MEMORY_WRITE,       // 45: write distilled summary into memory bank
    OP_MEMORY_GATE,        // 46: gated residual injection of memory context
    OP_BACKWARD_MEMORY_GATE // 47: backward for gated memory injection
};

// Command descriptor — 64 bytes, cache-line aligned
struct __align__(64) EngineCommand {
    int opcode;         // EngineOp
    float* arg0;        // Input A / parameter
    float* arg1;        // Input B / gradient
    float* arg2;        // Output
    float* arg3;        // Auxiliary (bias, targets, etc.)
    int dim0;           // Primary dimension (N elements)
    int dim1;           // Secondary dimension
    int dim2;           // Tertiary dimension
    int dim3;           // Quaternary dimension
    float scalar0;      // Scalar arg (learning rate, scale, etc.)
    float scalar1;      // Second scalar arg
    int pad[16];        // Explicit padding so sizeof(EngineCommand) == 128 bytes
};

static constexpr int MEMORY_MAX_TOPK = 128;
static constexpr int MEMORY_MAX_DMODEL = 512;

// ===== The persistent kernel =====
__global__ void persistent_engine_kernel(
    EngineCommand* __restrict__ commands,
    int n_commands,
    float* __restrict__ loss_out,
    float* __restrict__ grad_norm_out
) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int cmd_idx = 0; cmd_idx < n_commands; cmd_idx++) {
        EngineCommand cmd = commands[cmd_idx];

        switch (cmd.opcode) {

        case OP_BATCH_GP: {
            int N = cmd.dim0;
            for (int i = tid; i < N; i += total_threads) {
                dev_gp(cmd.arg0 + i*8, cmd.arg1 + i*8, cmd.arg2 + i*8);
            }
            break;
        }

        case OP_BATCH_REVERSE: {
            int N = cmd.dim0;
            for (int i = tid; i < N; i += total_threads) {
                dev_reverse(cmd.arg0 + i*8, cmd.arg2 + i*8);
            }
            break;
        }

        case OP_BATCH_SANDWICH: {
            int N = cmd.dim0;
            for (int i = tid; i < N; i += total_threads) {
                dev_sandwich(cmd.arg0 + i*8, cmd.arg1 + i*8, cmd.arg2 + i*8);
            }
            break;
        }

        case OP_BATCH_BVEXP: {
            int N = cmd.dim0;
            int input_components = cmd.dim1 > 0 ? cmd.dim1 : 8;
            for (int i = tid; i < N; i += total_threads) {
                if (input_components == 3) {
                    float bv[8];
                    dev_zero(bv);
                    const float* in = cmd.arg0 + i * 3;
                    bv[4] = in[0];
                    bv[5] = in[1];
                    bv[6] = in[2];
                    dev_bivector_exp(bv, cmd.arg2 + i * 8);
                } else {
                    dev_bivector_exp(cmd.arg0 + i*8, cmd.arg2 + i*8);
                }
            }
            break;
        }

        case OP_BATCH_ADD: {
            int N = cmd.dim0;
            int components = cmd.dim1 > 0 ? cmd.dim1 : 8;
            if (components == 8) {
                for (int i = tid; i < N; i += total_threads) {
                    dev_add(cmd.arg0 + i*8, cmd.arg1 + i*8, cmd.arg2 + i*8);
                }
            } else {
                int total = N * components;
                for (int idx = tid; idx < total; idx += total_threads) {
                    cmd.arg2[idx] = cmd.arg0[idx] + cmd.arg1[idx];
                }
            }
            break;
        }

        case OP_BATCH_SCALE: {
            int N = cmd.dim0;
            float s = cmd.scalar0;
            for (int i = tid; i < N; i += total_threads) {
                dev_scale(cmd.arg0 + i*8, s, cmd.arg2 + i*8);
            }
            break;
        }

        case OP_BATCH_GRADE_PROJ: {
            int N = cmd.dim0;
            int grade = cmd.dim1;
            int output_components = cmd.dim2 > 0 ? cmd.dim2 : 8;
            int start = GRADE_START[grade], end = GRADE_END[grade];
            int width = end - start;
            if (output_components == 8) {
                for (int i = tid; i < N; i += total_threads) {
                    float* out = cmd.arg2 + i*8;
                    const float* in = cmd.arg0 + i*8;
                    #pragma unroll
                    for (int k = 0; k < 8; k++)
                        out[k] = (k >= start && k < end) ? in[k] : 0.0f;
                }
            } else {
                for (int i = tid; i < N; i += total_threads) {
                    float* out = cmd.arg2 + i * output_components;
                    const float* in = cmd.arg0 + i * 8;
                    for (int lane = 0; lane < output_components; lane++) {
                        out[lane] = (lane < width) ? in[start + lane] : 0.0f;
                    }
                }
            }
            break;
        }

        case OP_BATCH_SCALAR_PROD: {
            // Batched scalar product: out[i] = scalar_part(GP(a[i], b[i]))
            // arg0 = A (N, 8), arg1 = B (N, 8), arg2 = out (N) floats
            int N = cmd.dim0;
            for (int i = tid; i < N; i += total_threads) {
                cmd.arg2[i] = dev_scalar_product(cmd.arg0 + i*8, cmd.arg1 + i*8);
            }
            break;
        }

        case OP_BATCH_GELU: {
            int N = cmd.dim0;
            for (int i = tid; i < N; i += total_threads) {
                dev_clifford_gelu(cmd.arg0 + i*8, cmd.arg2 + i*8);
            }
            break;
        }

        case OP_LINEAR_FWD: {
            // CliffordLinear forward: y[b,j] = sum_i GP(W[j,i], x[b,i]) + bias[j]
            // arg0 = W (d_out, d_in, 8), arg1 = x (batch, d_in, 8)
            // arg2 = y (batch, d_out, 8), arg3 = bias (d_out, 8) or NULL
            // dim0 = batch, dim1 = d_in, dim2 = d_out
            int batch = cmd.dim0, d_in = cmd.dim1, d_out = cmd.dim2;
            int total_outputs = batch * d_out;
            for (int idx = tid; idx < total_outputs; idx += total_threads) {
                int b = idx / d_out;
                int j = idx % d_out;
                float* y_ptr = cmd.arg2 + (b * d_out + j) * 8;
                dev_zero(y_ptr);
                float tmp[8];
                for (int i = 0; i < d_in; i++) {
                    const float* w_ptr = cmd.arg0 + (j * d_in + i) * 8;
                    const float* x_ptr = cmd.arg1 + (b * d_in + i) * 8;
                    dev_gp(w_ptr, x_ptr, tmp);
                    dev_add(y_ptr, tmp, y_ptr);
                }
                if (cmd.arg3 != NULL) {
                    const float* b_ptr = cmd.arg3 + j * 8;
                    dev_add(y_ptr, b_ptr, y_ptr);
                }
            }
            break;
        }

        case OP_ATTN_SCORE: {
            // score[b,h,i,j] = sum_d scalar_part(Q[b,i,h,d] * ~K[b,j,h,d]) * scale
            // arg0 = Q (batch, seq, H*d_head, 8), arg1 = K (batch, seq, H*d_head, 8)
            // arg2 = scores (batch, H, seq, seq)
            // dim0 = batch, dim1 = H, dim2 = seq_len, dim3 = d_head
            int batch = cmd.dim0, n_heads = cmd.dim1, seq = cmd.dim2, d_head = cmd.dim3;
            float scale = cmd.scalar0;
            int total_scores = batch * n_heads * seq * seq;
            for (int idx = tid; idx < total_scores; idx += total_threads) {
                int per_batch = n_heads * seq * seq;
                int b = idx / per_batch;
                int rem0 = idx % per_batch;
                int h = rem0 / (seq * seq);
                int rem = rem0 % (seq * seq);
                int qi = rem / seq;
                int kj = rem % seq;
                if (kj > qi) {
                    cmd.arg2[idx] = -1.0e9f;
                    continue;
                }
                float score = 0.0f;
                for (int d = 0; d < d_head; d++) {
                    int feature = h * d_head + d;
                    const float* q_ptr = cmd.arg0 + ((b * seq + qi) * (n_heads * d_head) + feature) * 8;
                    const float* k_ptr = cmd.arg1 + ((b * seq + kj) * (n_heads * d_head) + feature) * 8;
                    float k_rev[8];
                    dev_reverse(k_ptr, k_rev);
                    score += dev_scalar_product(q_ptr, k_rev);
                }
                cmd.arg2[idx] = score * scale;
            }
            break;
        }

        case OP_SOFTMAX: {
            // Softmax along last dimension
            // arg0 = input (N, D), arg2 = output (N, D)
            // dim0 = N, dim1 = D
            int N = cmd.dim0, D = cmd.dim1;
            for (int row = tid; row < N; row += total_threads) {
                const float* in_row = cmd.arg0 + row * D;
                float* out_row = cmd.arg2 + row * D;
                float max_val = in_row[0];
                for (int d = 1; d < D; d++)
                    if (in_row[d] > max_val) max_val = in_row[d];
                float sum = 0.0f;
                for (int d = 0; d < D; d++) {
                    out_row[d] = expf(in_row[d] - max_val);
                    sum += out_row[d];
                }
                float inv_sum = 1.0f / (sum + 1e-12f);
                for (int d = 0; d < D; d++)
                    out_row[d] *= inv_sum;
            }
            break;
        }

        case OP_WEIGHTED_SUM: {
            // out[b,i,h,d,:] = sum_j weights[b,h,i,j] * V[b,j,h,d,:]
            // arg0 = weights (batch, H, seq, seq), arg1 = V (batch, seq, H*d_head, 8)
            // arg2 = out (batch, seq, H*d_head, 8)
            // dim0 = batch, dim1 = H, dim2 = seq, dim3 = d_head
            int batch = cmd.dim0, n_heads = cmd.dim1, seq = cmd.dim2, d_head = cmd.dim3;
            int total = batch * n_heads * seq * d_head;
            for (int idx = tid; idx < total; idx += total_threads) {
                int per_batch = n_heads * seq * d_head;
                int b = idx / per_batch;
                int rem0 = idx % per_batch;
                int h = rem0 / (seq * d_head);
                int rem = rem0 % (seq * d_head);
                int qi = rem / d_head;
                int d = rem % d_head;
                int feature = h * d_head + d;
                float* out_ptr = cmd.arg2 + ((b * seq + qi) * (n_heads * d_head) + feature) * 8;
                dev_zero(out_ptr);
                for (int j = 0; j < seq; j++) {
                    int weight_idx = ((b * n_heads + h) * seq + qi) * seq + j;
                    float w = cmd.arg0[weight_idx];
                    const float* v_ptr = cmd.arg1 + ((b * seq + j) * (n_heads * d_head) + feature) * 8;
                    #pragma unroll
                    for (int c = 0; c < 8; c++)
                        out_ptr[c] += w * v_ptr[c];
                }
            }
            break;
        }

        case OP_MEAN_POOL_SEQ: {
            // arg0 = x (batch, seq, d_model, 8), arg2 = pooled (batch, 1, d_model, 8)
            int batch = cmd.dim0, seq = cmd.dim1, d_model = cmd.dim2;
            int total = batch * d_model;
            for (int idx = tid; idx < total; idx += total_threads) {
                int b = idx / d_model;
                int d = idx % d_model;
                float* out = cmd.arg2 + (b * d_model + d) * 8;
                dev_zero(out);
                for (int s = 0; s < seq; s++) {
                    const float* x = cmd.arg0 + ((b * seq + s) * d_model + d) * 8;
                    #pragma unroll
                    for (int c = 0; c < 8; c++) out[c] += x[c];
                }
                float inv_seq = 1.0f / fmaxf((float)seq, 1.0f);
                #pragma unroll
                for (int c = 0; c < 8; c++) out[c] *= inv_seq;
            }
            break;
        }

        case OP_MEMORY_READ: {
            // arg0 = query (batch, 1, 1, 8), arg1 = bank_state ((n_slots + 1), 8), arg2 = out (batch, 1, 1, 8)
            int batch = cmd.dim0;
            int top_k = cmd.dim1 > 0 ? cmd.dim1 : 1;
            int n_slots = cmd.dim2;
            if (top_k > MEMORY_MAX_TOPK) top_k = MEMORY_MAX_TOPK;
            int n_valid = 0;
            if (cmd.arg1 != NULL) {
                n_valid = (int)(cmd.arg1[0] + 0.5f);
                if (n_valid < 0) n_valid = 0;
                if (n_valid > n_slots) n_valid = n_slots;
            }
            const float* bank = cmd.arg1 != NULL ? (cmd.arg1 + 8) : NULL;
            for (int b = tid; b < batch; b += total_threads) {
                float* out = cmd.arg2 + b * 8;
                dev_zero(out);
                if (bank == NULL || n_valid <= 0) {
                    continue;
                }
                const float* query = cmd.arg0 + b * 8;
                float top_scores[MEMORY_MAX_TOPK];
                int top_indices[MEMORY_MAX_TOPK];
                int found = 0;
                for (int i = 0; i < top_k; i++) {
                    top_scores[i] = -3.402823466e38F;
                    top_indices[i] = -1;
                }
                for (int slot = 0; slot < n_valid; slot++) {
                    const float* mem = bank + slot * 8;
                    float score = dev_scalar_product(query, mem);
                    if (found < top_k) {
                        top_scores[found] = score;
                        top_indices[found] = slot;
                        found++;
                        continue;
                    }
                    int min_idx = 0;
                    float min_score = top_scores[0];
                    for (int i = 1; i < top_k; i++) {
                        if (top_scores[i] < min_score) {
                            min_score = top_scores[i];
                            min_idx = i;
                        }
                    }
                    if (score > min_score) {
                        top_scores[min_idx] = score;
                        top_indices[min_idx] = slot;
                    }
                }
                if (found <= 0) {
                    continue;
                }
                float max_score = top_scores[0];
                for (int i = 1; i < found; i++) {
                    if (top_scores[i] > max_score) max_score = top_scores[i];
                }
                float weight_sum = 0.0f;
                float weights[MEMORY_MAX_TOPK];
                for (int i = 0; i < found; i++) {
                    weights[i] = expf(top_scores[i] - max_score);
                    weight_sum += weights[i];
                }
                float inv_weight_sum = 1.0f / (weight_sum + 1e-12f);
                for (int i = 0; i < found; i++) {
                    const float* mem = bank + top_indices[i] * 8;
                    float interaction[8];
                    dev_gp(query, mem, interaction);
                    float w = weights[i] * inv_weight_sum;
                    #pragma unroll
                    for (int c = 0; c < 8; c++) out[c] += w * interaction[c];
                }
            }
            break;
        }

        case OP_MEMORY_GATE: {
            // arg0 = x (batch, seq, d_model, 8), arg1 = context (batch, 1, d_model, 8), arg3 = gate scalar, arg2 = out
            int batch = cmd.dim0, seq = cmd.dim1, d_model = cmd.dim2;
            float gate = cmd.arg3 != NULL ? dev_sigmoid(cmd.arg3[0]) : 0.0f;
            int total = batch * seq * d_model * 8;
            for (int idx = tid; idx < total; idx += total_threads) {
                int elem = idx / 8;
                int c = idx % 8;
                int d = elem % d_model;
                int bs = elem / d_model;
                int b = bs / seq;
                cmd.arg2[idx] = cmd.arg0[idx] + gate * cmd.arg1[(b * d_model + d) * 8 + c];
            }
            break;
        }

        case OP_EMBED_LOOKUP: {
            // arg0 = embedding table (vocab, d_model, 8)
            // arg1 = token_ids (N) as int* cast to float*
            // arg2 = output (N, d_model, 8)
            // dim0 = N, dim1 = d_model
            int N = cmd.dim0, d_model = cmd.dim1;
            const int* ids = (const int*)cmd.arg1;
            for (int idx = tid; idx < N * d_model; idx += total_threads) {
                int pos = idx / d_model;
                int d = idx % d_model;
                int token = ids[pos];
                dev_copy(cmd.arg0 + (token * d_model + d) * 8,
                         cmd.arg2 + (pos * d_model + d) * 8);
            }
            break;
        }

        case OP_MATMUL_SCALAR: {
            // mode 0:
            //   C = A @ B^T, with A=(M,K), B=(N,K), C=(M,N)
            // mode 1:
            //   A=(M,K) scalar, B=(N,K,8) multivector, C=(M,N,8) multivector
            //   Only the scalar lane of B participates; the result is written
            //   into the scalar lane of C.
            int M = cmd.dim0, K = cmd.dim1, N = cmd.dim2;
            int mode = cmd.dim3;
            for (int idx = tid; idx < M * N; idx += total_threads) {
                int m = idx / N, n = idx % N;
                float sum = 0.0f;
                if (mode == 1) {
                    for (int k = 0; k < K; k++) {
                        sum += cmd.arg0[m * K + k] * cmd.arg1[(n * K + k) * 8];
                    }
                    float* out = cmd.arg2 + idx * 8;
                    out[0] = sum;
                    #pragma unroll
                    for (int c = 1; c < 8; c++) out[c] = 0.0f;
                } else {
                    for (int k = 0; k < K; k++)
                        sum += cmd.arg0[m * K + k] * cmd.arg1[n * K + k];
                    cmd.arg2[idx] = sum;
                }
            }
            break;
        }

        case OP_BATCH_NORM_SCALE: {
            // CliffordLayerNorm: normalize by RMS, scale by gamma (GP), add beta
            // arg0 = input (N, d_model, 8), arg1 = gamma (d_model, 8)
            // arg3 = beta (d_model, 8), arg2 = output (N, d_model, 8)
            // dim0 = N, dim1 = d_model, scalar0 = eps
            int N = cmd.dim0, d_model = cmd.dim1;
            float eps = cmd.scalar0;
            for (int n = tid; n < N; n += total_threads) {
                // Compute RMS norm across d_model
                float sum_sq = 0.0f;
                for (int d = 0; d < d_model; d++) {
                    const float* mv = cmd.arg0 + (n * d_model + d) * 8;
                    float r[8];
                    dev_reverse(mv, r);
                    float ns = fabsf(dev_scalar_product(mv, r));
                    sum_sq += ns;
                }
                float rms = sqrtf(sum_sq / d_model + eps);
                float inv_rms = 1.0f / rms;

                for (int d = 0; d < d_model; d++) {
                    const float* mv = cmd.arg0 + (n * d_model + d) * 8;
                    const float* gamma = cmd.arg1 + d * 8;
                    const float* beta = cmd.arg3 + d * 8;
                    float* out = cmd.arg2 + (n * d_model + d) * 8;

                    float normed[8];
                    dev_scale(mv, inv_rms, normed);
                    float scaled[8];
                    dev_gp(gamma, normed, scaled);
                    dev_add(scaled, beta, out);
                }
            }
            break;
        }

        case OP_CE_LOSS_FWD: {
            // Cross-entropy loss + gradient
            // arg0 = logits (N, V), arg1 = targets (N) as int*
            // arg2 = grad_logits (N, V) or NULL for loss-only
            // dim0 = N, dim1 = V
            int N = cmd.dim0, V = cmd.dim1;
            const int* targets = (const int*)cmd.arg1;

            for (int n = tid; n < N; n += total_threads) {
                float* logits_row = cmd.arg0 + n * V;
                float* grad_row = (cmd.arg2 != NULL) ? (cmd.arg2 + n * V) : NULL;
                int target = targets[n];

                // Numerically stable softmax
                float max_val = logits_row[0];
                for (int v = 1; v < V; v++)
                    if (logits_row[v] > max_val) max_val = logits_row[v];
                float sum = 0.0f;
                for (int v = 0; v < V; v++) {
                    float e = expf(logits_row[v] - max_val);
                    if (grad_row != NULL) grad_row[v] = e;
                    sum += e;
                }
                float inv_sum = 1.0f / (sum + 1e-12f);
                float target_prob = expf(logits_row[target] - max_val) * inv_sum;
                float nll = -logf(target_prob + 1e-12f);

                // Gradient: softmax - one_hot, averaged over N
                if (grad_row != NULL) {
                    for (int v = 0; v < V; v++) {
                        grad_row[v] = (grad_row[v] * inv_sum - (v == target ? 1.0f : 0.0f)) / N;
                    }
                }

                // Atomic add loss
                atomicAdd(loss_out, nll / N);
            }
            break;
        }

        case OP_ADAM_STEP: {
            // Simplified param update: param -= lr * grad
            // arg0 = param (N*8 floats), arg1 = grad (N*8 floats)
            // dim0 = N (multivectors), scalar0 = lr
            int N = cmd.dim0;
            float lr = cmd.scalar0;
            for (int i = tid; i < N * 8; i += total_threads) {
                cmd.arg0[i] -= lr * cmd.arg1[i];
            }
            break;
        }

        case OP_ZERO: {
            int N = cmd.dim0;  // total floats
            for (int i = tid; i < N; i += total_threads) {
                cmd.arg2[i] = 0.0f;
            }
            break;
        }

        case OP_ACCUMULATE: {
            int N = cmd.dim0;
            for (int i = tid; i < N; i += total_threads) {
                cmd.arg2[i] += cmd.arg0[i];
            }
            break;
        }

        case OP_COPY: {
            if (cmd.dim1 == 0 && cmd.dim2 == 0) {
                int N = cmd.dim0;
                for (int i = tid; i < N; i += total_threads) {
                    cmd.arg2[i] = cmd.arg0[i];
                }
            } else {
                // Structured slice copy for head concat/split.
                // dim0 = n_tokens, dim1 = d_slice, dim2 = d_total, dim3 = offset
                // scalar0 > 0.5 => forward concat slice -> full
                // scalar0 <= 0.5 => backward split full -> slice
                int n_tokens = cmd.dim0;
                int d_slice = cmd.dim1;
                int d_total = cmd.dim2;
                int offset = cmd.dim3;
                bool forward_concat = cmd.scalar0 > 0.5f;
                int total = n_tokens * d_slice;
                for (int idx = tid; idx < total; idx += total_threads) {
                    int tok = idx / d_slice;
                    int d = idx % d_slice;
                    if (forward_concat) {
                        float* dst = cmd.arg2 + ((tok * d_total) + offset + d) * 8;
                        const float* src = cmd.arg0 + (idx * 8);
                        dev_copy(src, dst);
                    } else {
                        float* dst = cmd.arg2 + (idx * 8);
                        const float* src = cmd.arg0 + (((tok * d_total) + offset + d) * 8);
                        dev_copy(src, dst);
                    }
                }
            }
            break;
        }

case OP_GRAD_CLIP: {
            int N = cmd.dim0;
            int mode = cmd.dim1;
            float max_norm = cmd.scalar0;

            if (mode == 0 || mode == 2) {
                float local_sum = 0.0f;
                for (int i = tid; i < N; i += total_threads) {
                    float val = cmd.arg0[i];
                    local_sum += val * val;
                }
                atomicAdd(grad_norm_out, local_sum);
            }

            grid.sync();

            if (mode == 1 || mode == 2) {
                float global_norm = sqrtf(*grad_norm_out);
                if (global_norm > max_norm) {
                    float clip_coef = max_norm / (global_norm + 1e-6f);
                    for (int i = tid; i < N; i += total_threads) {
                        cmd.arg0[i] *= clip_coef;
                    }
                }
            }
            break;
        }

        case OP_BARRIER:
            // Explicit barrier (implicit between all ops via grid.sync below)
            break;

        case OP_DONE:
            return;

        case OP_TIED_LM_HEAD: {
            // logits[n, v] = sum_d hidden[n, d, 0] * embed[v, d, 0]
            int N = cmd.dim0, D = cmd.dim1, V = cmd.dim2;
            int total = N * V;
            for (int idx = tid; idx < total; idx += total_threads) {
                int n = idx / V;
                int v = idx % V;
                float sum = 0.0f;
                for (int d = 0; d < D; d++) {
                    sum += cmd.arg0[(n * D + d) * 8] * cmd.arg1[(v * D + d) * 8];
                }
                cmd.arg2[idx] = sum;
            }
            break;
        }

        case OP_ADAM_FULL: {
            // Clifford-aware Adam on grouped float buffers.
            // arg0 = param, arg1 = grad, arg2 = m, arg3 = v
            // dim0 = n_floats, dim1 = step, dim2 = beta1*1e4, dim3 = beta2*1e4
            // scalar0 = lr, scalar1 = eps, pad[0] = group size
            int N = cmd.dim0;
            int step = cmd.dim1;
            float beta1 = ((float)cmd.dim2) / 10000.0f;
            float beta2 = ((float)cmd.dim3) / 10000.0f;
            float lr = cmd.scalar0;
            float eps = cmd.scalar1;
            int group_size = cmd.pad[0] > 0 ? cmd.pad[0] : 8;
            float bias_c1 = 1.0f - powf(beta1, (float)step);
            float bias_c2 = 1.0f - powf(beta2, (float)step);

            int n_groups = (N + group_size - 1) / group_size;
            for (int group_idx = tid; group_idx < n_groups; group_idx += total_threads) {
                int base = group_idx * group_size;
                int valid = N - base;
                if (valid > group_size) valid = group_size;
                if (valid <= 0) continue;

                float norm_sq = 0.0f;
                if (group_size == 8 && valid == 8) {
                    float g_clamped_mv[8];
                    #pragma unroll
                    for (int lane = 0; lane < 8; lane++) {
                        float g = cmd.arg1[base + lane];
                        g_clamped_mv[lane] = fminf(1e4f, fmaxf(-1e4f, g));
                    }
                    norm_sq = fabsf(dev_norm_sq(g_clamped_mv));
                } else {
                    for (int lane = 0; lane < valid; lane++) {
                        float g = cmd.arg1[base + lane];
                        float g_clamped = fminf(1e4f, fmaxf(-1e4f, g));
                        norm_sq += g_clamped * g_clamped;
                    }
                    norm_sq = fabsf(norm_sq);
                }

                for (int lane = 0; lane < valid; lane++) {
                    int idx = base + lane;
                    float g = cmd.arg1[idx];
                    if (grad_norm_out != NULL) atomicAdd(grad_norm_out, g * g);
                    float m = beta1 * cmd.arg2[idx] + (1.0f - beta1) * g;
                    float v = beta2 * cmd.arg3[idx] + (1.0f - beta2) * norm_sq;
                    cmd.arg2[idx] = m;
                    cmd.arg3[idx] = v;
                    float m_hat = m / (bias_c1 + 1e-12f);
                    float v_hat = v / (bias_c2 + 1e-12f);
                    float grade_scale = (group_size == 8 && lane == 7) ? 0.5f : 1.0f;
                    cmd.arg0[idx] -= grade_scale * lr * m_hat / (sqrtf(fmaxf(v_hat, 1e-30f)) + eps);
                }
            }
            break;
        }

        case OP_BACKWARD_CE: {
            // arg0 = logits (N, V), arg1 = targets (N) as int*, arg2 = grad_logits (N, V)
            int N = cmd.dim0, V = cmd.dim1;
            const int* targets = (const int*)cmd.arg1;
            for (int n = tid; n < N; n += total_threads) {
                const float* logits_row = cmd.arg0 + n * V;
                float* grad_row = cmd.arg2 + n * V;
                int target = targets[n];
                float max_val = logits_row[0];
                for (int v = 1; v < V; v++)
                    if (logits_row[v] > max_val) max_val = logits_row[v];
                float sum = 0.0f;
                for (int v = 0; v < V; v++) {
                    grad_row[v] = expf(logits_row[v] - max_val);
                    sum += grad_row[v];
                }
                float inv_sum = 1.0f / (sum + 1e-12f);
                for (int v = 0; v < V; v++) {
                    grad_row[v] = (grad_row[v] * inv_sum - (v == target ? 1.0f : 0.0f)) / N;
                }
            }
            break;
        }

        case OP_BACKWARD_ADD: {
            int N = cmd.dim0;
            for (int i = tid; i < N * 8; i += total_threads) {
                cmd.arg2[i] = cmd.arg0[i];
            }
            break;
        }

        case OP_BACKWARD_GRADE_PROJECT: {
            int N = cmd.dim0;
            int grade = cmd.dim1;
            if (grade == 0) {
                for (int i = tid; i < N; i += total_threads) {
                    float* out = cmd.arg2 + i * 8;
                    out[0] = cmd.arg0[i];
                    #pragma unroll
                    for (int c = 1; c < 8; c++) out[c] = 0.0f;
                }
            } else {
                int start = GRADE_START[grade];
                int end = GRADE_END[grade];
                int width = end - start;
                for (int i = tid; i < N; i += total_threads) {
                    float* out = cmd.arg2 + i * 8;
                    #pragma unroll
                    for (int c = 0; c < 8; c++) out[c] = 0.0f;
                    for (int lane = 0; lane < width; lane++) {
                        out[start + lane] = cmd.arg0[i * width + lane];
                    }
                }
            }
            break;
        }

        case OP_BACKWARD_BVEXP: {
            int N = cmd.dim0;
            int input_components = cmd.dim1 > 0 ? cmd.dim1 : 8;
            for (int i = tid; i < N; i += total_threads) {
                float bv[8];
                float grad_bv[8];
                if (input_components == 3) {
                    dev_zero(bv);
                    const float* in = cmd.arg1 + i * 3;
                    bv[4] = in[0];
                    bv[5] = in[1];
                    bv[6] = in[2];
                } else {
                    dev_copy(cmd.arg1 + i * 8, bv);
                }
                dev_bivector_exp_backward(cmd.arg0 + i * 8, bv, grad_bv);
                if (input_components == 3) {
                    float* out = cmd.arg2 + i * 3;
                    out[0] = grad_bv[4];
                    out[1] = grad_bv[5];
                    out[2] = grad_bv[6];
                } else {
                    dev_copy(grad_bv, cmd.arg2 + i * 8);
                }
            }
            break;
        }

        case OP_BACKWARD_MATMUL: {
            // arg0 = grad_out, arg1 = lhs/input, arg2 = out, arg3 = rhs/weight
            // dim0 = M, dim1 = K, dim2 = N, dim3 = mode (0=input, 2=weight)
            int M = cmd.dim0, K = cmd.dim1, N = cmd.dim2;
            int mode = cmd.dim3;
            if (mode == 2) {
                // grad_weight[n, d] = sum_m grad[m, n] * x[m, d]
                int total = M * N; // interpreted as N(output rows) x D(output cols)
                total = cmd.dim0 * cmd.dim2;
                for (int idx = tid; idx < total; idx += total_threads) {
                    int n = idx / N;
                    int d = idx % N;
                    float sum = 0.0f;
                    for (int m = 0; m < K; m++) {
                        sum += cmd.arg0[m * M + n] * cmd.arg1[m * N + d];
                    }
                    cmd.arg2[idx] = sum;
                }
            } else {
                // grad_input[m, n] = sum_k grad[m, k] * weight[k, n]
                int total = M * N;
                for (int idx = tid; idx < total; idx += total_threads) {
                    int m = idx / N;
                    int n = idx % N;
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += cmd.arg0[m * K + k] * cmd.arg3[k * N + n];
                    }
                    cmd.arg2[idx] = sum;
                }
            }
            break;
        }

        case OP_BACKWARD_GELU: {
            int N = cmd.dim0;
            for (int i = tid; i < N; i += total_threads) {
                dev_clifford_gelu_backward(cmd.arg1 + i * 8, cmd.arg0 + i * 8, cmd.arg2 + i * 8);
            }
            break;
        }

        case OP_BACKWARD_EMBED: {
            // arg0 = grad_output (N, D, 8), arg1 = token_ids (N), arg2 = grad_embed (V, D, 8)
            int N = cmd.dim0, D = cmd.dim1;
            const int* ids = (const int*)cmd.arg1;
            int total = N * D;
            for (int idx = tid; idx < total; idx += total_threads) {
                int n = idx / D;
                int d = idx % D;
                int token = ids[n];
                const float* g = cmd.arg0 + idx * 8;
                float* out = cmd.arg2 + (token * D + d) * 8;
                #pragma unroll
                for (int c = 0; c < 8; c++) atomicAdd(out + c, g[c]);
            }
            break;
        }

        case OP_BACKWARD_LINEAR: {
            // arg0 = grad_output, arg1 = weight or x, arg2 = out
            // dim0 = batch_tokens, dim1 = d_in, dim2 = d_out, dim3 = mode
            int B = cmd.dim0, d_in = cmd.dim1, d_out = cmd.dim2, mode = cmd.dim3;
            if (mode == 2) {
                // Bias gradient: sum over batch tokens
                for (int j = tid; j < d_out; j += total_threads) {
                    float accum[8];
                    dev_zero(accum);
                    for (int b = 0; b < B; b++) {
                        const float* g = cmd.arg0 + (b * d_out + j) * 8;
                        #pragma unroll
                        for (int c = 0; c < 8; c++) accum[c] += g[c];
                    }
                    dev_copy(accum, cmd.arg2 + j * 8);
                }
            } else if (mode == 1) {
                // Weight gradient: grad_w[j,i] = sum_b GP(grad[b,j], reverse(x[b,i]))
                int total = d_out * d_in;
                for (int idx = tid; idx < total; idx += total_threads) {
                    int j = idx / d_in;
                    int i = idx % d_in;
                    float accum[8];
                    dev_zero(accum);
                    for (int b = 0; b < B; b++) {
                        const float* g = cmd.arg0 + (b * d_out + j) * 8;
                        const float* x = cmd.arg1 + (b * d_in + i) * 8;
                        float x_rev[8], tmp[8];
                        dev_reverse(x, x_rev);
                        dev_gp(g, x_rev, tmp);
                        dev_add(accum, tmp, accum);
                    }
                    dev_copy(accum, cmd.arg2 + idx * 8);
                }
            } else {
                // Input gradient: grad_x[b,i] = sum_j GP(reverse(W[j,i]), grad[b,j])
                int total = B * d_in;
                for (int idx = tid; idx < total; idx += total_threads) {
                    int b = idx / d_in;
                    int i = idx % d_in;
                    float accum[8];
                    dev_zero(accum);
                    for (int j = 0; j < d_out; j++) {
                        const float* w = cmd.arg1 + (j * d_in + i) * 8;
                        const float* g = cmd.arg0 + (b * d_out + j) * 8;
                        float w_rev[8], tmp[8];
                        dev_reverse(w, w_rev);
                        dev_gp(w_rev, g, tmp);
                        dev_add(accum, tmp, accum);
                    }
                    dev_copy(accum, cmd.arg2 + idx * 8);
                }
            }
            break;
        }

        case OP_BACKWARD_GP: {
            if (cmd.dim3 == 10 || cmd.dim3 == 11) {
                // Attention-score backward.
                int batch = cmd.dim0, seq = cmd.dim1, d_head = cmd.dim2;
                bool right = (cmd.dim3 == 11);
                int total = batch * seq * d_head;
                float scale = cmd.scalar0;
                for (int idx = tid; idx < total; idx += total_threads) {
                    int b = idx / (seq * d_head);
                    int rem = idx % (seq * d_head);
                    int q = rem / d_head;
                    int d = rem % d_head;
                    float* out = cmd.arg2 + idx * 8;
                    dev_zero(out);
                    if (!right) {
                        for (int k = 0; k < seq; k++) {
                            float g = cmd.arg0[(b * seq + q) * seq + k] * scale;
                            const float* kv = cmd.arg3 + ((b * seq + k) * d_head + d) * 8;
                            #pragma unroll
                            for (int c = 0; c < 8; c++) out[c] += g * kv[c];
                        }
                    } else {
                        int k = q;
                        int total_q = seq;
                        for (int q_idx = 0; q_idx < total_q; q_idx++) {
                            float g = cmd.arg0[(b * seq + q_idx) * seq + k] * scale;
                            const float* qv = cmd.arg1 + ((b * seq + q_idx) * d_head + d) * 8;
                            #pragma unroll
                            for (int c = 0; c < 8; c++) out[c] += g * qv[c];
                        }
                    }
                }
            } else {
                int N = cmd.dim0;
                bool right = (cmd.dim1 == 1);
                for (int i = tid; i < N; i += total_threads) {
                    if (right) {
                        float a_rev[8];
                        dev_reverse(cmd.arg1 + i * 8, a_rev);
                        dev_gp(a_rev, cmd.arg0 + i * 8, cmd.arg2 + i * 8);
                    } else {
                        float b_rev[8];
                        dev_reverse(cmd.arg3 + i * 8, b_rev);
                        dev_gp(cmd.arg0 + i * 8, b_rev, cmd.arg2 + i * 8);
                    }
                }
            }
            break;
        }

        case OP_BACKWARD_NORM: {
            // arg0 = grad_output, arg1 = x, arg2 = out, arg3 = gamma
            // dim0 = batch_tokens, dim1 = d_model, dim2 = mode
            int B = cmd.dim0, D = cmd.dim1, mode = cmd.dim2;
            float eps = cmd.scalar0;
            if (mode == 2) {
                for (int d = tid; d < D; d += total_threads) {
                    float accum[8];
                    dev_zero(accum);
                    for (int b = 0; b < B; b++) {
                        const float* g = cmd.arg0 + (b * D + d) * 8;
                        #pragma unroll
                        for (int c = 0; c < 8; c++) accum[c] += g[c];
                    }
                    dev_copy(accum, cmd.arg2 + d * 8);
                }
            } else if (mode == 1) {
                for (int d = tid; d < D; d += total_threads) {
                    float accum[8];
                    dev_zero(accum);
                    for (int b = 0; b < B; b++) {
                        // Recompute x_hat
                        float sum_sq = 0.0f;
                        for (int j = 0; j < D; j++) {
                            const float* mv = cmd.arg1 + (b * D + j) * 8;
                            #pragma unroll
                            for (int c = 0; c < 8; c++) sum_sq += mv[c] * mv[c];
                        }
                        float rms = sqrtf(sum_sq / D + 2.0f * eps);
                        float x_hat[8];
                        const float* x = cmd.arg1 + (b * D + d) * 8;
                        #pragma unroll
                        for (int c = 0; c < 8; c++) x_hat[c] = x[c] / rms;
                        float x_hat_rev[8], tmp[8];
                        dev_reverse(x_hat, x_hat_rev);
                        dev_gp(cmd.arg0 + (b * D + d) * 8, x_hat_rev, tmp);
                        dev_add(accum, tmp, accum);
                    }
                    dev_copy(accum, cmd.arg2 + d * 8);
                }
            } else {
                int total = B * D;
                for (int idx = tid; idx < total; idx += total_threads) {
                    int b = idx / D;
                    int d = idx % D;
                    float sum_sq = 0.0f;
                    for (int j = 0; j < D; j++) {
                        const float* mv = cmd.arg1 + (b * D + j) * 8;
                        #pragma unroll
                        for (int c = 0; c < 8; c++) sum_sq += mv[c] * mv[c];
                    }
                    float rms = sqrtf(sum_sq / D + 2.0f * eps);
                    float inv_rms = 1.0f / (rms + 1e-12f);

                    float grad_x_hat[8];
                    float gamma_rev[8];
                    dev_reverse(cmd.arg3 + d * 8, gamma_rev);
                    dev_gp(gamma_rev, cmd.arg0 + idx * 8, grad_x_hat);

                    float inner = 0.0f;
                    for (int j = 0; j < D; j++) {
                        float grad_tmp[8];
                        float gamma_rev_j[8];
                        dev_reverse(cmd.arg3 + j * 8, gamma_rev_j);
                        dev_gp(gamma_rev_j, cmd.arg0 + (b * D + j) * 8, grad_tmp);
                        const float* xj = cmd.arg1 + (b * D + j) * 8;
                        #pragma unroll
                        for (int c = 0; c < 8; c++) inner += grad_tmp[c] * xj[c];
                    }
                    const float* x = cmd.arg1 + idx * 8;
                    float* out = cmd.arg2 + idx * 8;
                    float corr_scale = inner / (D * (rms * rms * rms + 1e-12f));
                    #pragma unroll
                    for (int c = 0; c < 8; c++) {
                        out[c] = grad_x_hat[c] * inv_rms - x[c] * corr_scale;
                    }
                }
            }
            break;
        }

        case OP_BACKWARD_SOFTMAX: {
            int N = cmd.dim0, D = cmd.dim1;
            for (int row = tid; row < N; row += total_threads) {
                const float* grad_row = cmd.arg0 + row * D;
                const float* softmax_row = cmd.arg1 + row * D;
                float* out_row = cmd.arg2 + row * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) dot += grad_row[d] * softmax_row[d];
                for (int d = 0; d < D; d++) out_row[d] = softmax_row[d] * (grad_row[d] - dot);
            }
            break;
        }

        case OP_BACKWARD_WEIGHTED_SUM: {
            // arg0 = grad_out, arg1 = weights, arg2 = out, arg3 = values
            // dim0 = batch, dim1 = n_heads, dim2 = seq, dim3 = +/-d_head
            int batch = cmd.dim0, n_heads = cmd.dim1, seq = cmd.dim2;
            bool weights_mode = cmd.dim3 < 0;
            int d_head = weights_mode ? -cmd.dim3 : cmd.dim3;
            if (weights_mode) {
                int total = batch * n_heads * seq * seq;
                for (int idx = tid; idx < total; idx += total_threads) {
                    int per_batch = n_heads * seq * seq;
                    int b = idx / per_batch;
                    int rem0 = idx % per_batch;
                    int h = rem0 / (seq * seq);
                    int rem = rem0 % (seq * seq);
                    int q = rem / seq;
                    int k = rem % seq;
                    float sum = 0.0f;
                    for (int d = 0; d < d_head; d++) {
                        int feature = h * d_head + d;
                        const float* g = cmd.arg0 + ((b * seq + q) * (n_heads * d_head) + feature) * 8;
                        const float* v = cmd.arg3 + ((b * seq + k) * (n_heads * d_head) + feature) * 8;
                        #pragma unroll
                        for (int c = 0; c < 8; c++) sum += g[c] * v[c];
                    }
                    cmd.arg2[idx] = sum;
                }
            } else {
                int total = batch * n_heads * seq * d_head;
                for (int idx = tid; idx < total; idx += total_threads) {
                    int per_batch = n_heads * seq * d_head;
                    int b = idx / per_batch;
                    int rem0 = idx % per_batch;
                    int h = rem0 / (seq * d_head);
                    int rem = rem0 % (seq * d_head);
                    int k = rem / d_head;
                    int d = rem % d_head;
                    int feature = h * d_head + d;
                    float* out = cmd.arg2 + ((b * seq + k) * (n_heads * d_head) + feature) * 8;
                    dev_zero(out);
                    for (int q = 0; q < seq; q++) {
                        float w = cmd.arg1[((b * n_heads + h) * seq + q) * seq + k];
                        const float* g = cmd.arg0 + ((b * seq + q) * (n_heads * d_head) + feature) * 8;
                        #pragma unroll
                        for (int c = 0; c < 8; c++) out[c] += w * g[c];
                    }
                }
            }
            break;
        }

        case OP_TIED_LM_HEAD_BWD: {
            // grad_embed[v, d, 0] += sum_n grad_logits[n, v] * hidden[n, d, 0]
            int N = cmd.dim0, D = cmd.dim1, V = cmd.dim2;
            int total = V * D;
            for (int idx = tid; idx < total; idx += total_threads) {
                int v = idx / D;
                int d = idx % D;
                float grad = 0.0f;
                for (int n = 0; n < N; n++) {
                    grad += cmd.arg0[n * V + v] * cmd.arg1[(n * D + d) * 8];
                }
                atomicAdd(cmd.arg3 + (v * D + d) * 8, grad);
            }
            break;
        }

        case OP_BACKWARD_MEMORY_GATE: {
            // arg0 = grad_output, arg1 = context, arg2 = out, arg3 = gate scalar
            // dim0 = batch, dim1 = seq, dim2 = d_model, dim3 = mode
            int batch = cmd.dim0, seq = cmd.dim1, d_model = cmd.dim2, mode = cmd.dim3;
            float gate = cmd.arg3 != NULL ? dev_sigmoid(cmd.arg3[0]) : 0.0f;
            if (mode == 0) {
                int total = batch * seq * d_model * 8;
                for (int idx = tid; idx < total; idx += total_threads) {
                    cmd.arg2[idx] = cmd.arg0[idx];
                }
            } else if (mode == 1) {
                int total = batch * d_model * 8;
                for (int idx = tid; idx < total; idx += total_threads) {
                    int elem = idx / 8;
                    int c = idx % 8;
                    int d = elem % d_model;
                    int b = elem / d_model;
                    float sum = 0.0f;
                    for (int s = 0; s < seq; s++) {
                        sum += cmd.arg0[((b * seq + s) * d_model + d) * 8 + c];
                    }
                    cmd.arg2[idx] = gate * sum;
                }
            } else {
                if (tid == 0) {
                    float raw = 0.0f;
                    for (int b = 0; b < batch; b++) {
                        for (int s = 0; s < seq; s++) {
                            for (int d = 0; d < d_model; d++) {
                                const float* grad = cmd.arg0 + ((b * seq + s) * d_model + d) * 8;
                                const float* ctx = cmd.arg1 + (b * d_model + d) * 8;
                                #pragma unroll
                                for (int c = 0; c < 8; c++) raw += grad[c] * ctx[c];
                            }
                        }
                    }
                    cmd.arg2[0] = raw * gate * (1.0f - gate);
                }
            }
            break;
        }

        case OP_MEMORY_WRITE: {
            // arg0 = x_write (batch, seq, d_model, 8), arg1 = bank_state ((n_slots + 1), 8)
            // dim0 = batch, dim1 = seq, dim2 = d_model, dim3 = n_slots
            int batch = cmd.dim0, seq = cmd.dim1, d_model = cmd.dim2, n_slots = cmd.dim3;
            if (tid == 0 && cmd.arg1 != NULL && n_slots > 0 && d_model <= MEMORY_MAX_DMODEL) {
                float* state = cmd.arg1;
                float* bank = state + 8;
                int n_valid = (int)(state[0] + 0.5f);
                int write_head = (int)(state[1] + 0.5f);
                if (n_valid < 0) n_valid = 0;
                if (n_valid > n_slots) n_valid = n_slots;
                for (int b = 0; b < batch; b++) {
                    float current[MEMORY_MAX_DMODEL][8];
                    float next_level[MEMORY_MAX_DMODEL][8];
                    for (int d = 0; d < d_model; d++) {
                        float accum[8];
                        dev_zero(accum);
                        for (int s = 0; s < seq; s++) {
                            const float* x = cmd.arg0 + ((b * seq + s) * d_model + d) * 8;
                            #pragma unroll
                            for (int c = 0; c < 8; c++) accum[c] += x[c];
                        }
                        float inv_seq = 1.0f / fmaxf((float)seq, 1.0f);
                        float pooled_norm_sq = 0.0f;
                        #pragma unroll
                        for (int c = 0; c < 8; c++) {
                            current[d][c] = accum[c] * inv_seq;
                            pooled_norm_sq += current[d][c] * current[d][c];
                        }
                        float inv_pooled_norm = rsqrtf(pooled_norm_sq + 1e-12f);
                        #pragma unroll
                        for (int c = 0; c < 8; c++) current[d][c] *= inv_pooled_norm;
                    }

                    int current_len = d_model;
                    while (current_len > 1) {
                        int paired = current_len / 2;
                        for (int p = 0; p < paired; p++) {
                            dev_gp(current[2 * p], current[2 * p + 1], next_level[p]);
                            float pair_norm_sq = 0.0f;
                            #pragma unroll
                            for (int c = 0; c < 8; c++) pair_norm_sq += next_level[p][c] * next_level[p][c];
                            if (!(pair_norm_sq > 0.0f) || !isfinite(pair_norm_sq)) {
                                #pragma unroll
                                for (int c = 0; c < 8; c++) {
                                    next_level[p][c] = current[2 * p][c] + current[2 * p + 1][c];
                                }
                                pair_norm_sq = 0.0f;
                                #pragma unroll
                                for (int c = 0; c < 8; c++) pair_norm_sq += next_level[p][c] * next_level[p][c];
                            }
                            float inv_pair_norm = rsqrtf(pair_norm_sq + 1e-12f);
                            #pragma unroll
                            for (int c = 0; c < 8; c++) next_level[p][c] *= inv_pair_norm;
                        }
                        if (current_len & 1) {
                            #pragma unroll
                            for (int c = 0; c < 8; c++) next_level[paired][c] = current[current_len - 1][c];
                            current_len = paired + 1;
                        } else {
                            current_len = paired;
                        }
                        for (int i = 0; i < current_len; i++) {
                            #pragma unroll
                            for (int c = 0; c < 8; c++) current[i][c] = next_level[i][c];
                        }
                    }

                    float summary[8];
                    float norm_sq = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < 8; c++) {
                        summary[c] = current[0][c];
                        norm_sq += summary[c] * summary[c];
                    }
                    float inv_norm = rsqrtf(norm_sq + 1e-12f);
                    #pragma unroll
                    for (int c = 0; c < 8; c++) summary[c] *= inv_norm;

                    float rotor_bv[8];
                    float rotor[8];
                    float contextualized[8];
                    dev_zero(rotor_bv);
                    rotor_bv[4] = (float)write_head * 0.001f;
                    rotor_bv[5] = (float)write_head * 0.01f;
                    rotor_bv[6] = (float)write_head * 0.1f;
                    dev_bivector_exp(rotor_bv, rotor);
                    dev_sandwich(rotor, summary, contextualized);

                    int slot = (n_valid < n_slots) ? n_valid : (write_head % n_slots);
                    float* dst = bank + slot * 8;
                    #pragma unroll
                    for (int c = 0; c < 8; c++) dst[c] = contextualized[c];
                    write_head += 1;
                    if (n_valid < n_slots) n_valid += 1;
                }
                state[0] = (float)n_valid;
                state[1] = (float)write_head;
                if (cmd.arg2 != NULL) cmd.arg2[0] = (float)n_valid;
            }
            break;
        }

        default:
            break;
        }

        // Grid-wide barrier between operations
        grid.sync();
    }
}

// ===== Launch functions (C linkage for Python FFI) =====
extern "C" {

int persistent_engine_launch(
    EngineCommand* d_commands,
    int n_commands,
    float* d_loss_out,
    float* d_grad_norm_out,
    int n_blocks,
    int block_size
) {
    void* args[] = {&d_commands, &n_commands, &d_loss_out, &d_grad_norm_out};
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)persistent_engine_kernel,
        dim3(n_blocks), dim3(block_size),
        args, 0, 0
    );
    if (err != cudaSuccess) {
        return (int)err;
    }
    err = cudaGetLastError();
    return (int)err;
}

int persistent_engine_max_blocks(int block_size) {
    int num_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm, persistent_engine_kernel, block_size, 0);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    return num_blocks_per_sm * num_sms;
}

} // extern "C"
