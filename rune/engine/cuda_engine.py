"""
cuda_engine.py — Main Python interface for the persistent CUDA training engine.

Loads the CUDA engine library, manages command buffers, and runs training
steps entirely on GPU without Python overhead in the inner loop.

Usage:
    engine = CUDAEngine()
    engine.compile(model_config, batch_size=16, seq_len=2048)
    for step in range(total_steps):
        loss = engine.run_step(input_ids, target_ids, lr=get_lr(step))
"""

import ctypes
import os
import glob
import platform
import math
import numpy as np
from typing import Optional, Dict, Any, Tuple

from rune.engine.program import ProgramBuilder, EngineOp, EngineCommand
from rune.engine.memory_pool import MemoryPool
from rune.engine.training_step import TrainingStepBuilder


class CUDAEngine:
    """
    Persistent CUDA training engine.

    Eliminates ALL Python overhead from the training loop by compiling
    the model into a GPU command sequence and executing it with a single
    cooperative kernel launch.
    """

    def __init__(self):
        self._engine_lib: Optional[ctypes.CDLL] = None
        self._cudart: Optional[ctypes.CDLL] = None
        self._pool: Optional[MemoryPool] = None
        self._program: Optional[ProgramBuilder] = None
        self._compiled = False

        # Device info
        self._n_sms = 0
        self._max_blocks = 0
        self._block_size = 256

        # Command buffer on GPU
        self._d_commands: int = 0
        self._n_commands: int = 0

        # Indices of commands that need patching per step (e.g., LR)
        self._adam_cmd_indices: list = []
        self._use_execution_plan = False
        self._plan_buffer_ptrs: Dict[int, int] = {}
        self._plan_buffer_specs: Dict[int, Dict[str, Any]] = {}
        self._plan_param_name_to_buffer: Dict[str, int] = {}
        self._plan_param_id_to_buffer: Dict[int, int] = {}
        self._plan_param_name_to_grad_buffer: Dict[str, int] = {}
        self._plan_param_id_to_grad_buffer: Dict[int, int] = {}
        self._plan_input_buffer_names: Dict[str, int] = {}
        self._plan_output_buffer_names: Dict[str, int] = {}
        self._plan_adam_state: Dict[int, Tuple[int, int]] = {}
        self._plan_grad_buffer_ids: list[int] = []
        self._plan_grad_buffer_views: Dict[int, object] = {}
        self._plan_loss_buffer_id: Optional[int] = None
        self._plan_loss_ptr: int = 0
        self._plan_grad_norm_ptr: int = 0
        self._plan_memory_bank_state_buffer_id: Optional[int] = None
        self._plan_max_grad_norm: Optional[float] = None
        self._step = 0

        self._load_cuda_libs()
        self._query_device()

    # ---- Library loading ----

    def _load_cuda_libs(self):
        """Load the persistent engine shared library and CUDA runtime."""
        base = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'mjolnir'
        ))
        is_windows = platform.system() == 'Windows'

        # Search for engine library
        if is_windows:
            lib_paths = [
                os.path.join(base, 'build_cuda', 'Release', 'mjolnir_engine.dll'),
                os.path.join(base, 'build_cuda', 'Debug', 'mjolnir_engine.dll'),
                os.path.join(base, 'build_cuda', 'mjolnir_engine.dll'),
                os.path.join(base, 'build_cuda', 'Release', 'mjolnir_cuda.dll'),
                os.path.join(base, 'build_cuda', 'Debug', 'mjolnir_cuda.dll'),
                os.path.join(base, 'build_cuda', 'mjolnir_cuda.dll'),
            ]
        else:
            lib_paths = [
                os.path.join(base, 'build_cuda', 'libmjolnir_engine.so'),
                os.path.join(base, 'build_cuda', 'libmjolnir_cuda.so'),
                os.path.join(base, 'build', 'libmjolnir_cuda.so'),
            ]

        for path in lib_paths:
            if os.path.exists(path):
                try:
                    self._engine_lib = ctypes.CDLL(path)
                    break
                except OSError:
                    continue

        if self._engine_lib is None:
            raise RuntimeError(
                "Could not load mjolnir_cuda library. "
                "Build with: cmake -DBUILD_CUDA=ON -DBUILD_PERSISTENT_ENGINE=ON"
            )

        # Set up engine function signatures
        self._engine_lib.persistent_engine_launch.restype = ctypes.c_int
        self._engine_lib.persistent_engine_launch.argtypes = [
            ctypes.c_void_p,  # EngineCommand* d_commands
            ctypes.c_int,     # int n_commands
            ctypes.c_void_p,  # float* d_loss_out
            ctypes.c_void_p,  # float* d_grad_norm_out
            ctypes.c_int,     # int n_blocks
            ctypes.c_int,     # int block_size
        ]

        self._engine_lib.persistent_engine_max_blocks.restype = ctypes.c_int
        self._engine_lib.persistent_engine_max_blocks.argtypes = [ctypes.c_int]

        # Load CUDA runtime
        self._load_cudart()

    def _load_cudart(self):
        """Load CUDA runtime library."""
        is_windows = platform.system() == 'Windows'
        search = []

        if is_windows:
            cuda_path = os.environ.get(
                'CUDA_PATH',
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
            )
            search += glob.glob(os.path.join(cuda_path, 'bin', 'cudart64_*.dll'))
            search += ['cudart64_131.dll', 'cudart64_13.dll', 'cudart64_12.dll']
        else:
            cuda_homes = [
                os.environ.get('CUDA_HOME', '/usr/local/cuda'),
                '/usr/local/cuda',
            ]
            for ch in cuda_homes:
                search += glob.glob(os.path.join(ch, 'lib64', 'libcudart.so*'))
            search += ['libcudart.so']

        for path in search:
            try:
                self._cudart = ctypes.CDLL(path)
                self._setup_cudart()
                return
            except OSError:
                continue

        raise RuntimeError("Could not load CUDA runtime library")

    def _setup_cudart(self):
        """Set up CUDA runtime signatures."""
        self._cudart.cudaMalloc.restype = ctypes.c_int
        self._cudart.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t
        ]
        self._cudart.cudaFree.restype = ctypes.c_int
        self._cudart.cudaFree.argtypes = [ctypes.c_void_p]
        self._cudart.cudaMemcpy.restype = ctypes.c_int
        self._cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        self._cudart.cudaMemset.restype = ctypes.c_int
        self._cudart.cudaMemset.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        ]
        self._cudart.cudaDeviceSynchronize.restype = ctypes.c_int
        self._cudart.cudaDeviceSynchronize.argtypes = []
        self._cudart.cudaDeviceGetAttribute.restype = ctypes.c_int
        self._cudart.cudaDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
        ]

    def _query_device(self):
        """Query GPU device properties."""
        # cudaDevAttrMultiProcessorCount = 16
        n_sms = ctypes.c_int(0)
        self._cudart.cudaDeviceGetAttribute(ctypes.byref(n_sms), 16, 0)
        self._n_sms = n_sms.value

        self._max_blocks = self._engine_lib.persistent_engine_max_blocks(
            self._block_size
        )

    def _malloc_bytes(self, size_bytes: int, *, zero: bool = True) -> int:
        ptr = ctypes.c_void_p()
        err = self._cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size_bytes))
        self._check_cuda_error(err, f"cudaMalloc({size_bytes})")
        if zero and size_bytes:
            err = self._cudart.cudaMemset(ptr, 0, ctypes.c_size_t(size_bytes))
            self._check_cuda_error(err, f"cudaMemset({size_bytes})")
        return ptr.value

    def _upload_ptr(self, ptr: int, data: np.ndarray):
        data = np.ascontiguousarray(data)
        err = self._cudart.cudaMemcpy(
            ctypes.c_void_p(ptr),
            data.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(data.nbytes),
            1,
        )
        self._check_cuda_error(err, "cudaMemcpy H2D upload")

    def _download_ptr(self, ptr: int, shape, dtype) -> np.ndarray:
        arr = np.empty(shape, dtype=dtype)
        err = self._cudart.cudaMemcpy(
            arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(arr.nbytes),
            2,
        )
        self._check_cuda_error(err, "cudaMemcpy D2H download")
        return arr

    @staticmethod
    def _spec_array_shape(spec: Dict[str, Any]):
        shape = tuple(spec.get('shape', ()))
        storage = int(spec.get('storage_components', 1))
        if storage == 1:
            return shape
        return shape + (storage,)

    # ---- Compilation ----

    def compile(self, model_config: Dict[str, Any],
                batch_size: int, seq_len: int,
                params: Optional[Dict[str, np.ndarray]] = None):
        """
        Compile a model into a GPU execution program.

        Args:
            model_config: Model configuration dict with keys:
                vocab_size, d_model, n_heads, n_layers, d_ff
            batch_size: Training batch size.
            seq_len: Sequence length.
            params: Optional dict of parameter name -> numpy array.
                    If None, parameters are initialized randomly.
        """
        self.free()
        self._use_execution_plan = False
        self._step = 0

        vocab_size = model_config['vocab_size']
        d_model = model_config['d_model']
        n_heads = model_config['n_heads']
        n_layers = model_config['n_layers']
        d_ff = model_config.get('d_ff', 4 * d_model)
        d_head = d_model // n_heads

        # 1. Allocate memory pool
        self._pool = MemoryPool(self._cudart)
        builder = TrainingStepBuilder(
            pool=self._pool,
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            batch_size=batch_size,
            seq_len=seq_len,
            use_positional_encoding=bool(model_config.get('use_positional_encoding', False)),
        )

        # 2. Allocate all buffers (params + activations + gradients)
        builder.allocate_buffers()
        total_bytes = self._pool.finalize()
        print(f"[CUDAEngine] Memory pool: {total_bytes / 1024**2:.1f} MB "
              f"({len(self._pool.names())} buffers)")

        # 3. Upload parameters
        if params is not None:
            for name, arr in params.items():
                if self._pool.has(name):
                    self._pool.upload(name, arr)
        else:
            builder.init_params_random()

        for name in self._pool.names():
            if name.startswith('grad.') or name.startswith('adam_') or name in ('loss_out', 'grad_norm_out'):
                self._pool.zero_buffer(name)

        # 4. Build forward + loss + backward + update program
        self._program = ProgramBuilder()
        self._adam_cmd_indices = builder.build_program(self._program)

        self._program.done()
        print(f"[CUDAEngine] Program: {len(self._program)} commands")
        self._n_commands = len(self._program)

        # 5. Upload command buffer to GPU
        cmd_array, n = self._program.build()
        cmd_bytes = ctypes.sizeof(cmd_array)
        ptr = ctypes.c_void_p()
        err = self._cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(cmd_bytes))
        if err != 0:
            raise RuntimeError(f"cudaMalloc for commands failed: {err}")
        self._d_commands = ptr.value

        err = self._cudart.cudaMemcpy(
            ctypes.c_void_p(self._d_commands),
            ctypes.byref(cmd_array),
            ctypes.c_size_t(cmd_bytes),
            1,  # H2D
        )
        self._check_cuda_error(err, "cudaMemcpy for command buffer upload")

        self._compiled = True
        self._builder = builder
        print(f"[CUDAEngine] Compiled. n_blocks={self._max_blocks}, "
              f"block_size={self._block_size}, SMs={self._n_sms}")

    def compile_execution_plan(
        self,
        execution_plan,
        params_by_id: Dict[int, np.ndarray],
        *,
        max_grad_norm: Optional[float] = None,
    ):
        """
        Compile a compiler-generated execution plan into the persistent engine.

        Args:
            execution_plan: rune.compiler.codegen.ExecutionPlan
            params_by_id: mapping of `id(param_array)` -> numpy array payload
        """
        from rune.backend import to_numpy

        self.free()
        self._use_execution_plan = True
        self._step = 0
        self._program = ProgramBuilder()
        self._plan_buffer_ptrs = {}
        self._plan_buffer_specs = dict(execution_plan.buffer_specs)
        self._plan_input_buffer_names = dict(execution_plan.input_buffer_names)
        self._plan_output_buffer_names = dict(execution_plan.output_buffer_names)
        self._plan_param_id_to_buffer = dict(execution_plan.param_buffer_map)
        self._plan_param_name_to_buffer = {
            execution_plan.param_name_map[pid]: buf_id
            for pid, buf_id in execution_plan.param_buffer_map.items()
            if pid in execution_plan.param_name_map
        }
        self._plan_param_id_to_grad_buffer = dict(execution_plan.param_grad_buffer_map)
        self._plan_param_name_to_grad_buffer = {
            execution_plan.param_name_map[pid]: buf_id
            for pid, buf_id in execution_plan.param_grad_buffer_map.items()
            if pid in execution_plan.param_name_map
        }
        self._plan_adam_state = {}
        self._plan_grad_buffer_ids = []
        self._plan_grad_buffer_views = {}
        self._adam_cmd_indices = []
        self._plan_loss_ptr = self._malloc_bytes(np.dtype(np.float32).itemsize, zero=True)
        self._plan_grad_norm_ptr = self._malloc_bytes(np.dtype(np.float32).itemsize, zero=True)
        self._plan_memory_bank_state_buffer_id = None
        self._plan_max_grad_norm = (
            float(max_grad_norm) if max_grad_norm is not None and max_grad_norm > 0 else None
        )

        # Allocate runtime buffers described by the execution plan.
        for buf_id, spec in sorted(execution_plan.buffer_specs.items()):
            size_bytes = int(spec.get('size_bytes', 0))
            self._plan_buffer_ptrs[buf_id] = self._malloc_bytes(size_bytes, zero=True)
            if spec.get('name') == 'memory.bank_state':
                self._plan_memory_bank_state_buffer_id = buf_id

        # Upload parameter buffers.
        for param_id, buf_id in execution_plan.param_buffer_map.items():
            if param_id not in params_by_id:
                raise ValueError(f"Missing parameter payload for param_id={param_id}")
            target_dtype = np.dtype(execution_plan.buffer_specs[buf_id].get('dtype', np.float32))
            arr = np.ascontiguousarray(to_numpy(params_by_id[param_id]), dtype=target_dtype)
            try:
                self._upload_ptr(self._plan_buffer_ptrs[buf_id], arr)
            except Exception as exc:
                param_name = execution_plan.param_name_map.get(param_id, str(param_id))
                size_bytes = int(execution_plan.buffer_specs[buf_id]['size_bytes'])
                raise RuntimeError(
                    f"Failed to upload compiled parameter '{param_name}' into buffer {buf_id} "
                    f"(shape={arr.shape}, dtype={arr.dtype}, nbytes={arr.nbytes}, "
                    f"buffer_bytes={size_bytes})"
                ) from exc
            size_bytes = int(execution_plan.buffer_specs[buf_id]['size_bytes'])
            m_ptr = self._malloc_bytes(size_bytes, zero=True)
            v_ptr = self._malloc_bytes(size_bytes, zero=True)
            self._plan_adam_state[buf_id] = (m_ptr, v_ptr)

        # Upload static constant buffers.
        for buf_id, payload in execution_plan.constant_buffer_data.items():
            target_dtype = np.dtype(execution_plan.buffer_specs[buf_id].get('dtype', np.float32))
            arr = np.ascontiguousarray(payload, dtype=target_dtype)
            try:
                self._upload_ptr(self._plan_buffer_ptrs[buf_id], arr)
            except Exception as exc:
                size_bytes = int(execution_plan.buffer_specs[buf_id]['size_bytes'])
                raise RuntimeError(
                    f"Failed to upload compiled constant buffer {buf_id} "
                    f"(shape={arr.shape}, dtype={arr.dtype}, nbytes={arr.nbytes}, "
                    f"buffer_bytes={size_bytes})"
                ) from exc

        self.load_plan(execution_plan)
        self._program.done()
        print(f"[CUDAEngine] Program: {len(self._program)} commands")
        self._n_commands = len(self._program)

        cmd_array, _ = self._program.build()
        cmd_bytes = ctypes.sizeof(cmd_array)
        self._d_commands = self._malloc_bytes(cmd_bytes, zero=False)
        err = self._cudart.cudaMemcpy(
            ctypes.c_void_p(self._d_commands),
            ctypes.byref(cmd_array),
            ctypes.c_size_t(cmd_bytes),
            1,
        )
        self._check_cuda_error(err, "cudaMemcpy for execution-plan command upload")

        self._plan_loss_buffer_id = self._plan_output_buffer_names.get(
            'output_loss',
            execution_plan.output_buffers[0] if execution_plan.output_buffers else None,
        )

        self._compiled = True
        print(f"[CUDAEngine] Compiled from execution plan. n_blocks={self._max_blocks}, "
              f"block_size={self._block_size}, SMs={self._n_sms}")

    # ---- Training step ----

    def run_step(self, input_ids: np.ndarray, target_ids: np.ndarray,
                 lr: float = 1e-4) -> float:
        """
        Execute one training step entirely on GPU.

        Args:
            input_ids: (batch_size, seq_len) int32
            target_ids: (batch_size * seq_len,) int32
            lr: Learning rate for this step.

        Returns:
            loss: Scalar float loss value.
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled. Call compile() first.")

        self._step += 1

        if self._use_execution_plan:
            token_buf = self._plan_input_buffer_names.get('token_ids')
            target_buf = self._plan_input_buffer_names.get('target_ids')
            if token_buf is None or target_buf is None:
                raise RuntimeError("Execution-plan engine missing token_ids/target_ids buffers")

            self._upload_ptr(
                self._plan_buffer_ptrs[token_buf],
                input_ids.reshape(-1).astype(np.int32),
            )
            self._upload_ptr(
                self._plan_buffer_ptrs[target_buf],
                target_ids.reshape(-1).astype(np.int32),
            )

            if self._plan_loss_ptr:
                err = self._cudart.cudaMemset(
                    ctypes.c_void_p(self._plan_loss_ptr),
                    0,
                    ctypes.c_size_t(np.dtype(np.float32).itemsize),
                )
                self._check_cuda_error(err, "cudaMemset loss buffer")
            if self._plan_grad_norm_ptr:
                err = self._cudart.cudaMemset(
                    ctypes.c_void_p(self._plan_grad_norm_ptr),
                    0,
                    ctypes.c_size_t(np.dtype(np.float32).itemsize),
                )
                self._check_cuda_error(err, "cudaMemset grad-norm buffer")

            self._patch_lr(lr, step=self._step)

            loss_ptr = self._plan_loss_ptr if self._plan_loss_ptr else 0
            grad_norm_ptr = self._plan_grad_norm_ptr if self._plan_grad_norm_ptr else 0
            launch_err = self._engine_lib.persistent_engine_launch(
                ctypes.c_void_p(self._d_commands),
                ctypes.c_int(self._n_commands),
                ctypes.c_void_p(loss_ptr),
                ctypes.c_void_p(grad_norm_ptr),
                ctypes.c_int(self._max_blocks),
                ctypes.c_int(self._block_size),
            )
            self._check_cuda_error(launch_err, "persistent_engine_launch")
            sync_err = self._cudart.cudaDeviceSynchronize()
            self._check_cuda_error(sync_err, "cudaDeviceSynchronize after engine launch")
            if not self._plan_loss_ptr:
                return 0.0
            return float(self._download_ptr(self._plan_loss_ptr, (1,), np.dtype(np.float32))[0])

        # 1. Upload input_ids and target_ids
        self._pool.upload_int('input_ids', input_ids.reshape(-1).astype(np.int32))
        self._pool.upload_int('target_ids', target_ids.reshape(-1).astype(np.int32))

        # 2. Zero loss accumulator
        self._pool.zero_buffer('loss_out')
        self._pool.zero_buffer('grad_norm_out')

        # 3. Patch learning rate in adam commands
        self._patch_lr(lr, step=self._step)

        # 4. Launch persistent kernel
        launch_err = self._engine_lib.persistent_engine_launch(
            ctypes.c_void_p(self._d_commands),
            ctypes.c_int(self._n_commands),
            ctypes.c_void_p(self._pool.ptr('loss_out')),
            ctypes.c_void_p(self._pool.ptr('grad_norm_out')),
            ctypes.c_int(self._max_blocks),
            ctypes.c_int(self._block_size),
        )
        self._check_cuda_error(launch_err, "persistent_engine_launch")

        # 5. Synchronize and check for CUDA errors
        sync_err = self._cudart.cudaDeviceSynchronize()
        self._check_cuda_error(sync_err, "cudaDeviceSynchronize after engine launch")

        loss = self._pool.download_scalar('loss_out')
        return loss

    def _patch_lr(self, lr: float, *, step: Optional[int] = None):
        """Patch learning rate in the ADAM commands on GPU."""
        if not self._adam_cmd_indices:
            return

        # Update scalar0 (lr) in each adam command
        # EngineCommand.scalar0 offset: need to compute field offset
        # Fields: opcode(4) + 4 pointers(8 each = 32) + 4 ints(4 each = 16)
        # scalar0 is at offset 4 + 32 + 16 = 52 on 64-bit
        scalar0_offset = EngineCommand.scalar0.offset

        for cmd_idx in self._adam_cmd_indices:
            cmd_gpu_addr = self._d_commands + cmd_idx * ctypes.sizeof(EngineCommand)
            lr_addr = cmd_gpu_addr + scalar0_offset
            lr_buf = np.array([lr], dtype=np.float32)
            err = self._cudart.cudaMemcpy(
                ctypes.c_void_p(lr_addr),
                lr_buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(4),
                1,  # H2D
            )
            self._check_cuda_error(err, "cudaMemcpy for LR patch")
            if step is not None:
                step_addr = cmd_gpu_addr + EngineCommand.dim1.offset
                step_buf = np.array([step], dtype=np.int32)
                err = self._cudart.cudaMemcpy(
                    ctypes.c_void_p(step_addr),
                    step_buf.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_size_t(step_buf.nbytes),
                    1,
                )
                self._check_cuda_error(err, "cudaMemcpy for Adam step patch")

    def load_plan(self, execution_plan) -> None:
        """
        Load a compiler-generated ExecutionPlan and convert it into engine commands.

        This bridges the compiler's output (ExecutionPlan with EngineCommands)
        into the runtime engine's ProgramBuilder format. The compiler's ExecutionPlan
        contains high-level commands that map to the engine's low-level opcodes.

        Args:
            execution_plan: An ExecutionPlan from the codegen stage.
        """
        from rune.compiler.codegen import EngineOp as CompilerEngineOp
        if self._program is None:
            self._program = ProgramBuilder()

        compiler_to_runtime = {
            CompilerEngineOp.LOAD_EMBEDDING: EngineOp.EMBED_LOOKUP,
            CompilerEngineOp.COPY_BUFFER: EngineOp.COPY,
            CompilerEngineOp.ZERO_BUFFER: EngineOp.ZERO,
            CompilerEngineOp.GEOMETRIC_PRODUCT: EngineOp.BATCH_GP,
            CompilerEngineOp.REVERSE: EngineOp.BATCH_REVERSE,
            CompilerEngineOp.SANDWICH: EngineOp.BATCH_SANDWICH,
            CompilerEngineOp.BIVECTOR_EXP: EngineOp.BATCH_BVEXP,
            CompilerEngineOp.GRADE_PROJECT: EngineOp.BATCH_GRADE_PROJ,
            CompilerEngineOp.ADD: EngineOp.BATCH_ADD,
            CompilerEngineOp.SCALE: EngineOp.BATCH_SCALE,
            CompilerEngineOp.CLIFFORD_LINEAR: EngineOp.LINEAR_FWD,
            CompilerEngineOp.ATTENTION_SCORE: EngineOp.ATTN_SCORE,
            CompilerEngineOp.SOFTMAX: EngineOp.SOFTMAX,
            CompilerEngineOp.WEIGHTED_SUM: EngineOp.WEIGHTED_SUM,
            CompilerEngineOp.MEAN_POOL_SEQ: EngineOp.MEAN_POOL_SEQ,
            CompilerEngineOp.LAYER_NORM: EngineOp.BATCH_NORM_SCALE,
            CompilerEngineOp.GELU: EngineOp.BATCH_GELU,
            CompilerEngineOp.MATMUL: EngineOp.MATMUL_SCALAR,
            CompilerEngineOp.CROSS_ENTROPY: EngineOp.CE_LOSS_FWD,
            CompilerEngineOp.BACKWARD_CE: EngineOp.BACKWARD_CE,
            CompilerEngineOp.BACKWARD_LINEAR: EngineOp.BACKWARD_LINEAR,
            CompilerEngineOp.BACKWARD_GP: EngineOp.BACKWARD_GP,
            CompilerEngineOp.BACKWARD_NORM: EngineOp.BACKWARD_NORM,
            CompilerEngineOp.BACKWARD_GELU: EngineOp.BACKWARD_GELU,
            CompilerEngineOp.BACKWARD_EMBED: EngineOp.BACKWARD_EMBED,
            CompilerEngineOp.BACKWARD_ADD: EngineOp.BACKWARD_ADD,
            CompilerEngineOp.BACKWARD_MATMUL: EngineOp.BACKWARD_MATMUL,
            CompilerEngineOp.BACKWARD_GRADE_PROJECT: EngineOp.BACKWARD_GRADE_PROJECT,
            CompilerEngineOp.BACKWARD_BVEXP: EngineOp.BACKWARD_BVEXP,
            CompilerEngineOp.BACKWARD_SOFTMAX: EngineOp.BACKWARD_SOFTMAX,
            CompilerEngineOp.BACKWARD_WEIGHTED_SUM: EngineOp.BACKWARD_WEIGHTED_SUM,
            CompilerEngineOp.BACKWARD_COPY: EngineOp.BACKWARD_COPY,
            CompilerEngineOp.BACKWARD_MEMORY_GATE: EngineOp.BACKWARD_MEMORY_GATE,
            CompilerEngineOp.ADAM_FULL: EngineOp.ADAM_FULL,
            CompilerEngineOp.MEMORY_READ: EngineOp.MEMORY_READ,
            CompilerEngineOp.MEMORY_WRITE: EngineOp.MEMORY_WRITE,
            CompilerEngineOp.MEMORY_GATE: EngineOp.MEMORY_GATE,
            CompilerEngineOp.BARRIER: EngineOp.BARRIER,
        }

        clip_grad_buffers: list[int] = []
        if self._plan_max_grad_norm:
            seen_grad_buffers = set()
            for cmd in execution_plan.commands:
                if cmd.op != CompilerEngineOp.ADAM_FULL:
                    continue
                if len(cmd.input_buffers) < 2:
                    continue
                grad_buf = cmd.input_buffers[1]
                if grad_buf in seen_grad_buffers:
                    continue
                seen_grad_buffers.add(grad_buf)
                clip_grad_buffers.append(grad_buf)

        inserted_grad_clip = False

        for cmd in execution_plan.commands:
            if cmd.op == CompilerEngineOp.STORE_OUTPUT:
                continue
            if cmd.op == CompilerEngineOp.ZERO_BUFFER and cmd.name in self._plan_input_buffer_names:
                continue

            if (
                not inserted_grad_clip
                and clip_grad_buffers
                and cmd.op == CompilerEngineOp.ADAM_FULL
            ):
                self._insert_execution_plan_grad_clip(clip_grad_buffers)
                inserted_grad_clip = True

            if cmd.op == CompilerEngineOp.COPY_BUFFER:
                if len(cmd.input_buffers) == 1:
                    src_ptr = self._plan_buffer_ptrs[cmd.input_buffers[0]]
                    out_ptr = self._plan_buffer_ptrs[cmd.output_buffer]
                    spec = self._plan_buffer_specs[cmd.output_buffer]
                    self._program.add(
                        opcode=EngineOp.COPY,
                        arg0=src_ptr,
                        arg2=out_ptr,
                        dim0=int(spec['size_bytes'] // 4),
                    )
                    continue

                out_spec = self._plan_buffer_specs[cmd.output_buffer]
                out_shape = tuple(out_spec.get('shape', ()))
                n_tokens = int(np.prod(out_shape[:-1])) if len(out_shape) > 1 else 1
                d_total = int(out_shape[-1]) if out_shape else 1
                offset = 0
                for input_buf in cmd.input_buffers:
                    in_spec = self._plan_buffer_specs[input_buf]
                    in_shape = tuple(in_spec.get('shape', ()))
                    d_slice = int(in_shape[-1]) if in_shape else 1
                    self._program.add(
                        opcode=EngineOp.COPY,
                        arg0=self._plan_buffer_ptrs[input_buf],
                        arg2=self._plan_buffer_ptrs[cmd.output_buffer],
                        dim0=n_tokens,
                        dim1=d_slice,
                        dim2=d_total,
                        dim3=offset,
                        scalar0=1.0,
                    )
                    offset += d_slice
                continue

            if cmd.op == CompilerEngineOp.BACKWARD_COPY:
                src_spec = self._plan_buffer_specs[cmd.input_buffers[0]]
                out_spec = self._plan_buffer_specs[cmd.output_buffer]
                src_shape = tuple(src_spec.get('shape', ()))
                out_shape = tuple(out_spec.get('shape', ()))
                n_tokens = int(np.prod(src_shape[:-1])) if len(src_shape) > 1 else 1
                d_total = int(src_shape[-1]) if src_shape else 1
                d_slice = int(out_shape[-1]) if out_shape else 1
                offset = int(cmd.params.get('input_index', 0)) * d_slice
                self._program.add(
                    opcode=EngineOp.COPY,
                    arg0=self._plan_buffer_ptrs[cmd.input_buffers[0]],
                    arg2=self._plan_buffer_ptrs[cmd.output_buffer],
                    dim0=n_tokens,
                    dim1=d_slice,
                    dim2=d_total,
                    dim3=offset,
                    scalar0=0.0,
                )
                continue

            runtime_op = compiler_to_runtime.get(cmd.op)
            if runtime_op is None:
                raise ValueError(f"Unsupported execution-plan op for runtime: {cmd.op}")

            ptrs = [self._plan_buffer_ptrs[buf_id] for buf_id in cmd.input_buffers]
            out_ptr = self._plan_buffer_ptrs.get(cmd.output_buffer, 0)
            arg0 = arg1 = arg2 = arg3 = 0
            params = cmd.params if isinstance(cmd.params, dict) else {}

            if runtime_op == EngineOp.EMBED_LOOKUP:
                # Compiler input order is [token_ids, embed.weight]; runtime expects table then ids.
                arg0 = ptrs[1]
                arg1 = ptrs[0]
                arg2 = out_ptr
            elif runtime_op == EngineOp.LINEAR_FWD:
                arg0 = ptrs[0]
                arg1 = ptrs[1]
                arg2 = out_ptr
                arg3 = ptrs[2] if len(ptrs) > 2 else 0
            elif runtime_op == EngineOp.MEAN_POOL_SEQ:
                arg0 = ptrs[0] if len(ptrs) > 0 else 0
                arg2 = out_ptr
            elif runtime_op == EngineOp.MEMORY_READ:
                arg0 = ptrs[0] if len(ptrs) > 0 else 0
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                arg2 = out_ptr
            elif runtime_op == EngineOp.MEMORY_GATE:
                arg0 = ptrs[0] if len(ptrs) > 0 else 0
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                arg2 = out_ptr
                arg3 = ptrs[2] if len(ptrs) > 2 else 0
            elif runtime_op == EngineOp.BACKWARD_MEMORY_GATE:
                arg0 = ptrs[0] if len(ptrs) > 0 else 0
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                arg2 = out_ptr
                arg3 = ptrs[2] if len(ptrs) > 2 else 0
            elif runtime_op == EngineOp.MEMORY_WRITE:
                arg0 = ptrs[0] if len(ptrs) > 0 else 0
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                arg2 = out_ptr
                arg3 = ptrs[2] if len(ptrs) > 2 else 0
            elif runtime_op in (EngineOp.ATTN_SCORE, EngineOp.WEIGHTED_SUM, EngineOp.BATCH_GP,
                                EngineOp.BATCH_REVERSE, EngineOp.BATCH_SANDWICH, EngineOp.BATCH_BVEXP,
                                EngineOp.BATCH_ADD, EngineOp.BATCH_SCALE, EngineOp.BATCH_GRADE_PROJ,
                                EngineOp.BATCH_GELU, EngineOp.BATCH_NORM_SCALE, EngineOp.MATMUL_SCALAR,
                                EngineOp.CE_LOSS_FWD, EngineOp.BACKWARD_CE, EngineOp.BACKWARD_LINEAR,
                                EngineOp.BACKWARD_GP, EngineOp.BACKWARD_NORM, EngineOp.BACKWARD_GELU,
                                EngineOp.BACKWARD_EMBED, EngineOp.BACKWARD_ADD, EngineOp.BACKWARD_MATMUL,
                                EngineOp.BACKWARD_GRADE_PROJECT, EngineOp.BACKWARD_BVEXP,
                                EngineOp.BACKWARD_SOFTMAX, EngineOp.BACKWARD_WEIGHTED_SUM,
                                EngineOp.BACKWARD_COPY):
                arg0 = ptrs[0] if len(ptrs) > 0 else 0
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                arg2 = out_ptr
                arg3 = ptrs[2] if len(ptrs) > 2 else 0
                if runtime_op == EngineOp.BATCH_NORM_SCALE and len(ptrs) > 2:
                    arg3 = ptrs[2]
            elif runtime_op == EngineOp.COPY:
                arg0 = ptrs[0]
                arg2 = out_ptr
            elif runtime_op == EngineOp.ZERO:
                arg2 = out_ptr
            elif runtime_op == EngineOp.ADAM_FULL:
                param_buf = cmd.input_buffers[0]
                grad_buf = cmd.input_buffers[1]
                m_ptr, v_ptr = self._plan_adam_state[param_buf]
                arg0 = self._plan_buffer_ptrs[param_buf]
                arg1 = self._plan_buffer_ptrs[grad_buf]
                arg2 = m_ptr
                arg3 = v_ptr
                if grad_buf not in self._plan_grad_buffer_ids:
                    self._plan_grad_buffer_ids.append(grad_buf)
            else:
                arg0 = ptrs[0] if len(ptrs) > 0 else 0
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                arg2 = out_ptr
                arg3 = ptrs[2] if len(ptrs) > 2 else 0

            dim0 = int(params.get('n', params.get('n_floats', params.get('n_tokens', params.get('batch_tokens', params.get('n_rows', params.get('m', params.get('batch', 0))))))))
            dim1 = int(params.get('d_model', params.get('d_in', params.get('row_len', params.get('k', params.get('step', params.get('n_heads', 0)))))))
            dim2 = int(params.get('d_out', params.get('n', params.get('seq', 0))))
            dim3 = int(params.get('d_head', 0))

            if runtime_op == EngineOp.BATCH_GRADE_PROJ:
                dim1 = int(params.get('target_grade', 0))
                dim2 = int(params.get('output_components', 8))
            elif runtime_op == EngineOp.BATCH_BVEXP:
                dim0 = int(params.get('n', 0))
                dim1 = int(params.get('input_components', 8))
            elif runtime_op == EngineOp.BATCH_ADD:
                dim0 = int(params.get('n', 0))
                dim1 = int(params.get('components', 8))
            elif runtime_op == EngineOp.ATTN_SCORE:
                dim0 = int(params.get('batch', 0))
                dim1 = int(params.get('n_heads', 1))
                dim2 = int(params.get('seq', 0))
                dim3 = int(params.get('d_head', 0))
            elif runtime_op == EngineOp.WEIGHTED_SUM:
                dim0 = int(params.get('batch', 0))
                dim1 = int(params.get('n_heads', 1))
                dim2 = int(params.get('seq', 0))
                dim3 = int(params.get('d_head', 0))
            elif runtime_op == EngineOp.MEAN_POOL_SEQ:
                dim0 = int(params.get('batch', 0))
                dim1 = int(params.get('seq', 0))
                dim2 = int(params.get('d_model', 0))
            elif runtime_op == EngineOp.MEMORY_READ:
                dim0 = int(params.get('batch', 0))
                dim1 = int(params.get('top_k', 1))
                dim2 = int(params.get('n_slots', 0))
            elif runtime_op == EngineOp.MEMORY_GATE:
                dim0 = int(params.get('batch', 0))
                dim1 = int(params.get('seq', 0))
                dim2 = int(params.get('d_model', 0))
            elif runtime_op == EngineOp.BACKWARD_MEMORY_GATE:
                dim0 = int(params.get('batch', 0))
                dim1 = int(params.get('seq', 0))
                dim2 = int(params.get('d_model', 0))
                mode_map = {'input': 0, 'context': 1, 'gate': 2}
                dim3 = mode_map.get(params.get('mode', 'input'), 0)
            elif runtime_op == EngineOp.MEMORY_WRITE:
                dim0 = int(params.get('batch', 0))
                dim1 = int(params.get('seq', 0))
                dim2 = int(params.get('d_model', 0))
                dim3 = int(params.get('n_slots', 0))
            elif runtime_op == EngineOp.BATCH_NORM_SCALE:
                dim0 = int(params.get('batch_tokens', 0))
                dim1 = int(params.get('d_model', 0))
            elif runtime_op == EngineOp.MATMUL_SCALAR:
                dim0 = int(params.get('m', 0))
                dim1 = int(params.get('k', 0))
                dim2 = int(params.get('n', 0))
                mode_map = {'input': 0, 'weight': 2}
                dim3 = mode_map.get(params.get('mode', 0), params.get('mode', 0))
            elif runtime_op == EngineOp.CE_LOSS_FWD:
                dim0 = int(params.get('n', 0))
                dim1 = int(params.get('vocab', 0))
                arg2 = 0
            elif runtime_op == EngineOp.BACKWARD_CE:
                dim0 = int(params.get('n', 0))
                dim1 = int(params.get('vocab', 0))
                arg2 = out_ptr
            elif runtime_op == EngineOp.BACKWARD_LINEAR:
                mode = params.get('mode', 'input')
                arg0 = ptrs[0]
                if mode == 'input':
                    arg1 = ptrs[1]
                    arg3 = 0
                elif mode == 'weight':
                    arg1 = ptrs[2]
                    arg3 = 0
                else:
                    arg1 = 0
                    arg3 = 0
                dim0 = int(params.get('batch_tokens', 0))
                dim1 = int(params.get('d_in', 0))
                dim2 = int(params.get('d_out', 0))
                mode_map = {'input': 0, 'weight': 1, 'bias': 2}
                dim3 = mode_map.get(mode, 0)
            elif runtime_op == EngineOp.BACKWARD_GP:
                arg0 = ptrs[0]
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                arg3 = ptrs[2] if len(ptrs) > 2 else 0
                if 'd_head' in params:
                    dim0 = int(params.get('batch', 0))
                    dim1 = int(params.get('seq', 0))
                    dim2 = int(params.get('d_head', 0))
                    mode_map = {'left': 10, 'right': 11}
                    dim3 = mode_map.get(params.get('mode', 'left'), 10)
                else:
                    dim0 = int(params.get('n', 0))
                    mode_map = {'left': 0, 'right': 1}
                    dim1 = mode_map.get(params.get('mode', 'left'), 0)
            elif runtime_op == EngineOp.BACKWARD_NORM:
                mode = params.get('mode', 'input')
                arg0 = ptrs[0]
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                arg3 = ptrs[2] if len(ptrs) > 2 else 0
                dim0 = int(params.get('batch_tokens', 0))
                dim1 = int(params.get('d_model', 0))
                mode_map = {'input': 0, 'gamma': 1, 'beta': 2}
                dim2 = mode_map.get(mode, 0)
            elif runtime_op == EngineOp.BACKWARD_GELU:
                dim0 = int(params.get('n', 0))
            elif runtime_op == EngineOp.BACKWARD_EMBED:
                arg0 = ptrs[0]
                arg1 = ptrs[1]
                dim0 = int(params.get('n_tokens', 0))
                dim1 = int(params.get('d_model', 0))
            elif runtime_op == EngineOp.BACKWARD_MATMUL:
                arg0 = ptrs[0] if len(ptrs) > 0 else 0
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                arg2 = out_ptr
                arg3 = ptrs[2] if len(ptrs) > 2 else 0
                dim0 = int(params.get('m', 0))
                dim1 = int(params.get('k', 0))
                dim2 = int(params.get('n', 0))
                mode_map = {'input': 0, 'weight': 2}
                dim3 = mode_map.get(params.get('mode', 'input'), 0)
            elif runtime_op == EngineOp.BACKWARD_ADD:
                dim0 = int(params.get('n', 0))
            elif runtime_op == EngineOp.BACKWARD_GRADE_PROJECT:
                arg0 = ptrs[0]
                dim0 = int(params.get('n', 0))
                dim1 = int(params.get('target_grade', 0))
            elif runtime_op == EngineOp.BACKWARD_BVEXP:
                arg0 = ptrs[0]
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                dim0 = int(params.get('n', 0))
                dim1 = int(params.get('input_components', 8))
            elif runtime_op == EngineOp.BACKWARD_SOFTMAX:
                dim0 = int(params.get('n_rows', 0))
                dim1 = int(params.get('row_len', 0))
            elif runtime_op == EngineOp.BACKWARD_WEIGHTED_SUM:
                arg0 = ptrs[0]
                arg1 = ptrs[1] if len(ptrs) > 1 else 0
                arg3 = ptrs[2] if len(ptrs) > 2 else 0
                dim0 = int(params.get('batch', 0))
                dim1 = int(params.get('n_heads', 1))
                dim2 = int(params.get('seq', 0))
                dim3 = int(params.get('d_head', 0))
                if params.get('mode') == 'weights':
                    dim3 = -dim3
            elif runtime_op == EngineOp.BACKWARD_COPY:
                dim0 = int(params.get('n', 0))
                dim1 = int(params.get('d_out', 0))
                dim2 = int(params.get('input_index', 0))
            elif runtime_op == EngineOp.ADAM_FULL:
                dim0 = int(params.get('n_floats', 0))
                dim1 = int(params.get('step', 1))
                dim2 = int(float(params.get('beta1', 0.9)) * 10000)
                dim3 = int(float(params.get('beta2', 0.999)) * 10000)

            scalar0 = float(params.get('scale', params.get('eps', params.get('lr', 0.0))))
            scalar1 = float(params.get('eps', 0.0))
            self._program.add(
                opcode=runtime_op,
                arg0=arg0, arg1=arg1, arg2=arg2, arg3=arg3,
                dim0=dim0, dim1=dim1, dim2=dim2, dim3=dim3,
                scalar0=scalar0, scalar1=scalar1,
            )
            if runtime_op == EngineOp.ADAM_FULL:
                param_spec = self._plan_buffer_specs[cmd.input_buffers[0]]
                self._program.commands[-1].pad[0] = self._adam_group_size_for_spec(param_spec)
                self._adam_cmd_indices.append(len(self._program.commands) - 1)

    def _buffer_n_floats(self, buf_id: int) -> int:
        spec = self._plan_buffer_specs[buf_id]
        if 'n_floats' in spec:
            return int(spec['n_floats'])
        size_bytes = int(spec.get('size_bytes', 0))
        dtype = np.dtype(spec.get('dtype', np.float32))
        if dtype.itemsize == 0:
            raise ValueError(f"Invalid dtype itemsize for buffer {buf_id}")
        return size_bytes // dtype.itemsize

    def _adam_group_size_for_spec(self, spec: Dict[str, Any]) -> int:
        storage = int(spec.get('storage_components', 1))
        if storage in (1, 3, 8):
            return storage
        shape = tuple(spec.get('shape', ()))
        if not shape:
            return 1
        last_dim = int(shape[-1])
        if last_dim == 8:
            return 8
        if last_dim == 3:
            return 3
        if last_dim == 1:
            return 1
        return 1

    def _insert_execution_plan_grad_clip(self, grad_buffer_ids: list[int]) -> None:
        if self._program is None or not self._plan_max_grad_norm:
            return

        max_norm = float(self._plan_max_grad_norm)
        for grad_buf in grad_buffer_ids:
            self._program.grad_clip(
                self._plan_buffer_ptrs[grad_buf],
                self._buffer_n_floats(grad_buf),
                max_norm,
                mode='accumulate',
            )
        self._program.add(opcode=EngineOp.BARRIER)
        for grad_buf in grad_buffer_ids:
            self._program.grad_clip(
                self._plan_buffer_ptrs[grad_buf],
                self._buffer_n_floats(grad_buf),
                max_norm,
                mode='apply',
            )
        self._program.zero(self._plan_grad_norm_ptr, 1)

    def _check_cuda_error(self, err: int, operation: str):
        """Check CUDA return value and raise RuntimeError on failure."""
        if err != 0:
            raise RuntimeError(
                f"CUDA error {err} during {operation}. "
                f"Check GPU memory and kernel configuration."
            )

    # ---- Utility ----

    def get_param(self, name: str) -> np.ndarray:
        """Download a parameter from GPU."""
        if self._use_execution_plan:
            if name not in self._plan_param_name_to_buffer:
                raise KeyError(name)
            buf_id = self._plan_param_name_to_buffer[name]
            spec = self._plan_buffer_specs[buf_id]
            return self._download_ptr(
                self._plan_buffer_ptrs[buf_id],
                self._spec_array_shape(spec),
                np.dtype(spec.get('dtype', np.float32)),
            )
        return self._pool.download(name)

    def get_param_by_id(self, param_id: int) -> np.ndarray:
        if not self._use_execution_plan:
            raise RuntimeError("param-id access is only valid for execution-plan engines")
        if param_id not in self._plan_param_id_to_buffer:
            raise KeyError(param_id)
        buf_id = self._plan_param_id_to_buffer[param_id]
        spec = self._plan_buffer_specs[buf_id]
        return self._download_ptr(
            self._plan_buffer_ptrs[buf_id],
            self._spec_array_shape(spec),
            np.dtype(spec.get('dtype', np.float32)),
        )

    def get_param_grad(self, name: str) -> np.ndarray:
        """Download the current execution-plan gradient buffer for a named parameter."""
        if not self._use_execution_plan:
            raise RuntimeError("gradient access is only valid for execution-plan engines")
        if name not in self._plan_param_name_to_grad_buffer:
            raise KeyError(name)
        buf_id = self._plan_param_name_to_grad_buffer[name]
        spec = self._plan_buffer_specs[buf_id]
        return self._download_ptr(
            self._plan_buffer_ptrs[buf_id],
            self._spec_array_shape(spec),
            np.dtype(spec.get('dtype', np.float32)),
        )

    def get_param_grad_by_id(self, param_id: int) -> np.ndarray:
        """Download the current execution-plan gradient buffer for a parameter id."""
        if not self._use_execution_plan:
            raise RuntimeError("gradient access is only valid for execution-plan engines")
        if param_id not in self._plan_param_id_to_grad_buffer:
            raise KeyError(param_id)
        buf_id = self._plan_param_id_to_grad_buffer[param_id]
        spec = self._plan_buffer_specs[buf_id]
        return self._download_ptr(
            self._plan_buffer_ptrs[buf_id],
            self._spec_array_shape(spec),
            np.dtype(spec.get('dtype', np.float32)),
        )

    def get_memory_bank_state(self) -> np.ndarray:
        """Download the mutable compiled memory-bank state buffer."""
        if not self._use_execution_plan:
            raise RuntimeError("memory-bank access is only valid for execution-plan engines")
        if self._plan_memory_bank_state_buffer_id is None:
            raise KeyError("memory.bank_state")
        buf_id = self._plan_memory_bank_state_buffer_id
        spec = self._plan_buffer_specs[buf_id]
        return self._download_ptr(
            self._plan_buffer_ptrs[buf_id],
            self._spec_array_shape(spec),
            np.dtype(spec.get('dtype', np.float32)),
        )

    def set_param(self, name: str, data: np.ndarray):
        """Upload a parameter to GPU."""
        if self._use_execution_plan:
            if name not in self._plan_param_name_to_buffer:
                raise KeyError(name)
            self._upload_ptr(self._plan_buffer_ptrs[self._plan_param_name_to_buffer[name]], data)
            return
        self._pool.upload(name, data)

    def get_loss_history(self) -> float:
        """Read current loss from GPU."""
        if self._use_execution_plan and self._plan_loss_ptr:
            return float(self._download_ptr(self._plan_loss_ptr, (1,), np.dtype(np.float32))[0])
        return self._pool.download_scalar('loss_out')

    def get_grad_norm(self) -> float:
        """Read current gradient norm from GPU."""
        if self._use_execution_plan and self._plan_grad_norm_ptr:
            sum_sq = float(self._download_ptr(self._plan_grad_norm_ptr, (1,), np.dtype(np.float32))[0])
            if sum_sq > 0.0:
                return math.sqrt(sum_sq)
            # Fall back to summing live grad buffers on-GPU when the kernel-side
            # accumulator is unavailable or zeroed.
            try:
                import cupy as cp
            except Exception:
                return 0.0

            total = cp.asarray(0.0, dtype=cp.float64)
            for buf_id in self._plan_grad_buffer_ids:
                view = self._plan_grad_buffer_views.get(buf_id)
                if view is None:
                    spec = self._plan_buffer_specs[buf_id]
                    shape = self._spec_array_shape(spec)
                    dtype = np.dtype(spec.get('dtype', np.float32))
                    nbytes = int(spec.get('size_bytes', 0))
                    mem = cp.cuda.UnownedMemory(
                        self._plan_buffer_ptrs[buf_id],
                        nbytes,
                        self,
                    )
                    mp = cp.cuda.MemoryPointer(mem, 0)
                    view = cp.ndarray(shape=shape, dtype=dtype, memptr=mp)
                    self._plan_grad_buffer_views[buf_id] = view
                total = total + cp.sum(view.astype(cp.float32) ** 2, dtype=cp.float64)
            return float(cp.sqrt(total).get())
        if self._pool:
            sum_sq = float(self._pool.download_scalar('grad_norm_out'))
            return math.sqrt(max(sum_sq, 0.0))
        return float('nan')

    def get_buffer_scalar(self, buffer_id: int) -> float:
        spec = self._plan_buffer_specs[buffer_id]
        arr = self._download_ptr(
            self._plan_buffer_ptrs[buffer_id],
            self._spec_array_shape(spec),
            np.dtype(spec.get('dtype', np.float32)),
        )
        return float(arr.reshape(-1)[0])

    @property
    def pool(self) -> MemoryPool:
        return self._pool

    @property
    def program(self) -> ProgramBuilder:
        return self._program

    @property
    def device_info(self) -> Dict[str, int]:
        return {
            'n_sms': self._n_sms,
            'max_blocks': self._max_blocks,
            'block_size': self._block_size,
        }

    def summary(self) -> str:
        """Print a summary of the compiled engine."""
        lines = ["=== CUDAEngine Summary ==="]
        lines.append(f"Device: {self._n_sms} SMs, "
                     f"{self._max_blocks} blocks x {self._block_size} threads")
        if self._pool:
            lines.append(self._pool.summary())
        if self._program:
            lines.append(f"Program: {len(self._program)} commands")
        return "\n".join(lines)

    # ---- Cleanup ----

    def free(self):
        """Free all GPU resources."""
        if self._d_commands and self._cudart:
            self._cudart.cudaFree(ctypes.c_void_p(self._d_commands))
            self._d_commands = 0
        if self._use_execution_plan and self._cudart:
            for ptr in self._plan_buffer_ptrs.values():
                if ptr:
                    self._cudart.cudaFree(ctypes.c_void_p(ptr))
            for m_ptr, v_ptr in self._plan_adam_state.values():
                if m_ptr:
                    self._cudart.cudaFree(ctypes.c_void_p(m_ptr))
                if v_ptr:
                    self._cudart.cudaFree(ctypes.c_void_p(v_ptr))
            if self._plan_loss_ptr:
                self._cudart.cudaFree(ctypes.c_void_p(self._plan_loss_ptr))
                self._plan_loss_ptr = 0
            if self._plan_grad_norm_ptr:
                self._cudart.cudaFree(ctypes.c_void_p(self._plan_grad_norm_ptr))
                self._plan_grad_norm_ptr = 0
            self._plan_buffer_ptrs = {}
            self._plan_buffer_specs = {}
            self._plan_param_name_to_buffer = {}
            self._plan_param_id_to_buffer = {}
            self._plan_param_name_to_grad_buffer = {}
            self._plan_param_id_to_grad_buffer = {}
            self._plan_input_buffer_names = {}
            self._plan_output_buffer_names = {}
            self._plan_adam_state = {}
            self._plan_grad_buffer_ids = []
            self._plan_grad_buffer_views = {}
            self._plan_loss_buffer_id = None
            self._plan_memory_bank_state_buffer_id = None
            self._plan_max_grad_norm = None
        if self._pool:
            self._pool.free()
            self._pool = None
        self._compiled = False
        self._use_execution_plan = False

    def __del__(self):
        self.free()

    def __repr__(self):
        status = "compiled" if self._compiled else "not compiled"
        return f"CUDAEngine({status}, {self._n_sms} SMs)"
