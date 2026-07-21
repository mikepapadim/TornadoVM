# jTile R3 prototype — cuTile → Tile IR → tileiras → sm_89 cubin → driver load

> **Experimental / research prototype.** Standalone Python scripts — **not** a Maven module and
> **not** part of the TornadoVM build. Requires `pip install --user 'cuda-tile[tileiras]'`
> (userspace CUDA-13 Tile IR toolchain). Explores the "Tile path" (R3) for the jTile API.

Proves the TornadoVM "Tile path" backend end-to-end **minus the final launch**, entirely in
userspace on an RTX 4090 (driver 565 / CUDA 12.7). No root, no reboot.

## Setup
```
pip install --user 'cuda-tile[tileiras]'
python3 tile_compiler.py     # cuTile GEMM -> Tile IR bytecode -> tileiras -> sm_89 cubin
python3 tile_load.py         # driver-API load (mirrors TornadoVM cuda-jni)
```

## Toolchain (userspace, pip `--user`)
```
pip install --user 'cuda-tile[tileiras]'
  cuda.tile            # cuTile Python frontend (@kernel, load/store/bid/matmul/mma)
  nvidia-cuda-tileiras # tileiras binary: ~/.local/.../nvidia/cu13/bin/tileiras (13.3.36)
  cuda-toolkit 13.3    # nvcc/ptxas (compile only; no driver)
```

## Artifacts
- `tile_compiler.py` — **TileCompiler** (reusable): a cuTile kernel + `KernelSignature` →
  `export_kernel(output_format="tileir_bytecode")` → `.tilebc` (magic `\x7fTil`) →
  `tileiras --gpu-name sm_89` → `.cubin`. This is the **codegen half of `tornado-drivers/tile`**.
- `tile_load.py` — driver-API loader via ctypes on the real `libcuda.so.1`. Mirrors TornadoVM's
  `cuda-jni`: `cuInit` → `cuCtxCreate` → `cuModuleLoadData` → `cuModuleGetFunction`.
- `tileir_out/…cubin` — sm_89 GEMM cubin; SASS = **256× `HMMA.16816.F16`** (Ada tensor cores,
  via system `cuobjdump -sass`). Resources: REG 255, SHARED 56 KB, param block 424 B.

## End-state (verified)
| Step | Result |
|------|--------|
| emit Tile IR bytecode | ✅ offline |
| tileiras → sm_89 cubin | ✅ offline |
| SASS tensor-core (HMMA) | ✅ 256× `HMMA.16816.F16` |
| cuInit / ctx | ✅ |
| **cuModuleLoadData** | ❌ `CUDA_ERROR_INVALID_IMAGE` (driver 565 rejects CUDA-13 cubin) |
| cuLaunchKernel | blocked on load |

**Only blocker: GPU driver 565 → ≥ R580** (`sudo apt install nvidia-driver-580`; reboot). Then
`cuModuleLoadData` succeeds and launch works.

## Mapping to TornadoVM `tornado-drivers/tile`
1. **Codegen** = `TileCompiler`: R2 front-end `Tile*` nodes → emit `cuda_tile` (bytecode via the
   pip frontend now; textual MLIR later) → `tileiras` → cubin. Cache like the CUDA backend.
2. **Load/launch** = reuse `cuda-jni` `cuModuleLoad`/`cuModuleGetFunction`/`cuLaunchKernel`
   unchanged (Tile IR cubins are ordinary cubins). Entry symbol is the mangled
   `gemm_Kt1_A2f16…_I512_I512_I512` (derivable from the signature).
3. **Runtime-marshalling task** (the real remaining work): the `cutile_python_v1` calling
   convention packs each array arg as `{ptr, shape[], strides[]}` + int consts into a **424-byte
   param block**. TornadoVM's launcher must build that block from its device buffers + grid
   (tile-block 3-D grid = `WorkerGrid`). This is the one non-trivial integration piece.

## Repro
```
python3 tile_compiler.py     # -> tileir_out/<sym>.tilebc, .cubin, entry.txt
python3 tile_load.py         # -> cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (until driver >= R580)
/usr/local/cuda-12.6/bin/cuobjdump -sass tileir_out/*.cubin | grep -c HMMA   # 256
```
