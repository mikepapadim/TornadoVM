"""
jTile R3 prototype - TileCompiler: cuTile kernel -> Tile IR bytecode -> tileiras -> sm_XX cubin.
All offline (no GPU/driver). This is the codegen half of a TornadoVM `tornado-drivers/tile` backend.
"""
import io, os, subprocess, tempfile
import cuda.tile as ct
from cuda.tile.compilation import KernelSignature, ArrayConstraint, CallingConvention, export_kernel

_TILEIRAS = os.path.expanduser("~/.local/lib/python3.10/site-packages/nvidia/cu13/bin/tileiras")

def _arr(dtype, ndim):
    return ArrayConstraint(dtype=dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=(), may_alias_internally=False,
        stride_constant=tuple([None]*(ndim-1)+[1]),
        stride_divisible_by=tuple([1]*ndim), shape_divisible_by=tuple([1]*ndim),
        base_addr_divisible_by=1)

class TileCompiler:
    def __init__(self, sm="sm_89", tileiras=_TILEIRAS):
        self.sm, self.tileiras = sm, tileiras

    def compile(self, kernel, params, outdir):
        """params: list of ArrayConstraint | int consts. Returns (cubin_path, entry_symbol, tilebc_path)."""
        sig = KernelSignature(params, CallingConvention.cutile_python_v1()) \
                 .with_mangled_symbol(kernel._annotated_function.pyfunc.__name__)
        buf = io.BytesIO()
        export_kernel(kernel, signatures=[sig], output_file=buf,
                      gpu_code=self.sm, output_format="tileir_bytecode")
        os.makedirs(outdir, exist_ok=True)
        bc = os.path.join(outdir, sig.symbol + ".tilebc")
        cu = os.path.join(outdir, sig.symbol + ".cubin")
        open(bc, "wb").write(buf.getvalue())
        subprocess.run([self.tileiras, "--gpu-name", self.sm, bc, "-o", cu], check=True)
        return cu, sig.symbol, bc

# ---- jTile GEMM kernel (the R2 front-end `Tile.matmul` maps to this) ----
@ct.kernel
def gemm(a, b, c, M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int]):
    i = ct.bid(0); j = ct.bid(1)
    ta = ct.load(a, (i, 0), shape=(64, K))
    tb = ct.load(b, (0, j), shape=(K, 64))
    ct.store(c, (i, j), ct.matmul(ta, tb))

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "tileir_out")
    tc = TileCompiler(sm="sm_89")
    cu, sym, bc = tc.compile(gemm, [_arr(ct.float16,2), _arr(ct.float16,2), _arr(ct.float32,2), 512,512,512], out)
    print(f"[TileCompiler] entry   = {sym}")
    print(f"[TileCompiler] bytecode= {bc} ({os.path.getsize(bc)} B)")
    print(f"[TileCompiler] cubin   = {cu} ({os.path.getsize(cu)} B, sm_89)")
    # write the entry name for the loader
    open(os.path.join(out, "entry.txt"), "w").write(sym)
