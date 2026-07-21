import ctypes, os, sys
from ctypes import c_int, c_uint, c_void_p, c_char_p, byref, POINTER
OUT = os.path.join(os.path.dirname(__file__), "tileir_out")
cubin = [f for f in os.listdir(OUT) if f.endswith(".cubin")][0]
entry = open(os.path.join(OUT, "entry.txt")).read().strip()
data = open(os.path.join(OUT, cubin), "rb").read()

cu = ctypes.CDLL("libcuda.so.1")
for n,(res,arg) in {
  "cuInit":(c_int,[c_uint]),
  "cuDriverGetVersion":(c_int,[POINTER(c_int)]),
  "cuDeviceGet":(c_int,[POINTER(c_int),c_int]),
  "cuCtxCreate_v2":(c_int,[POINTER(c_void_p),c_uint,c_int]),
  "cuModuleLoadData":(c_int,[POINTER(c_void_p),c_void_p]),
  "cuModuleGetFunction":(c_int,[POINTER(c_void_p),c_void_p,c_char_p]),
  "cuGetErrorName":(c_int,[c_int,POINTER(c_char_p)]),
}.items():
    f=getattr(cu,n); f.restype=res; f.argtypes=arg
def name(rc):
    p=c_char_p(); cu.cuGetErrorName(rc, byref(p)); return p.value.decode() if p.value else str(rc)
def step(lbl, rc):
    print(f"  [{lbl}] -> {'OK' if rc==0 else 'ERR '+name(rc)}"); sys.stdout.flush(); return rc==0

drv=c_int(0); cu.cuDriverGetVersion(byref(drv)); print(f"driver CUDA {drv.value//1000}.{(drv.value%1000)//10}"); sys.stdout.flush()
step("cuInit", cu.cuInit(0))
dev=c_int(0); step("cuDeviceGet", cu.cuDeviceGet(byref(dev),0))
ctx=c_void_p(); step("cuCtxCreate", cu.cuCtxCreate_v2(byref(ctx),0,dev))
mod=c_void_p(); print(f"loading {cubin} ({len(data)} B, sm_89) ..."); sys.stdout.flush()
if step("cuModuleLoadData", cu.cuModuleLoadData(byref(mod), data)):
    fn=c_void_p()
    if step("cuModuleGetFunction", cu.cuModuleGetFunction(byref(fn),mod,entry.encode())):
        print(f"  entry resolved: {entry}\n  => LOADED+RESOLVED; only cuLaunchKernel remains.")
