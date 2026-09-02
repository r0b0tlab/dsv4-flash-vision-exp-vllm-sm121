import pathlib, subprocess, vllm
print("version", vllm.__version__)
print("file", vllm.__file__)
p = pathlib.Path(vllm.__file__).resolve().parent
print("pkg", p)
vl = p / "models/deepseek_v4/nvidia/vl_model.py"
print("vl_exists", vl.is_file(), "bytes", vl.stat().st_size if vl.is_file() else 0)
if vl.is_file():
    t = vl.read_text()
    print("has_sorted", "sorted(" in t)
    print("has_AutoWeightsLoader", "AutoWeightsLoader" in t)
    for i, line in enumerate(t.splitlines(), 1):
        if "def load_weights" in line or "sorted(" in line or "process_weights_after_loading" in line:
            print(f"  vl:{i}:{line}")
cfg = p / "config/vllm.py"
print("config_vllm", cfg.is_file())
if cfg.is_file():
    for i, line in enumerate(cfg.read_text().splitlines(), 1):
        if "nvfp4" in line.lower():
            print(f"  vllm.py:{i}:{line.strip()[:200]}")
sp = p / "config/speculative.py"
if sp.is_file():
    t = sp.read_text()
    print("has_normalize_dspark", "_normalize_deepseek_v4_dspark_hf_config" in t)
    print("n_predict_block", "n_predict = block_size" in t)
r = subprocess.run(["vllm", "serve", "--help"], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "kv-cache" in line.lower() or "kv_cache_dtype" in line.lower():
        print("help:", line.strip()[:200])
# enum
try:
    import vllm.config.vllm as vc
    src = pathlib.Path(vc.__file__).read_text()
    if "class KVCacheDType" in src or "kv_cache_dtype" in src:
        print("kv dtype file", vc.__file__)
except Exception as e:
    print("enum_err", e)
