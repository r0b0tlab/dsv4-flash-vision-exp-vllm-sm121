#!/usr/bin/env python3
"""Wait for DSV4V rank0 API readiness; report served model + KV cache tokens."""
import argparse
import json
import sys
import time
import urllib.request


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    deadline = time.time() + args.timeout
    last_err = ""
    while time.time() < deadline:
        try:
            models = get(base + "/v1/models", 30)
            served = models["data"][0]["id"]
            kv_tokens = None
            try:
                metrics = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
                for line in metrics.splitlines():
                    if line.startswith("vllm:kv_cache_size_tokens"):
                        kv_tokens = line.split()[-1]
                        break
            except Exception:
                pass
            print(f"READY served={served} kv_cache_size_tokens={kv_tokens}")
            sys.exit(0)
        except Exception as e:
            last_err = str(e)[:200]
        time.sleep(15)
    print(f"TIMEOUT after {args.timeout}s last_error={last_err}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
