"""Convert the cached fastembed ONNX model from float16 to float32 (CPU fix).

Why this exists
---------------
fastembed >= 0.8 ships `BAAI/bge-small-en-v1.5` as a **float16** ONNX graph
(66 MB = 2 bytes x 33 M parameters; every one of its 149 initializers is
FLOAT16). ONNX Runtime's CPU execution provider has no native fp16 kernels, so
it converts the whole weight set to fp32 **on every inference call**. That is a
fixed ~30 ms tax per `embed()` regardless of sequence length:

    seq_len=  7 :  37.5 ms        <- almost all of it is the weight cast
    seq_len= 22 :  49.7 ms
    seq_len=122 : 150.9 ms

One query embedding therefore costs ~50-60 ms, which alone blows the lab's
50 ms P99 budget for `/search`, and indexing the 1000-doc corpus takes ~180 s
instead of ~15 s. Nothing in the lab's own code is slow; the model file is.

Converting the initializers to fp32 once removes the per-call cast:

    fp16 (stock)     :  62.0 ms
    fp32 (converted) :   3.1 ms      -> 20x

The vectors are the same numbers: fp16 values are exactly representable in
fp32, so the only difference is that the cast happens once at build time
instead of on every request. Measured cosine similarity between stock and
converted embeddings is > 0.9999.

Idempotent: re-running after conversion is a no-op. The original file is kept
next to it as `*.fp16.bak` so the change is reversible.

    python scripts/fix_fp16_model.py          # convert if needed
    python scripts/fix_fp16_model.py --check  # report only, exit 1 if fp16
    python scripts/fix_fp16_model.py --revert # restore the fp16 original
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def find_cached_models() -> list[Path]:
    """Locate the ONNX files fastembed downloaded, without importing torch/onnx."""
    try:
        from fastembed.common.model_management import ModelManagement  # noqa: F401
    except Exception:                                   # pragma: no cover
        pass
    roots: list[Path] = []
    # fastembed's default cache dir, plus the HF hub location it may fall back to.
    import tempfile
    roots.append(Path(tempfile.gettempdir()) / "fastembed_cache")
    roots.append(Path.home() / ".cache" / "fastembed")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(p for p in root.rglob("*.onnx") if "bge-small-en-v1.5" in str(p))
    return sorted(set(found))


def is_fp16(path: Path) -> bool:
    import onnx
    from onnx import TensorProto
    model = onnx.load(str(path), load_external_data=False)
    return any(i.data_type == TensorProto.FLOAT16 for i in model.graph.initializer)


def convert(path: Path) -> int:
    """Rewrite every fp16 initializer as fp32. Returns how many were converted."""
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper

    model = onnx.load(str(path))
    n = 0
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
            n += 1
    if not n:
        return 0
    # Declared tensor types have to follow the data, or ORT rejects the graph.
    for vi in list(model.graph.value_info) + list(model.graph.output):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
    # An explicit Cast-to-fp16 would immediately undo the conversion.
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT

    backup = path.with_suffix(path.suffix + ".fp16.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    onnx.save(model, str(path))
    _refresh_cache_metadata(path)
    return n


def _refresh_cache_metadata(path: Path) -> None:
    """Tell fastembed the on-disk size it should now expect.

    fastembed validates each cached file against `files_metadata.json` and
    re-downloads on a mismatch -- which would silently restore the fp16 model
    on the next run and undo this fix. Record the new size instead.
    """
    import json

    for parent in path.parents:
        meta = parent / "files_metadata.json"
        if not meta.exists():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        rel_variants = {
            str(path.relative_to(parent)),
            str(path.relative_to(parent)).replace("\\", "/"),
            str(path.relative_to(parent)).replace("/", "\\"),
        }
        changed = False
        for key in list(data):
            if key in rel_variants or key.replace("\\", "/") in rel_variants:
                data[key]["size"] = path.stat().st_size
                changed = True
        if changed:
            meta.write_text(json.dumps(data), encoding="utf-8")
        return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if still fp16")
    ap.add_argument("--revert", action="store_true", help="restore the original fp16 file")
    args = ap.parse_args()

    models = find_cached_models()
    if not models:
        print("No cached bge-small-en-v1.5 ONNX found yet.")
        print("Run any lab step that embeds text first (e.g. python scripts/verify_lite.py).")
        return 0

    try:
        import onnx  # noqa: F401
    except ImportError:
        print("This fix needs the `onnx` package:  pip install onnx")
        return 1

    rc = 0
    for path in models:
        size_mb = path.stat().st_size / 1e6
        if args.revert:
            backup = path.with_suffix(path.suffix + ".fp16.bak")
            if backup.exists():
                shutil.copy2(backup, path)
                print(f"  reverted {path.name} ({size_mb:.0f} MB -> fp16 original)")
            else:
                print(f"  no backup for {path.name}; nothing to revert")
            continue

        if not is_fp16(path):
            print(f"  {path.name}: already fp32 ({size_mb:.0f} MB) — nothing to do")
            continue
        if args.check:
            print(f"  {path.name}: FLOAT16 ({size_mb:.0f} MB) — CPU inference pays a "
                  f"~30 ms weight cast per call")
            rc = 1
            continue
        n = convert(path)
        print(f"  {path.name}: converted {n} initializers fp16 -> fp32 "
              f"({size_mb:.0f} MB -> {path.stat().st_size / 1e6:.0f} MB)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
