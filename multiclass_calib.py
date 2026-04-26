#!/usr/bin/env python3
"""
Build a compressed ``.npz`` calibration archive for ``multiclass_quantize.py --calibration-mode npy``.

Reads a **float** ONNX model (any shape, e.g. NCHW for conv inputs), inspects inputs via ONNX Runtime,
and writes one array per input name. Shapes match ``NpyCalibrationDataReader`` in ``multiclass_quantize.py``:
each key is ``(N, *input_shape)`` with ``float32`` (stacked batch).

Synthetic data (structured + optional RNG) is **reproducible** with ``--seed``; it is not a substitute for
production traces when you care about activation ranges — use real traces in a matching ``.npz``.

Examples::

  py -3.12 multiclass_calib.py --onnx Multiclass_Conv.onnx
  py -3.12 multiclass_calib.py --onnx models/mynet.onnx -o calib/mynet_calib.npz --seed 0

**Random batch counts:** If ``--random-normal`` / ``--random-uniform`` are omitted, they are chosen from the
model’s resolved input shapes (total element count and number of inputs): larger tensors get more synthetic
draws (capped) so small models stay fast and wide models get extra coverage.

Requires: ``onnxruntime``, ``numpy``.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import onnxruntime as ort


def _resolve_input_shape(shape: Tuple) -> Tuple[int, ...]:
    """Concrete shape for calibration tensors (aligns with onnx_quantize NpyCalibrationDataReader).

    ONNX graphs often carry symbolic dims (strings like "batch") or dynamic
    markers (None / 0 / -1).  We collapse every non-positive or non-numeric
    dim to 1 so numpy can allocate real tensors for calibration.
    """
    out: List[int] = []
    for d in shape:
        if d is None or isinstance(d, str):
            out.append(1)
            continue
        di = int(d)
        out.append(1 if di <= 0 else di)
    return tuple(out)


def _collect_inputs(session: ort.InferenceSession) -> Dict[str, Tuple[int, ...]]:
    """Map every model input name to its resolved concrete shape.

    The returned dict drives all downstream sample generation: each key
    becomes a top-level array in the output .npz calibration archive.
    """
    shapes: Dict[str, Tuple[int, ...]] = {}
    for inp in session.get_inputs():
        shapes[inp.name] = _resolve_input_shape(tuple(inp.shape))
    return shapes


def auto_random_sample_counts(shapes: Dict[str, Tuple[int, ...]]) -> Tuple[int, int]:
    """
    Heuristic (n_normal, n_uniform) from input volume.

    Scales slowly with ``sum(prod(shape))`` and slightly with the number of inputs. Caps keep runtime
    reasonable; floors avoid tiny calib sets on scalar inputs.

    Why log-scale?  A 10x larger tensor does NOT need 10x more draws to
    approximate the same activation distribution; diminishing returns kick in
    fast.  The cap at 2000 keeps calibration wall-time under a few seconds
    even for heavyweight models, while the floor of 200 ensures enough
    variety for tiny scalar-input nets.
    """
    if not shapes:
        return 200, 50
    # total_el = total number of scalar elements across ALL model inputs
    # (e.g. a model with inputs [1,720] and [1,6] -> total_el = 726).
    # It is the single metric that governs how many random draws we need.
    total_el = sum(int(np.prod(s, dtype=np.int64)) if s else 1 for s in shapes.values())
    total_el = max(int(total_el), 1)
    n_in = len(shapes)

    # Log-scale growth: 720 el -> mid-200s normal; ~150k el -> ~400+ normal.
    # Extra inputs add a small bonus (25 each) because each input has its own
    # activation range the quantizer must observe.
    log_el = math.log10(float(min(total_el, 10_000_000)))
    core = 70.0 * log_el + 25.0 * float(n_in - 1)
    n_normal = int(round(core))
    n_normal = max(200, min(2000, n_normal))
    n_uniform = max(50, min(500, n_normal // 4))
    return n_normal, n_uniform


def _linspace_tensor(shape: Tuple[int, ...], low: float, high: float) -> np.ndarray:
    n = int(np.prod(shape)) if shape else 1
    flat = np.linspace(low, high, n, dtype=np.float32)
    return flat.reshape(shape)


def _add_sample(
    samples: List[Dict[str, np.ndarray]],
    shapes: Dict[str, Tuple[int, ...]],
    factory: Callable[[str], np.ndarray],
) -> None:
    samples.append({name: np.asarray(factory(name), dtype=np.float32).copy() for name in shapes})


def build_calibration_batches(
    shapes: Dict[str, Tuple[int, ...]],
    *,
    seed: int,
    n_normal: int,
    n_uniform: int,
) -> Dict[str, np.ndarray]:
    """Return stacked arrays ``(N, *shape)`` per input name."""
    rng = np.random.default_rng(seed)
    samples: List[Dict[str, np.ndarray]] = []

    def add(factory: Callable[[str], np.ndarray]) -> None:
        _add_sample(samples, shapes, factory)

    # ── Deterministic edge-case patterns ──────────────────────────────────
    # These RNG-free samples guarantee the quantizer always observes:
    #   • zero / near-zero activations  (bias-only behaviour)
    #   • linearly spaced ramps          (full signed range coverage)
    #   • constant positive & negative   (saturation / clipping corners)
    #   • alternating ±0.1              (high-frequency sign changes)
    # They form a small, fixed baseline (~13 samples) independent of seed.

    def z(name: str) -> np.ndarray:
        return np.zeros(shapes[name], dtype=np.float32)

    def o(name: str) -> np.ndarray:
        return np.full(shapes[name], 0.01, dtype=np.float32)

    def l05(name: str) -> np.ndarray:
        return _linspace_tensor(shapes[name], -0.5, 0.5)

    def l1(name: str) -> np.ndarray:
        return _linspace_tensor(shapes[name], -1.0, 1.0)

    add(z)
    add(o)
    add(l05)
    add(l1)
    for mag in (0.05, 0.1, 0.2, 0.35):

        def pos(m: float = mag) -> Callable[[str], np.ndarray]:
            return lambda name, mm=m: np.full(shapes[name], mm, dtype=np.float32)

        add(pos())
        add(lambda name, mm=mag: np.full(shapes[name], -mm, dtype=np.float32))

    def alternating(name: str) -> np.ndarray:
        s = shapes[name]
        n_el = int(np.prod(s)) if s else 1
        idx = np.arange(n_el, dtype=np.int64)
        flat = np.where(idx % 2 == 0, np.float32(0.1), np.float32(-0.1))
        return flat.reshape(s)

    add(alternating)

    # ── Random normal draws ───────────────────────────────────────────────
    # normal(0, σ=0.2) approximates typical sensor / pre-normalised feature
    # distributions; most calibration energy lands in [-0.6, 0.6], which is
    # the range where int8 quantisation bins matter most.
    for _ in range(n_normal):

        def nrm(name: str) -> np.ndarray:
            return rng.normal(0.0, 0.2, size=shapes[name]).astype(np.float32)

        add(nrm)

    # ── Random uniform draws ─────────────────────────────────────────────
    # uniform(-0.6, 0.6) fills the tails that normal rarely reaches,
    # ensuring the quantizer's min/max observers see plausible outliers
    # without extreme magnitudes that would waste int8 dynamic range.
    for _ in range(n_uniform):

        def uni(name: str) -> np.ndarray:
            return rng.uniform(-0.6, 0.6, size=shapes[name]).astype(np.float32)

        add(uni)

    # ── Stack into (N, *shape) arrays keyed by input name ─────────────────
    # This is the format NpyCalibrationDataReader expects: one .npz key per
    # ONNX input, each value shaped (N, *input_shape) with float32 dtype.
    names = list(shapes.keys())
    n = len(samples)
    out: Dict[str, np.ndarray] = {}
    for name in names:
        stacked = np.stack([samples[i][name] for i in range(n)], axis=0)
        out[name] = stacked
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build synthetic calibration .npz from a float ONNX model (all inputs, generic FC/binaryclass graphs)."
    )
    p.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Float ONNX model path (required).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .npz path (default: <onnx_stem>_calib.npz next to the ONNX file).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for normal/uniform samples")
    p.add_argument(
        "--random-normal",
        type=int,
        default=None,
        metavar="N",
        help="Number of normal(0, 0.2) samples (default: auto from ONNX input shapes; use 0 to disable)",
    )
    p.add_argument(
        "--random-uniform",
        type=int,
        default=None,
        metavar="N",
        help="Number of uniform(-0.6, 0.6) samples (default: auto from ONNX input shapes; use 0 to disable)",
    )
    p.add_argument(
        "--provider",
        action="append",
        default=None,
        help="ORT execution provider (repeatable). Default: CPUExecutionProvider",
    )
    args = p.parse_args()
    model_path = args.onnx.resolve()
    if not model_path.is_file():
        print(f"Not a file: {model_path}", file=sys.stderr)
        return 1

    # Step 1 — Load the float ONNX graph via ORT to introspect input metadata.
    providers = args.provider if args.provider else ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as e:
        print(f"Failed to load ONNX: {e}", file=sys.stderr)
        return 1

    # Step 2 — Resolve every model input to a concrete (int-tuple) shape.
    shapes = _collect_inputs(sess)
    if not shapes:
        print("Model has no inputs.", file=sys.stderr)
        return 1

    # Step 3 — Decide how many random draws to generate (auto or user-override).
    auto_n, auto_u = auto_random_sample_counts(shapes)
    n_normal = auto_n if args.random_normal is None else max(0, args.random_normal)
    n_uniform = auto_u if args.random_uniform is None else max(0, args.random_uniform)
    if args.random_normal is None or args.random_uniform is None:
        tot_el = sum(int(np.prod(s, dtype=np.int64)) if s else 1 for s in shapes.values())
        print(
            f"Auto random counts (total_elements={tot_el}, num_inputs={len(shapes)}): "
            f"--random-normal={n_normal} --random-uniform={n_uniform}"
        )

    # Step 4 — Build the full calibration dataset (deterministic + random).
    stacked = build_calibration_batches(
        shapes,
        seed=args.seed,
        n_normal=n_normal,
        n_uniform=n_uniform,
    )

    # Step 5 — Write compressed .npz; this is the file onnx_quantize.py reads.
    out_path = (
        args.output.resolve()
        if args.output is not None
        else (model_path.parent / f"{model_path.stem}_calib.npz").resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **stacked)

    n_batch = next(iter(stacked.values())).shape[0]
    detail = ", ".join(f"{k!r} {stacked[k].shape}" for k in stacked)
    print(f"Wrote {out_path} ({n_batch} stacked samples): {detail} dtype=float32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
