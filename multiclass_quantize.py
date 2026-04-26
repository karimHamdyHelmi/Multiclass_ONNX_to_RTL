#!/usr/bin/env python3
"""
Static INT8 QDQ quantization for ONNX (ONNX Runtime), used by ``multiclass_onnx_to_rtl.py``.

Produces a model with QuantizeLinear / DequantizeLinear and static scale initializers so
``multiclass_onnx_to_rtl.py`` can derive Fin/Fout for both Conv and FC layers.

**Calibration (scale source — documented):**

- **deterministic (default):** No RNG. Builds a fixed list of input tensors (zeros, small constants,
  per-feature ramps, alternating signs) so repeated runs are identical. Suitable when no dataset is
  available; use real data for production accuracy.
- **random:** Optional; uses ``numpy.random.Generator(seed)`` — reproducible only with the same seed.
- **npy / npz:** Load saved tensors from ``.npy`` files or a ``.npz`` archive (see ``--calibration-npy``).

  python multiclass_quantize.py --input model.onnx --output model_qdq_int8.onnx
  python multiclass_quantize.py --calibration-mode npy --calibration-npy ./batch.npz

Requires: pip install onnx onnxruntime
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


# ---------------------------------------------------------------------------
# Helper: resolve ONNX input metadata into concrete (name, shape) pairs.
# Dynamic/symbolic dims (strings like "batch", or None) are pinned to 1 so
# calibration tensors always have a fixed, predictable shape.
# ---------------------------------------------------------------------------
def _session_input_specs(session: ort.InferenceSession) -> List[tuple[str, List[int]]]:
    out: List[tuple[str, List[int]]] = []
    for inp in session.get_inputs():
        shape = [1 if isinstance(d, str) or d is None else int(d) for d in inp.shape]
        out.append((inp.name, shape))
    return out


# ---------------------------------------------------------------------------
# Synthetic calibration tensor builders.
# Each returns a single {input_name: ndarray} dict that becomes one
# calibration sample.  Together they let the quantizer observe a variety
# of activation ranges (zeros, constants, ramps, alternating signs).
# ---------------------------------------------------------------------------
def _make_ramp(name: str, shape: Sequence[int], lo: float = -0.5, hi: float = 0.5) -> Dict[str, np.ndarray]:
    """1D ramp along the last dimension (typical flattened feature vector)."""
    n = int(np.prod(shape))
    ramp = np.linspace(lo, hi, num=max(n, 2), dtype=np.float32).reshape(shape)
    return {name: ramp}


def _make_alternating(name: str, shape: Sequence[int], mag: float = 0.125) -> Dict[str, np.ndarray]:
    flat = np.prod(shape)
    a = np.array([mag if i % 2 == 0 else -mag for i in range(int(flat))], dtype=np.float32).reshape(shape)
    return {name: a}


# ---------------------------------------------------------------------------
# Calibration data readers.
#
# ORT's quantize_static() calls reader.get_next() in a loop to collect
# activation statistics (min/max per tensor).  Those statistics set the
# static scale/zero-point values embedded in the output QDQ graph.
# Each reader below populates self.data_list up front, then yields one
# sample per get_next() call until exhausted.
# ---------------------------------------------------------------------------

class DeterministicCalibrationDataReader(CalibrationDataReader):
    """
    Fixed, repeatable calibration tensors — no RNG involved.

    WHY deterministic?  Guarantees bit-identical quantization across machines
    and runs, which matters for RTL verification (golden-reference matching).
    The synthetic patterns (zeros, small/large constants, ramps, alternating
    signs) are chosen to exercise a broad activation range so the resulting
    scales are reasonable even without real training data.
    """

    def __init__(self, model_path: Path, *, magnitudes: Sequence[float] = (0.125, 0.25, 0.5)):
        self.enum_data: Optional[Iterator[Dict[str, np.ndarray]]] = None
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.specs = _session_input_specs(session)
        self.data_list: List[Dict[str, np.ndarray]] = []

        # pack() builds one calibration sample containing every model input
        def pack(build_one: Any) -> Dict[str, np.ndarray]:
            return {name: build_one(name, shape) for name, shape in self.specs}

        # Sequence of synthetic patterns — each one becomes a calibration sample:
        #   zeros          → baseline / bias-only observation
        #   ±0.01          → near-zero activations
        #   ±magnitudes    → small-to-mid range constants
        #   ramps          → smooth linear sweep across the feature axis
        #   alternating    → sign-flipping stress test
        self.data_list.append(pack(lambda n, sh: np.zeros(sh, dtype=np.float32)))
        self.data_list.append(pack(lambda n, sh: np.full(sh, 0.01, dtype=np.float32)))
        self.data_list.append(pack(lambda n, sh: np.full(sh, -0.01, dtype=np.float32)))
        for m in magnitudes:
            mm = float(m)
            self.data_list.append(pack(lambda n, sh, v=mm: np.full(sh, v, dtype=np.float32)))
            self.data_list.append(pack(lambda n, sh, v=mm: np.full(sh, -v, dtype=np.float32)))
        self.data_list.append(pack(lambda n, sh: _make_ramp(n, sh, -0.5, 0.5)[n]))
        self.data_list.append(pack(lambda n, sh: _make_ramp(n, sh, -1.0, 1.0)[n]))
        self.data_list.append(pack(lambda n, sh: _make_alternating(n, sh, mag=0.125)[n]))
        self.data_list.append(pack(lambda n, sh: _make_alternating(n, sh, mag=0.25)[n]))

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self) -> None:
        self.enum_data = None


class RandomCalibrationDataReader(CalibrationDataReader):
    """
    Gaussian-distributed calibration inputs (explicit seed for reproducibility).

    Use when the real data distribution is roughly Gaussian and you want
    calibration scales that better reflect typical activation magnitudes.
    For RTL golden-reference work, prefer DeterministicCalibrationDataReader
    or NpyCalibrationDataReader — they are fully platform-independent.
    """

    def __init__(self, model_path: Path, num_samples: int = 10, seed: int = 0):
        self.enum_data = None
        self._rng = np.random.default_rng(seed)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.data_list: list[dict[str, np.ndarray]] = []
        for _ in range(num_samples):
            sample: dict[str, np.ndarray] = {}
            for inp in session.get_inputs():
                shape = [1 if isinstance(d, str) or d is None else int(d) for d in inp.shape]
                sample[inp.name] = self._rng.standard_normal(shape, dtype=np.float32) * 0.25 + 0.01
            self.data_list.append(sample)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class NpyCalibrationDataReader(CalibrationDataReader):
    """
    Load calibration inputs from disk — the reader used by multiclass_onnx_to_rtl.py's
    pipeline (with a synthetic .npz produced by multiclass_calib.py).

    Three accepted input formats:

    - **Single .npy**: one sample; shape must match model input (or ``(1, …)`` batch).
    - **Directory of .npy**: each file is one sample for the sole model input,
      or named ``<input_name>__*.npy`` for multi-input models.
    - **.npz archive**: keys must match ONNX input names.  If the stored array
      has one extra leading dimension vs. the model shape, it is treated as a
      stacked batch and unpacked into individual samples for ORT.
    """

    def __init__(self, model_path: Path, npy_path: Path):
        self.enum_data = None
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in session.get_inputs()]
        self.input_shapes = [[1 if isinstance(d, str) or d is None else int(d) for d in i.shape] for i in session.get_inputs()]
        self.data_list = self._load(npy_path)

    def _load(self, path: Path) -> List[Dict[str, np.ndarray]]:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Calibration path not found: {path}")

        # --- .npz archive ---
        # Keys must match ONNX input names.  A stacked batch (rank = model_rank+1)
        # is split along dim-0 so each slice becomes one calibration sample.
        if path.suffix.lower() == ".npz":
            z = np.load(path, allow_pickle=False)
            keys = list(z.files)
            missing = [n for n in self.input_names if n not in keys]
            if missing:
                raise ValueError(f".npz missing input keys {missing}; have {keys}")
            ar0 = np.asarray(z[self.input_names[0]], dtype=np.float32)
            if ar0.ndim == len(self.input_shapes[0]):
                # Single sample — rank matches model input exactly
                return [{n: np.asarray(z[n], dtype=np.float32) for n in self.input_names}]
            if ar0.ndim == len(self.input_shapes[0]) + 1:
                # Stacked batch — unpack along leading dim into individual samples
                nbatch = ar0.shape[0]
                return [
                    {n: np.asarray(z[n], dtype=np.float32)[i] for n in self.input_names}
                    for i in range(nbatch)
                ]
            raise ValueError(f"Unexpected array rank for {self.input_names[0]}: {ar0.shape}")

        # --- Single .npy file ---
        if path.is_file() and path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=False).astype(np.float32)
            if len(self.input_names) != 1:
                raise ValueError("Single .npy requires exactly one model input")
            exp = tuple(self.input_shapes[0])
            if arr.shape == exp:
                return [{self.input_names[0]: arr}]
            if arr.shape == (1,) + exp:
                return [{self.input_names[0]: arr[0]}]
            raise ValueError(f".npy shape {arr.shape} does not match expected {exp} or (1,)+exp")

        # --- Directory of .npy files (one sample per file) ---
        if path.is_dir():
            files = sorted(path.glob("*.npy"))
            if not files:
                raise ValueError(f"No .npy files in {path}")
            if len(self.input_names) == 1:
                name = self.input_names[0]
                out: List[Dict[str, np.ndarray]] = []
                for f in files:
                    arr = np.load(f, allow_pickle=False).astype(np.float32)
                    exp = tuple(self.input_shapes[0])
                    if arr.shape == exp:
                        out.append({name: arr})
                    elif arr.shape == (1,) + exp:
                        out.append({name: arr[0]})
                    else:
                        raise ValueError(f"{f.name}: shape {arr.shape} vs expected {exp}")
                return out
            # multi-input: expect <inputname>__*.npy per sample — group by stem prefix
            raise ValueError(
                "Multi-input models: use a .npz with keys matching input names, or extend this loader."
            )

        raise ValueError(f"Unsupported calibration path: {path}")

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


# ---------------------------------------------------------------------------
# CLI entry point.
# Pipeline: parse args → select calibration reader → quantize_static() →
# emit QDQ INT8 model that binary_onnx_to_rtl.py will consume.
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Static QDQ INT8 quantization (ORT)")
    p.add_argument("--input", type=Path, required=True, help="Float ONNX model")
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output QDQ ONNX path",
    )
    p.add_argument(
        "--calibration-mode",
        choices=("deterministic", "random", "npy"),
        default="deterministic",
        help="deterministic=fixed patterns (default); random=Gaussian with --seed; npy=--calibration-npy",
    )
    p.add_argument(
        "--calibration-npy",
        type=Path,
        default=None,
        help="With mode=npy: path to .npy file, directory of .npy, or .npz (see NpyCalibrationDataReader)",
    )
    p.add_argument("--calibration-samples", type=int, default=10, help="random mode: number of samples")
    p.add_argument("--seed", type=int, default=0, help="random mode: RNG seed")
    args = p.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()
    if not inp.is_file():
        raise SystemExit(f"Input model not found: {inp}")

    # Select calibration reader based on --calibration-mode
    reader: CalibrationDataReader
    source_note: str
    if args.calibration_mode == "random":
        reader = RandomCalibrationDataReader(inp, num_samples=args.calibration_samples, seed=args.seed)
        source_note = f"random Gaussian (n={args.calibration_samples}, seed={args.seed})"
    elif args.calibration_mode == "npy":
        if args.calibration_npy is None:
            raise SystemExit("calibration-mode=npy requires --calibration-npy PATH")
        reader = NpyCalibrationDataReader(inp, args.calibration_npy.resolve())
        source_note = f"npy/npz from {args.calibration_npy}"
    else:
        reader = DeterministicCalibrationDataReader(inp)
        source_note = "deterministic fixed patterns (no RNG)"

    # Run ORT static quantization — inserts QuantizeLinear/DequantizeLinear
    # nodes with calibration-derived scales that rtl_mapper.py reads as Fin/Fout.
    quantize_static(
        model_input=str(inp),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"Static QDQ quantization complete: {out}")
    print(f"Calibration scale source: {source_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
