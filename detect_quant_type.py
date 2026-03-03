#!/usr/bin/env python3
"""
Automatically detect the quantization integer type (int4, int8, or int16)
from a quantized model.

Detection sources (in order of precedence):
  1. params_report.json - if model was exported with export_params_to_mem.py
  2. .mem files - infer from hex digit width per line
  3. ONNX model - initializer/value types, QuantizeLinear/DequantizeLinear nodes
  4. Checkpoint tensors - check dtype of stored parameters (if quantized)
  5. Model source code - scan for quantization patterns

Usage:
  python detect_quant_type.py --model-module path/to/QuantizedMNISTNet.py
  python detect_quant_type.py --mem-dir path/to/mem/files
  python detect_quant_type.py --onnx-model path/to/model.onnx
  python detect_quant_type.py --report path/to/params_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Literal, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

QuantType = Literal["int4", "int8", "int16"]


# ---------------------------------------------------------------------------
# Detection from params_report.json
# ---------------------------------------------------------------------------

def detect_from_report(report_path: Path) -> Optional[QuantType]:
    """
    Detect quantization type from params_report.json (export_params_to_mem.py output).
    """
    if not report_path.exists():
        return None
    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        bits = data.get("export_info", {}).get("tensors", [{}])[0].get("bits")
        if bits is not None:
            return {4: "int4", 8: "int8", 16: "int16"}.get(bits)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        LOGGER.debug(f"Could not parse report: {e}")
    return None


# ---------------------------------------------------------------------------
# Detection from .mem files
# ---------------------------------------------------------------------------

def _infer_from_mem_filename(mem_path: Path) -> Optional[QuantType]:
    """Check filename for _4, _8, _16 or _int4, _int8, _int16 hints."""
    name = mem_path.stem.lower()
    if "_int4" in name or "_4" in name:
        return "int4"
    if "_int8" in name or "_8" in name:
        return "int8"
    if "_int16" in name or "_16" in name:
        return "int16"
    return None


def detect_from_mem_files(mem_dir: Path, sample_size: int = 20) -> Optional[QuantType]:
    """
    Infer quantization type from hex line width in .mem files.
    - 1 hex digit per line → int4
    - 2 hex digits per line → int8
    - 4 hex digits per line → int16
    - Packed lines (many values per line): infer from value ranges.
    """
    mem_files = list(mem_dir.glob("*.mem"))
    if not mem_files:
        return None

    for mem_path in mem_files[:10]:
        qt = _infer_from_mem_filename(mem_path)
        if qt:
            return qt

    hex_widths: set[int] = set()
    for mem_path in mem_files[:5]:
        try:
            with mem_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    line = line.strip()
                    if line and not line.startswith("//"):
                        val = line.replace("0x", "").strip()
                        if val and all(c in "0123456789abcdefABCDEF" for c in val):
                            hex_widths.add(len(val))
        except OSError as e:
            LOGGER.debug(f"Could not read {mem_path}: {e}")

    if not hex_widths:
        return None

    width = min(hex_widths)
    if width == 1:
        return "int4"
    if width == 2:
        return "int8"
    if width == 4:
        return "int16"

    if width <= 2:
        return "int8"
    return "int16"


# ---------------------------------------------------------------------------
# Detection from model source code
# ---------------------------------------------------------------------------

INT16_PATTERNS = [
    r"float_to_int16|int16_to_float",
    r"int16|np\.int16",
    r"32767|32768|-32768",
    r"0xFFFF|0xffff",
    r"SCALE_FACTOR\s*=\s*256",
]

INT8_PATTERNS = [
    r"float_to_int8|int8_to_float",
    r"int8|np\.int8",
    r"127|128|-128",
    r"0xFF|0xff",
]

INT4_PATTERNS = [
    r"float_to_int4|int4_to_float",
    r"int4|bit_width\s*=\s*4|bits\s*=\s*4",
    r"[-]?8\b|7\b",  
    r"0xF\b|0xf\b",
]


def detect_from_source(
    source_path: Path,
    *,
    weak_int4: bool = False,
) -> Optional[QuantType]:
    """
    Scan Python source for quantization patterns.
    Returns int4, int8 or int16 based on strongest indicators
    """
    if not source_path.exists():
        return None
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as e:
        LOGGER.debug(f"Could not read {source_path}: {e}")
        return None

    scores: dict[QuantType, int] = {"int4": 0, "int8": 0, "int16": 0}

    for pat in INT16_PATTERNS:
        if re.search(pat, text):
            scores["int16"] += 2

    for pat in INT8_PATTERNS:
        if re.search(pat, text):
            scores["int8"] += 2

    for pat in INT4_PATTERNS:
        if re.search(pat, text):
            scores["int4"] += 1 if weak_int4 else 2

    if re.search(r"0xFFFF|float_to_int16|int16_to_float", text):
        scores["int16"] += 2

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return None


# ---------------------------------------------------------------------------
# Detection from checkpoint (PyTorch state_dict)
# ---------------------------------------------------------------------------

def detect_from_checkpoint(checkpoint_path: Path) -> Optional[QuantType]:
    """
    Check state_dict tensor dtypes. If quantized tensors are stored,
    infer from dtype (int8, int16). Note: many models store float32.
    """
    if not checkpoint_path.exists():
        return None
    try:
        import torch
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "state_dict" in data:
            data = data["state_dict"]
        if not isinstance(data, dict):
            return None
        for key, tensor in data.items():
            if hasattr(tensor, "dtype"):
                dt = tensor.dtype
                if dt == torch.int16:
                    return "int16"
                if dt == torch.int8:
                    arr = tensor.numpy()
                    if arr.size > 0:
                        mn, mx = int(arr.min()), int(arr.max())
                        if -8 <= mn and mx <= 7:
                            return "int4"
                    return "int8"
    except Exception as e:
        LOGGER.debug(f"Could not load checkpoint: {e}")
    return None


# ---------------------------------------------------------------------------
# Detection from ONNX model
# ---------------------------------------------------------------------------

# ONNX TensorProto.DataType enum values (for when onnx module may not be available)
_ONNX_DTYPE_INT4 = 22
_ONNX_DTYPE_INT8 = 3
_ONNX_DTYPE_INT16 = 5
_ONNX_DTYPE_UINT4 = 21
_ONNX_DTYPE_UINT8 = 2
_ONNX_DTYPE_UINT16 = 4


def detect_from_onnx(onnx_path: Path) -> Optional[QuantType]:
    """
    Detect quantization type from an ONNX model.
    Checks initializers (weights), value_info, and QuantizeLinear/DequantizeLinear nodes.
    """
    if not onnx_path.exists():
        return None
    try:
        # Try project-bundled onnx_lib first (sibling convert_model_to_RTL/onnx_lib)
        _script_dir = Path(__file__).resolve().parent
        _onnx_lib = _script_dir.parent / "convert_model_to_RTL" / "onnx_lib"
        if _onnx_lib.is_dir() and str(_onnx_lib) not in sys.path:
            sys.path.insert(0, str(_onnx_lib))
        import onnx
        model = onnx.load(str(onnx_path))
    except ImportError as e:
        LOGGER.debug(f"onnx package not installed: {e}")
        return None
    except Exception as e:
        LOGGER.debug(f"Could not load ONNX model: {e}")
        return None

    seen_types: set[int] = set()

    # 1. Check initializers (weights/biases)
    for init in model.graph.initializer:
        if init.data_type:
            seen_types.add(init.data_type)

    # 2. Check value_info (input/output/value types)
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.type and vi.type.tensor_type and vi.type.tensor_type.elem_type:
            seen_types.add(vi.type.tensor_type.elem_type)

    # 3. Check QuantizeLinear/DequantizeLinear nodes for output type
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            for out_name in node.output:
                for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
                    if vi.name == out_name and vi.type and vi.type.tensor_type and vi.type.tensor_type.elem_type:
                        seen_types.add(vi.type.tensor_type.elem_type)
                        break

    # Map ONNX dtypes to QuantType (prefer smallest/most quantized)
    if any(dt in (_ONNX_DTYPE_INT4, _ONNX_DTYPE_UINT4) for dt in seen_types):
        return "int4"
    if any(dt in (_ONNX_DTYPE_INT8, _ONNX_DTYPE_UINT8) for dt in seen_types):
        return "int8"
    if any(dt in (_ONNX_DTYPE_INT16, _ONNX_DTYPE_UINT16) for dt in seen_types):
        return "int16"

    return None


# ---------------------------------------------------------------------------
# Main detection logic
# ---------------------------------------------------------------------------

def detect_quantization_type(
    *,
    model_module: Optional[Path] = None,
    mem_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
    onnx_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    search_dir: Optional[Path] = None,
) -> QuantType:
    """
    Detect quantization type from available sources.
    Returns int4, int8, or int16. Defaults to int16 if detection fails.
    """
    detected: Optional[QuantType] = None
    source = ""

    # 1. Report (highest confidence)
    if report_path:
        detected = detect_from_report(report_path)
        if detected:
            source = "params_report.json"

    # 2. .mem files
    if not detected and mem_dir:
        detected = detect_from_mem_files(mem_dir)
        if detected:
            source = "mem files"

    # 3. ONNX model
    if not detected and onnx_path:
        detected = detect_from_onnx(onnx_path)
        if detected:
            source = "ONNX model"

    # 4. Checkpoint
    if not detected and checkpoint_path:
        detected = detect_from_checkpoint(checkpoint_path)
        if detected:
            source = "checkpoint"

    # 5. Model source
    if not detected and model_module:
        detected = detect_from_source(model_module, weak_int4=True)
        if detected:
            source = "model source"

    # 6. Search directory for report.json, .mem, or .onnx
    if not detected and search_dir:
        report_candidate = search_dir / "params_report.json"
        if report_candidate.exists():
            detected = detect_from_report(report_candidate)
            if detected:
                source = "params_report.json"

        if not detected:
            mem_dir_candidate = search_dir / "mem"
            if mem_dir_candidate.is_dir():
                detected = detect_from_mem_files(mem_dir_candidate)
            if not detected:
                detected = detect_from_mem_files(search_dir)
            if not detected:
                for onnx_file in search_dir.glob("*.onnx"):
                    detected = detect_from_onnx(onnx_file)
                    if detected:
                        source = "ONNX model"
                        break
            if detected and not source:
                source = "mem files"

    if detected:
        LOGGER.info(f"Detected quantization type: {detected} (from {source})")
        return detected

    LOGGER.warning("Could not detect quantization type; defaulting to int16")
    return "int16"


def quant_type_to_bits(qt: QuantType) -> int:
    """Convert 'int4'/'int8'/'int16' to bit width 4, 8, or 16."""
    return {"int4": 4, "int8": 8, "int16": 16}[qt]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect quantization type (int4, int8, int16) from a quantized model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-module",
        type=Path,
        default=None,
        help="Path to model Python module (e.g., QuantizedMNISTNet.py)",
    )
    parser.add_argument(
        "--mem-dir",
        type=Path,
        default=None,
        help="Directory containing .mem files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to params_report.json",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        dest="onnx_path",
        help="Path to ONNX model file (.onnx)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to PyTorch checkpoint (.pth)",
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=None,
        help="Directory to search for report.json or .mem files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the detected type (int4, int8, or int16)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Require at least one input
    if not any([args.model_module, args.mem_dir, args.report, args.onnx_path, args.checkpoint, args.search_dir]):
        parser.error(
            "At least one of --model-module, --mem-dir, --report, --onnx-model, "
            "--checkpoint, or --search-dir is required"
        )

    result = detect_quantization_type(
        model_module=args.model_module,
        mem_dir=args.mem_dir,
        report_path=args.report,
        onnx_path=args.onnx_path,
        checkpoint_path=args.checkpoint,
        search_dir=args.search_dir,
    )

    if args.quiet:
        print(result)
    else:
        print(f"Quantization type: {result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
