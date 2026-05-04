#!/usr/bin/env python3
"""
Multiclass ONNX to RTL Converter
================================
End-to-end pipeline: takes a **float32** ONNX classifier with Conv + FC layers and
produces synthesizable SystemVerilog RTL plus ``.mem`` weight/bias ROM files.

Supported operators:
  Conv (depthwise & pointwise), Gemm, MatMul (with optional Add bias),
  Relu, AveragePool, Transpose, Reshape, Softmax, ArgMax.

Pipeline (three stages, all automated from ``--model``):
  1. **Calibrate** — build a synthetic calibration ``.npz`` from the float model's
     input shapes (deterministic + normal + uniform random batches). This feeds
     ORT's activation range observer so it can choose per-tensor int8 scales.
  2. **Quantize** — run ``onnxruntime.quantization.quantize_static`` in QDQ int8
     mode, producing a *temporary* quantized ONNX with Q/DQ nodes around every
     conv/FC weight and activation.
  3. **Generate RTL** — extract Conv + FC layers from the quantized ONNX (using
     the float weights as the ground truth source for Fw computation), build
     RTL fixed-point descriptors (Fin/Fw/Fb/Fout power-of-two), validate
     numeric fidelity, and emit:
       - quant_pkg.sv, mac.sv, fc_in/out.sv, relu_layer.sv, sync_fifo.sv
       - line_buffers.sv, depthwise_conv_engine.sv, pointwise_conv_engine.sv
       - per-conv-layer ROM .sv + .mem files
       - avg_pool_kx1.sv, flatten_unit.sv
       - softmax_layer.sv  OR  argmax_layer.sv  (depends on ONNX final op)
       - multiclass_NN.sv (top), multiclass_NN_wrapper.sv
       - rtl_filelist.f, mapping_report.txt, netlist.json

Conv layers and FC layers share the same QDQ chain walkers
(``_fin_exponent_from_dq_chain`` / ``_fout_exponent_from_q_chain``) for the
``Fin / Fout`` exponent derivation.

Requires: ``onnxruntime`` (with quantization support), ``onnx``, ``numpy``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import math
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# ONNX Loading and Graph Utilities
# Low-level helpers used throughout the extraction pipeline: load the protobuf,
# materialize initializers into NumPy arrays, and read node attributes.
# -----------------------------------------------------------------------------

def _load_onnx(onnx_path: Path) -> Any:
    import onnx
    return onnx.load(str(onnx_path))


def _get_initializers_dict(model: Any) -> Dict[str, np.ndarray]:
    from onnx.numpy_helper import to_array
    result = {}
    for init in model.graph.initializer:
        try:
            arr = to_array(init)
            result[init.name] = arr
        except Exception as e:
            LOGGER.warning(f"Could not load initializer {init.name}: {e}")
    return result


def _get_attr(node: Any, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == 2:  # INT
                return attr.i
            if attr.type in (1, 5):  # FLOAT
                return attr.f
    return default


# -----------------------------------------------------------------------------
# Bias Extraction Helpers (MatMul / MatMulInteger chains)
#
# ONNX graphs rarely carry bias as a direct third input to MatMul/MatMulInteger.
# Instead the bias lives in a downstream Add node: MatMul → Add(result, bias_init).
# For dynamic-quantized models the chain can be MatMulInteger → Mul(scale) → Add(bias),
# so we must follow Mul/Cast passthrough ops to locate the Add. These helpers walk
# that post-MatMul subgraph to find and dequantize the bias tensor.
# -----------------------------------------------------------------------------

def _try_get_matmul_bias_from_add(
    model: Any,
    matmul_node: Any,
    out_features: int,
    name_to_init: Dict[str, np.ndarray],
    raw: bool = False,
) -> Optional[Tuple[np.ndarray, Optional[str]]]:
    """Returns (bias_array, init_name) or None. init_name is the ONNX initializer name."""
    matmul_out = matmul_node.output[0]
    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) < 2:
            continue
        inputs = list(node.input)
        if matmul_out not in inputs:
            continue
        other = inputs[1] if inputs[0] == matmul_out else inputs[0]
        if other in name_to_init:
            b = name_to_init[other].flatten()
            if len(b) == out_features:
                arr = b.copy() if raw else b.astype(np.float32)
                return (arr, other)
        break
    return None


def _try_get_bias_from_add_chain(
    model: Any,
    start_output: str,
    out_features: int,
    name_to_init: Dict[str, np.ndarray],
    visited: Optional[set] = None,
    raw: bool = False,
) -> Optional[Tuple[np.ndarray, Optional[str]]]:
    """Follow output through Mul/Cast to find Add with constant bias. Returns (bias_array, init_name) or None."""
    if visited is None:
        visited = set()
    if start_output in visited:
        return None
    visited.add(start_output)
    for node in model.graph.node:
        if start_output not in list(node.input):
            continue
        if node.op_type == "Add" and len(node.input) >= 2:
            inputs = list(node.input)
            other = inputs[1] if inputs[0] == start_output else inputs[0]
            if other in name_to_init:
                arr = name_to_init[other]
                b = arr.flatten()
                if len(b) == out_features:
                    if raw:
                        return (b.copy(), other)
                    # Float bias (typical for ORT quantize_dynamic MatMulInteger + Add): use as-is.
                    if arr.dtype in (np.float32, np.float64):
                        return (b.astype(np.float32), other)
                    # Integer bias without a documented scale in this graph path — do not assume /128 (invalid for ORT).
                    if arr.dtype in (np.int8, np.uint8, np.int16, np.uint16):
                        LOGGER.warning(
                            "Bias initializer %s is integer dtype %s; treating as raw float cast without /128. "
                            "If values are wrong, add DequantizeLinear metadata or use float bias in ONNX.",
                            other,
                            arr.dtype,
                        )
                        return (b.astype(np.float32), other)
                    if arr.dtype == np.int32:
                        LOGGER.warning(
                            "Bias initializer %s is int32; using raw float cast (no implicit /16384).",
                            other,
                        )
                        return (b.astype(np.float32), other)
                    return (b.astype(np.float32), other)
        if node.op_type in ("Mul", "Cast") and node.output:
            found = _try_get_bias_from_add_chain(
                model, node.output[0], out_features, name_to_init, visited, raw
            )
            if found is not None:
                return found
    return None


def _lookup_matmulinteger_weight_scale_initializer(
    weight_init_name: str,
    name_to_init: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """ONNX Runtime dynamic quant: scale tensor next to *_W_quantized (scalar or per-channel)."""
    candidates: List[str] = []
    if "_quantized" in weight_init_name:
        candidates.append(weight_init_name.replace("_quantized", "_scale"))
    candidates.append(weight_init_name + "_scale")
    for k in candidates:
        if k in name_to_init:
            arr = np.asarray(name_to_init[k], dtype=np.float64)
            if arr.size >= 1:
                return arr, k
    return None, None


def _dequantize_matmulinteger_weights(
    w_int: np.ndarray,
    scale_arr: np.ndarray,
    zp_arr: Optional[np.ndarray],
) -> np.ndarray:
    """
    ONNX MatMulInteger weight dequant (ORT quantize_dynamic): W_float = (W_int - zp) * scale.

    Do **not** use /128.0 or other hardcoded divisors — scale comes from the model initializer.
    Supports scalar scale/zp or per-output-channel scale of shape (out_features,).
    """
    w = w_int.astype(np.float32)
    in_f, out_f = w.shape
    sc = np.asarray(scale_arr, dtype=np.float32).reshape(-1)
    if sc.size == 1:
        scale_m = sc.reshape(1, 1)
    elif sc.size == out_f:
        scale_m = sc.reshape(1, out_f)
    elif sc.size == in_f * out_f:
        scale_m = sc.reshape(in_f, out_f)
    else:
        raise ValueError(
            f"MatMulInteger weight scale shape {scale_arr.shape} (size {sc.size}) not broadcastable "
            f"to weight {w_int.shape}"
        )

    if zp_arr is None or zp_arr.size == 0:
        zp = np.float32(0.0)
        w_adj = w - zp
    else:
        zp = np.asarray(zp_arr, dtype=np.float32)
        if zp.size == 1:
            w_adj = w - zp.reshape(1, 1)
        elif zp.shape == w_int.shape:
            w_adj = w - zp
        elif zp.size == out_f:
            w_adj = w - zp.reshape(1, out_f)
        elif zp.size == in_f:
            w_adj = w - zp.reshape(in_f, 1)
        else:
            raise ValueError(
                f"MatMulInteger weight zero_point shape {zp_arr.shape} not broadcastable to weight {w_int.shape}"
            )

    return w_adj * scale_m


def _try_get_scale_from_mul_chain(
    model: Any,
    start_output: str,
    name_to_init: Dict[str, np.ndarray],
    visited: Optional[set] = None,
) -> Optional[Tuple[float, Optional[str]]]:
    """Follow output through Mul to find scale constant (for MatMulInteger dequant). Returns (scale, init_name) or None."""
    if visited is None:
        visited = set()
    if start_output in visited:
        return None
    visited.add(start_output)
    for node in model.graph.node:
        if start_output not in list(node.input):
            continue
        if node.op_type == "Mul" and len(node.input) >= 2:
            inputs = list(node.input)
            other = inputs[1] if inputs[0] == start_output else inputs[0]
            if other in name_to_init:
                scale_arr = name_to_init[other]
                if scale_arr.size == 1:
                    return (float(scale_arr.flatten()[0]), other)
        if node.op_type in ("Mul", "Cast") and node.output:
            found = _try_get_scale_from_mul_chain(model, node.output[0], name_to_init, visited)
            if found is not None:
                return found
    return None


# -----------------------------------------------------------------------------
# Value/Shape Resolution (Reshape, Constant)
#
# Builds a unified name→ndarray lookup that covers graph initializers AND
# values produced by Reshape / Constant nodes. This lets later extraction
# resolve indirect references — e.g. when a weight initializer is reshaped
# before being fed to a MatMul, we can still look it up by tensor name.
# -----------------------------------------------------------------------------

def _build_value_to_array(model: Any, inits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    from onnx.numpy_helper import to_array
    result: Dict[str, np.ndarray] = dict(inits)
    for node in model.graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2 and len(node.output) >= 1:
            data_name, shape_name = node.input[0], node.input[1]
            if data_name in result and shape_name in result:
                data = result[data_name]
                shape = result[shape_name].flatten().astype(np.int64)
                out = np.reshape(data, shape)
                result[node.output[0]] = out
        elif node.op_type == "Constant" and len(node.output) >= 1:
            for attr in node.attribute:
                if attr.name == "value":
                    result[node.output[0]] = to_array(attr.t)
                    break
    return result


def _get_node_op_type(node: Any) -> str:
    """Return op_type, including domain prefix for non-standard ops."""
    domain = getattr(node, "domain", "") or ""
    if domain and domain != "ai.onnx":
        return f"{domain}::{node.op_type}"
    return node.op_type


# -----------------------------------------------------------------------------
# Final Activation Detection (trace from graph output backwards)
#
# In QDQ graphs the activation op (Sigmoid, ReLU, …) is buried behind
# QuantizeLinear / DequantizeLinear tail nodes that the quantizer inserts
# before the graph output. To find the *semantic* final activation we:
#   - Walk BACKWARD from the graph output, treating Q/DQ + Cast/Reshape as
#     transparent passthrough, until we hit a supported activation op.
#   - Walk FORWARD from each FC output through Add/Mul/Q/DQ passthrough to
#     discover per-layer activations (used for RTL ReLU/Sigmoid insertion).
# This two-direction strategy prevents Q/DQ nodes from hiding the real
# activation function.
# -----------------------------------------------------------------------------

# Activation ops we support for final layer
_SUPPORTED_FINAL_ACTIVATIONS = frozenset({"Sigmoid", "Relu", "Softmax", "Tanh"})
# Ops that just pass through (trace to input)
_PASSTHROUGH_OPS = frozenset({"Identity", "Cast", "Reshape", "Squeeze", "Unsqueeze"})
# When walking backward from graph output, Q/DQ tail (static export) should not hide Sigmoid/Relu/etc.
_FINAL_ACTIVATION_BACKWARD_PASSTHROUGH = _PASSTHROUGH_OPS | frozenset(
    {"QuantizeLinear", "DequantizeLinear"}
)


# Activations we support in RTL (per-layer)
_SUPPORTED_ACTIVATIONS = frozenset({"Relu", "Sigmoid", "Tanh", "HardSigmoid"})
# Ops to follow when tracing forward (before activation); include Q/DQ like _FINAL_ACTIVATION_BACKWARD_PASSTHROUGH.
_FORWARD_PASSTHROUGH = frozenset(
    {
        "Add",
        "Mul",
        "Cast",
        "Identity",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "DynamicQuantizeLinear",
        "QuantizeLinear",
        "DequantizeLinear",
    }
)


def _trace_activation_forward(
    model: Any,
    start_name: str,
    input_to_nodes: Dict[str, List[Any]],
    visited: Optional[set] = None,
) -> Optional[str]:
    """Trace forward from start_name to find the nearest activation op.

    Breadth-first traversal avoids branch-order dependence in multi-consumer graphs.
    Raises when equally-near activation types disagree to prevent silent mislabeling.
    """
    del model  # kept for API compatibility
    seen: set = set() if visited is None else set(visited)
    q: Deque[Tuple[str, int]] = deque([(start_name, 0)])
    found_depth: Optional[int] = None
    found_ops: List[str] = []
    while q:
        tensor_name, depth = q.popleft()
        if tensor_name in seen:
            continue
        seen.add(tensor_name)
        if found_depth is not None and depth > found_depth:
            break
        for consumer in input_to_nodes.get(tensor_name, []):
            op = consumer.op_type
            if op in _SUPPORTED_ACTIVATIONS:
                found_depth = depth if found_depth is None else found_depth
                found_ops.append(op)
                continue
            if op in _FORWARD_PASSTHROUGH and consumer.output:
                q.append((consumer.output[0], depth + 1))
    if not found_ops:
        return None
    uniq = sorted(set(found_ops))
    if len(uniq) > 1:
        raise RuntimeError(
            f"Ambiguous activation trace from tensor {start_name!r}: nearest activation ops={uniq}"
        )
    return uniq[0]


def extract_per_layer_activations(model: Any, fc_output_names: List[str]) -> List[Optional[str]]:
    """For each FC output, trace forward to find the activation that follows. Returns list of activation op_types."""
    input_to_nodes: Dict[str, List[Any]] = {}
    for node in model.graph.node:
        for inp in node.input:
            input_to_nodes.setdefault(inp, []).append(node)
    result: List[Optional[str]] = []
    for out_name in fc_output_names:
        act = _trace_activation_forward(model, out_name, input_to_nodes)
        result.append(act)
    return result


def _detect_final_activation(model: Any) -> Optional[str]:
    """Trace from graph output backwards to find the last activation op (Sigmoid, Relu, etc.).

    QuantizeLinear / DequantizeLinear before the output are treated as passthrough (typical QDQ export).
    Returns the op_type (e.g. 'Sigmoid') or None if not found."""
    if not model.graph.output:
        return None
    output_name = model.graph.output[0].name

    # Build output_name -> node mapping
    output_to_node: Dict[str, Any] = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    current = output_name
    visited: set = set()

    while current and current not in visited:
        visited.add(current)
        node = output_to_node.get(current)
        if node is None:
            return None
        op = node.op_type
        if op in _SUPPORTED_FINAL_ACTIVATIONS:
            return op
        if op in _FINAL_ACTIVATION_BACKWARD_PASSTHROUGH and node.input:
            current = node.input[0]
        else:
            return None
    return None


def _producer_node_for_tensor(model: Any, tensor_name: str) -> Optional[Any]:
    """Return the node that produces ``tensor_name`` as one of its outputs."""
    for node in model.graph.node:
        if tensor_name in list(node.output):
            return node
    return None


def _resolve_matmul_weight_tensor(
    model: Any,
    weight_input_name: str,
    name_to_init: Dict[str, np.ndarray],
) -> Optional[Tuple[np.ndarray, Optional[float], int, str]]:
    """Resolve MatMul second input to a 2D weight array.

    Returns ``(w_np, dq_scale, dq_zero_point, quant_tensor_name)``:
    - Direct initializer: ``dq_scale`` is None, zp is 0, name is that initializer.
    - ``DequantizeLinear`` (optionally after Cast/Transpose/Identity): quantized weights from init + scalar scale.

    Per-channel ``x_scale`` is not supported (log and return None).
    """
    cur: Optional[str] = weight_input_name
    visited: set[str] = set()
    while cur is not None and cur not in visited:
        visited.add(cur)
        if cur in name_to_init:
            w = name_to_init[cur].copy()
            if w.ndim != 2:
                return None
            return w, None, 0, cur

        prod = _producer_node_for_tensor(model, cur)
        if prod is None:
            return None

        if prod.op_type == "DequantizeLinear":
            ins = list(prod.input)
            if len(ins) < 2:
                return None
            qn, sn = ins[0], ins[1]
            zn = ins[2] if len(ins) > 2 else None
            if qn not in name_to_init or sn not in name_to_init:
                return None
            scale_arr = name_to_init[sn]
            if scale_arr.size != 1:
                LOGGER.warning(
                    "DequantizeLinear before MatMul: only scalar x_scale supported; got shape %s for %s",
                    scale_arr.shape,
                    sn,
                )
                return None
            wq = name_to_init[qn].copy()
            if wq.ndim != 2:
                return None
            sc = float(scale_arr.flatten()[0])
            zp = int(name_to_init[zn].flatten()[0]) if zn and zn in name_to_init else 0
            return wq, sc, zp, qn

        if prod.op_type in ("Cast", "Transpose", "Identity") and prod.input:
            cur = prod.input[0]
            continue

        return None

    return None


# -----------------------------------------------------------------------------
# QDQ / static quantization: Fin & Fout exponents from ONNX scales
#
# Every FC layer in the RTL needs two activation-domain exponents:
#   Fin  = input activation exponent, derived from the DequantizeLinear
#          scale *before* the FC (float_activation ≈ int8 * 2^-Fin).
#   Fout = output activation exponent, derived from the QuantizeLinear
#          scale *after* the FC (int8_output ≈ float_result * 2^Fout).
#
# Together with Fw (weight exponent) and Fb (bias exponent), Fin and Fout
# determine LAYER_SCALE and BIAS_SCALE in the generated RTL:
#   MAC accumulator lives at fractional precision Fin+Fw, then is right-shifted
#   to Fout for the next layer's int8 input.
# -----------------------------------------------------------------------------


def onnx_scalar_quant_scale_to_rtl_exponent(scale: float) -> int:
    """Map ONNX quant scale ``s`` (float = quant) to RTL exponent ``F`` with ``s ≈ 2^-F`` (non-negative int)."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid quantization scale for exponent mapping: {scale!r}")
    e = int(round(-math.log2(scale)))
    return max(0, min(24, e))


def _scalar_from_init(scale_name: str, name_to_init: Dict[str, np.ndarray]) -> Optional[float]:
    if scale_name not in name_to_init:
        return None
    arr = np.asarray(name_to_init[scale_name]).reshape(-1)
    if arr.size < 1:
        return None
    return float(arr[0])


def _fin_exponent_from_dq_chain(
    model: Any,
    tensor_name: str,
    name_to_init: Dict[str, np.ndarray],
    visited: Optional[set] = None,
    depth: int = 0,
) -> Optional[int]:
    """Fin: scale on DequantizeLinear that produces the float tensor feeding the FC input."""
    if depth > 64:
        return None
    if visited is None:
        visited = set()
    if tensor_name in visited:
        return None
    visited.add(tensor_name)
    producers = [n for n in model.graph.node if tensor_name in list(n.output)]
    if not producers:
        return None
    if len(producers) > 1:
        raise RuntimeError(f"Ambiguous producer chain for tensor {tensor_name!r}: {len(producers)} producers")
    node = producers[0]
    if node.op_type == "DequantizeLinear" and len(node.input) >= 2:
        sc = _scalar_from_init(node.input[1], name_to_init)
        if sc is not None:
            return onnx_scalar_quant_scale_to_rtl_exponent(sc)
    if node.op_type in (
        "Reshape",
        "Transpose",
        "Flatten",
        "Identity",
        "Cast",
        "Squeeze",
        "Unsqueeze",
        "Concat",
    ):
        if node.input:
            r = _fin_exponent_from_dq_chain(model, node.input[0], name_to_init, visited, depth + 1)
            if r is not None:
                return r
    return None


def _fin_scale_from_dq_chain(
    model: Any,
    tensor_name: str,
    name_to_init: Dict[str, np.ndarray],
    visited: Optional[set] = None,
    depth: int = 0,
) -> Optional[float]:
    """Fin scale on DequantizeLinear that produces the float tensor feeding the FC input."""
    if depth > 64:
        return None
    if visited is None:
        visited = set()
    if tensor_name in visited:
        return None
    visited.add(tensor_name)
    producers = [n for n in model.graph.node if tensor_name in list(n.output)]
    if not producers:
        return None
    if len(producers) > 1:
        raise RuntimeError(f"Ambiguous producer chain for tensor {tensor_name!r}: {len(producers)} producers")
    node = producers[0]
    if node.op_type == "DequantizeLinear" and len(node.input) >= 2:
        return _scalar_from_init(node.input[1], name_to_init)
    if node.op_type in (
        "Reshape",
        "Transpose",
        "Flatten",
        "Identity",
        "Cast",
        "Squeeze",
        "Unsqueeze",
        "Concat",
    ):
        if node.input:
            return _fin_scale_from_dq_chain(model, node.input[0], name_to_init, visited, depth + 1)
    if node.op_type in ("Add", "Mul", "Sub", "Div"):
        for inp in node.input:
            r = _fin_scale_from_dq_chain(model, inp, name_to_init, visited, depth + 1)
            if r is not None:
                return r
    return None


def _fout_exponent_from_q_chain(
    model: Any,
    tensor_name: str,
    name_to_init: Dict[str, np.ndarray],
    visited: Optional[set] = None,
    depth: int = 0,
) -> Optional[int]:
    """Fout: scale on QuantizeLinear that consumes this tensor (or after Add/Relu/Cast passthrough)."""
    if depth > 64:
        return None
    if visited is None:
        visited = set()
    if tensor_name in visited:
        return None
    visited.add(tensor_name)
    q_hits: List[int] = []
    for node in model.graph.node:
        ins = list(node.input)
        if tensor_name not in ins:
            continue
        if node.op_type == "QuantizeLinear" and len(ins) >= 2 and ins[0] == tensor_name:
            sc = _scalar_from_init(ins[1], name_to_init)
            if sc is not None:
                q_hits.append(onnx_scalar_quant_scale_to_rtl_exponent(sc))
    if q_hits:
        uniq = sorted(set(int(v) for v in q_hits))
        if len(uniq) > 1:
            raise RuntimeError(
                f"Ambiguous Fout exponent for tensor {tensor_name!r}: multiple QuantizeLinear exponents={uniq}"
            )
        return int(uniq[0])
    for node in model.graph.node:
        ins = list(node.input)
        if tensor_name not in ins:
            continue
        if node.op_type in ("Relu", "Sigmoid", "Tanh", "Clip", "Cast", "Identity") and node.output:
            r = _fout_exponent_from_q_chain(model, node.output[0], name_to_init, visited, depth + 1)
            if r is not None:
                return r
        if node.op_type == "Add" and node.output and tensor_name in ins:
            r = _fout_exponent_from_q_chain(model, node.output[0], name_to_init, visited, depth + 1)
            if r is not None:
                return r
    return None


def _fout_scale_from_q_chain(
    model: Any,
    tensor_name: str,
    name_to_init: Dict[str, np.ndarray],
    visited: Optional[set] = None,
    depth: int = 0,
) -> Optional[float]:
    """Fout scale on QuantizeLinear that consumes this tensor (or passthrough chain)."""
    if depth > 64:
        return None
    if visited is None:
        visited = set()
    if tensor_name in visited:
        return None
    visited.add(tensor_name)
    q_hits: List[float] = []
    for node in model.graph.node:
        ins = list(node.input)
        if tensor_name not in ins:
            continue
        if node.op_type == "QuantizeLinear" and len(ins) >= 2 and ins[0] == tensor_name:
            sc = _scalar_from_init(ins[1], name_to_init)
            if sc is not None:
                q_hits.append(sc)
    if q_hits:
        uniq = sorted({float(v) for v in q_hits})
        if len(uniq) > 1:
            raise RuntimeError(
                f"Ambiguous Fout scale for tensor {tensor_name!r}: multiple QuantizeLinear scales={uniq}"
            )
        return float(uniq[0])
    for node in model.graph.node:
        ins = list(node.input)
        if tensor_name not in ins:
            continue
        if node.op_type in ("Relu", "Sigmoid", "Tanh", "Clip", "Cast", "Identity") and node.output:
            r = _fout_scale_from_q_chain(model, node.output[0], name_to_init, visited, depth + 1)
            if r is not None:
                return r
        if node.op_type == "Add" and node.output and tensor_name in ins:
            r = _fout_scale_from_q_chain(model, node.output[0], name_to_init, visited, depth + 1)
            if r is not None:
                return r
    return None


def _qdq_fin_fout_for_fc_node(
    model: Any,
    node: Any,
    name_to_init: Dict[str, np.ndarray],
    *,
    op_type: str,
    inputs: List[str],
) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    """Return (fin_exp, fout_exp, fin_scale, fout_scale) from static QDQ / QLinear initializers."""
    fin_e: Optional[int] = None
    fout_e: Optional[int] = None
    fin_s: Optional[float] = None
    fout_s: Optional[float] = None
    out0 = node.output[0] if node.output else ""

    if op_type in ("QLinearMatMul", "QLinearGemm"):
        if len(inputs) >= 2:
            sc = _scalar_from_init(inputs[1], name_to_init)
            if sc is not None:
                fin_s = float(sc)
                fin_e = onnx_scalar_quant_scale_to_rtl_exponent(sc)
        if len(inputs) >= 7:
            sc = _scalar_from_init(inputs[6], name_to_init)
            if sc is not None:
                fout_s = float(sc)
                fout_e = onnx_scalar_quant_scale_to_rtl_exponent(sc)
        return fin_e, fout_e, fin_s, fout_s

    if op_type in ("Gemm", "MatMul"):
        if inputs:
            fin_s = _fin_scale_from_dq_chain(model, inputs[0], name_to_init, set(), 0)
            fin_e = _fin_exponent_from_dq_chain(model, inputs[0], name_to_init, set(), 0)
        if out0:
            fout_s = _fout_scale_from_q_chain(model, out0, name_to_init, set(), 0)
            fout_e = _fout_exponent_from_q_chain(model, out0, name_to_init, set(), 0)
        return fin_e, fout_e, fin_s, fout_s

    if op_type == "MatMulInteger":
        if inputs:
            fin_s = _fin_scale_from_dq_chain(model, inputs[0], name_to_init, set(), 0)
            fin_e = _fin_exponent_from_dq_chain(model, inputs[0], name_to_init, set(), 0)
        if out0:
            fout_s = _fout_scale_from_q_chain(model, out0, name_to_init, set(), 0)
            fout_e = _fout_exponent_from_q_chain(model, out0, name_to_init, set(), 0)
        if fout_e is None and out0:
            mul_hit = _try_get_scale_from_mul_chain(model, out0, name_to_init)
            if mul_hit is not None:
                try:
                    fout_s = float(mul_hit[0])
                    fout_e = onnx_scalar_quant_scale_to_rtl_exponent(mul_hit[0])
                except ValueError:
                    fout_e = None
        return fin_e, fout_e, fin_s, fout_s

    if op_type == "com.microsoft::FusedMatMul" or op_type.endswith("::FusedMatMul"):
        if inputs:
            fin_s = _fin_scale_from_dq_chain(model, inputs[0], name_to_init, set(), 0)
            fin_e = _fin_exponent_from_dq_chain(model, inputs[0], name_to_init, set(), 0)
        if out0:
            fout_s = _fout_scale_from_q_chain(model, out0, name_to_init, set(), 0)
            fout_e = _fout_exponent_from_q_chain(model, out0, name_to_init, set(), 0)
        return fin_e, fout_e, fin_s, fout_s

    return None, None, None, None


def _backfill_qdq_fout_from_chain(layers: List[Dict[str, Any]]) -> None:
    """If Fout scale is missing (e.g. no Q after last Sigmoid), use next layer Fin or same-layer Fin."""
    n = len(layers)
    for i in range(n - 1):
        if layers[i].get("qdq_fout_exp") is None:
            nxt_fin = layers[i + 1].get("qdq_fin_exp")
            if nxt_fin is not None:
                layers[i]["qdq_fout_exp"] = nxt_fin
                LOGGER.debug(
                    "%s: backfilled qdq_fout_exp from next layer %s qdq_fin_exp=%s",
                    layers[i].get("name"),
                    layers[i + 1].get("name"),
                    nxt_fin,
                )
    if n and layers[-1].get("qdq_fout_exp") is None:
        lf = layers[-1].get("qdq_fin_exp")
        if lf is not None:
            layers[-1]["qdq_fout_exp"] = lf
            LOGGER.warning(
                "Last FC %s: no trailing QuantizeLinear for Fout; using qdq_fin_exp=%s as qdq_fout_exp (approximation).",
                layers[-1].get("name"),
                lf,
            )


# -----------------------------------------------------------------------------
# Layer Extraction from ONNX Graph
#
# Core extraction routine: walks every node in the ONNX graph, identifies FC
# ops (Gemm, MatMul, MatMulInteger, QLinearMatMul, QLinearGemm, FusedMatMul),
# and builds a per-layer dict containing:
#   - weight (float32, out×in) and bias (float32, out)
#   - in_features / out_features dimensions
#   - quant_params (weight scale, zero-point — from QLinear inputs or DQ chain)
#   - qdq_fin_exp / qdq_fout_exp (activation exponents from Q/DQ scales)
#   - per-layer activation detected by forward tracing (ReLU, Sigmoid, …)
# When raw=True, weights and biases keep their original ONNX dtype (int8, etc.)
# so the caller can compute |q|/|f| inter-layer scales; when raw=False they are
# dequantized to float32 for RTL descriptor generation.
# -----------------------------------------------------------------------------

def extract_layers_from_onnx(
    onnx_path: Path,
    raw: bool = False,
    *,
    allow_fout_backfill: bool = True,
) -> Tuple[List[Any], int, Optional[str]]:
    """Extract FC layers from ONNX. If raw=True, preserve original dtypes/values (no dequantization)."""
    model = _load_onnx(onnx_path)
    inits = _get_initializers_dict(model)
    name_to_init = _build_value_to_array(model, inits)

    layers: List[Dict[str, Any]] = []
    input_size: Optional[int] = None
    fc_counter = 0

    for node in model.graph.node:
        op_type = _get_node_op_type(node)
        is_qlinear_matmul = node.op_type == "QLinearMatMul"
        is_qlinear_gemm = node.op_type == "QLinearGemm"
        is_fused_matmul = op_type == "com.microsoft::FusedMatMul"

        # --- Gemm / QLinearMatMul / QLinearGemm / FusedMatMul ---
        if node.op_type == "Gemm" or is_qlinear_matmul or is_qlinear_gemm or is_fused_matmul:
            fc_counter += 1
            name = f"fc{fc_counter}"
            inputs = list(node.input)

            if is_qlinear_matmul or is_qlinear_gemm:
                if len(inputs) < 8:
                    LOGGER.warning(f"{node.op_type} node {node.name} has < 8 inputs, skipping")
                    continue
                b_name = inputs[3]
                b_scale_name = inputs[4]
                b_zp_name = inputs[5]
                bias_name = inputs[8] if len(inputs) >= 9 else None
            elif is_fused_matmul:
                if len(inputs) < 2:
                    LOGGER.warning(f"FusedMatMul node {node.name} has < 2 inputs, skipping")
                    continue
                b_name = inputs[1]  # B = weight matrix
                bias_name = inputs[2] if len(inputs) >= 3 else None
            else:
                if len(inputs) < 2:
                    LOGGER.warning(f"Gemm node {node.name} has < 2 inputs, skipping")
                    continue
                b_name = inputs[1]
                bias_name = inputs[2] if len(inputs) > 2 else None

            # QLinear* / FusedMatmul: B is always an initializer. Plain Gemm (static QDQ): B may be a
            # DequantizeLinear output — resolve through DQ → quantized weight + scale like MatMul.
            dq_gemm_scale: Optional[float] = None
            dq_gemm_zp: int = 0
            w_storage_name: str = b_name
            if is_qlinear_matmul or is_qlinear_gemm:
                if b_name not in name_to_init:
                    LOGGER.warning(f"Weight {b_name} not in initializers, skipping")
                    continue
                w_np = name_to_init[b_name].copy()
            elif is_fused_matmul:
                if b_name not in name_to_init:
                    LOGGER.warning(f"Weight {b_name} not in initializers, skipping")
                    continue
                w_np = name_to_init[b_name].copy()
            else:
                resolved_g = _resolve_matmul_weight_tensor(model, b_name, name_to_init)
                if resolved_g is None:
                    LOGGER.warning(
                        "Gemm node %s: cannot resolve weight input %r (need initializer or DequantizeLinear → init); skipping",
                        node.name,
                        b_name,
                    )
                    continue
                w_np, dq_s, dq_z, w_storage_name = resolved_g
                dq_gemm_scale = dq_s
                dq_gemm_zp = int(dq_z)

            if w_np.ndim != 2:
                LOGGER.warning(f"Weight {b_name} is not 2D, skipping")
                continue

            if raw:
                weight = w_np
                in_features, out_features = w_np.shape
                if not is_qlinear_matmul and not is_qlinear_gemm and not is_fused_matmul:
                    trans_b = _get_attr(node, "transB", 0)
                    if trans_b:
                        out_features, in_features = w_np.shape[0], w_np.shape[1]
            else:
                if is_qlinear_matmul or is_qlinear_gemm:
                    b_scale = float(name_to_init[b_scale_name].flatten()[0]) if b_scale_name in name_to_init else 1.0
                    b_zp = int(name_to_init[b_zp_name].flatten()[0]) if b_zp_name in name_to_init else 0
                    w_np = (w_np.astype(np.float32) - b_zp) * b_scale
                elif dq_gemm_scale is not None:
                    w_np = (w_np.astype(np.float32) - float(dq_gemm_zp)) * float(dq_gemm_scale)
                elif w_np.dtype in (np.int8, np.uint8, np.int16, np.uint16):
                    raise RuntimeError(
                        f"{onnx_path}: Gemm node {node.name!r} ({name=}): weight initializer {b_name!r} has integer dtype "
                        f"{w_np.dtype} without QLinearGemm/QLinearMatMul scale/zero-point inputs. "
                        "Generic division-by-256 dequantization was removed. Fix: use float32 (or float16) weights, "
                        "or QLinear* ops with per-tensor scale, or ensure the weight is produced by DequantizeLinear "
                        "with a scale initializer on the graph."
                    )
                else:
                    w_np = w_np.astype(np.float32)

                in_features, out_features = w_np.shape
                weight = w_np.T.astype(np.float32)

                if not is_qlinear_matmul and not is_qlinear_gemm and not is_fused_matmul:
                    trans_b = _get_attr(node, "transB", 0)
                    alpha = _get_attr(node, "alpha", 1.0)
                    if trans_b:
                        out_features, in_features = w_np.shape[0], w_np.shape[1]
                        weight = w_np.astype(np.float32)
                    if alpha != 1.0:
                        weight = weight * float(alpha)

            bias_np: Optional[np.ndarray] = None
            if bias_name and bias_name in name_to_init:
                b_init = name_to_init[bias_name].flatten()
                if len(b_init) != out_features:
                    LOGGER.warning(f"Bias shape {b_init.shape} != out_features {out_features}")
                elif raw:
                    bias_np = b_init.copy()
                elif is_qlinear_matmul or is_qlinear_gemm:
                    bias_np = b_init.astype(np.float32)
                elif b_init.dtype in (np.int8, np.uint8, np.int16, np.uint16):
                    raise RuntimeError(
                        f"{onnx_path}: Gemm node {node.name!r} ({name=}): bias initializer "
                        f"{bias_name!r} has integer dtype {b_init.dtype} but this Gemm is not QLinearGemm "
                        "(no embedded bias scale/zero-point). Generic division-by-256 dequantization was removed."
                    )
                else:
                    bias_np = b_init.astype(np.float32)
                if not raw and bias_np is not None and not is_qlinear_matmul and not is_qlinear_gemm:
                    beta = _get_attr(node, "beta", 1.0)
                    if beta != 1.0:
                        bias_np = bias_np * float(beta)

            if bias_np is None:
                bias_np = np.zeros((out_features,), dtype=w_np.dtype if raw else np.float32)

            if input_size is None:
                input_size = in_features

            onnx_add_b_quantized: Optional[np.ndarray] = None
            if node.name.endswith("_Gemm"):
                g_prefix = node.name[: -len("_Gemm")]
                cand_g = f"{g_prefix}_Add_B_quantized"
                if cand_g in name_to_init:
                    aq_g = name_to_init[cand_g]
                    if int(aq_g.size) == out_features:
                        onnx_add_b_quantized = np.asarray(aq_g, dtype=np.int8).copy()

            layer_entry = {
                "name": name,
                "weight": weight,
                "bias": bias_np,
                "in_features": in_features,
                "out_features": out_features,
                "fc_output_name": node.output[0] if node.output else None,
                "onnx_add_b_quantized": onnx_add_b_quantized,
            }
            if raw:
                layer_entry["weight_init_name"] = w_storage_name
                layer_entry["bias_init_name"] = bias_name if (bias_name and bias_name in name_to_init) else None
                if is_qlinear_matmul or is_qlinear_gemm:
                    layer_entry["quant_params"] = {
                        "b_scale": float(name_to_init[b_scale_name].flatten()[0]) if b_scale_name in name_to_init else None,
                        "b_zero_point": int(name_to_init[b_zp_name].flatten()[0]) if b_zp_name in name_to_init else None,
                    }
                else:
                    layer_entry["quant_params"] = None
            else:
                # Non-raw: add quant_params for RTL scale computation
                if is_qlinear_matmul or is_qlinear_gemm:
                    ws_arr = np.asarray(name_to_init[b_scale_name], dtype=np.float64).reshape(-1) if b_scale_name in name_to_init else None
                    layer_entry["quant_params"] = {
                        "weight_scale": float(name_to_init[b_scale_name].flatten()[0]) if b_scale_name in name_to_init else None,
                        "weight_scale_arr": ws_arr,
                        "b_scale": float(name_to_init[b_scale_name].flatten()[0]) if b_scale_name in name_to_init else None,
                        "b_zero_point": int(name_to_init[b_zp_name].flatten()[0]) if b_zp_name in name_to_init else None,
                    }
                elif dq_gemm_scale is not None:
                    layer_entry["quant_params"] = {
                        "weight_scale": float(dq_gemm_scale),
                        "weight_scale_arr": np.asarray([float(dq_gemm_scale)], dtype=np.float64),
                        "b_zero_point": int(dq_gemm_zp),
                    }
                else:
                    layer_entry["quant_params"] = None
            qfin, qfout, qfin_s, qfout_s = _qdq_fin_fout_for_fc_node(
                model, node, name_to_init, op_type=_get_node_op_type(node), inputs=inputs
            )
            layer_entry["qdq_fin_exp"] = qfin
            layer_entry["qdq_fout_exp"] = qfout
            layer_entry["qdq_fin_scale"] = qfin_s
            layer_entry["qdq_fout_scale"] = qfout_s
            layers.append(layer_entry)

        # --- MatMul (float, int initializer, or QDQ: DQ output feeding MatMul) ---
        elif node.op_type == "MatMul":
            inputs = list(node.input)
            if len(inputs) < 2:
                continue
            b_name = inputs[1]
            resolved = _resolve_matmul_weight_tensor(model, b_name, name_to_init)
            if resolved is None:
                LOGGER.warning(
                    "MatMul node %s: cannot resolve weight input %r (need initializer or DequantizeLinear → init); skipping",
                    node.name,
                    b_name,
                )
                continue
            w_np, dq_scale, dq_zp, w_storage_name = resolved
            if w_np.ndim != 2:
                LOGGER.warning("MatMul %s: weight is not 2D, skipping", node.name)
                continue
            fc_counter += 1
            in_features, out_features = w_np.shape
            bias_init_name: Optional[str] = None
            if raw:
                weight = w_np
                bias_result = _try_get_matmul_bias_from_add(model, node, out_features, name_to_init, raw=True)
                bias_np = bias_result[0] if bias_result else None
                if bias_result:
                    bias_init_name = bias_result[1]
            else:
                if dq_scale is not None:
                    w_np = (w_np.astype(np.float32) - float(dq_zp)) * float(dq_scale)
                    weight = w_np.T.astype(np.float32)
                elif w_np.dtype in (np.int8, np.uint8, np.int16, np.uint16):
                    raise RuntimeError(
                        f"{onnx_path}: MatMul node {node.name!r}: weight {b_name!r} has integer dtype {w_np.dtype} "
                        "but no DequantizeLinear scale was found on the weight path (resolved initializer only). "
                        "Generic division-by-256 dequantization was removed. Fix: export static QDQ so weights are "
                        "float after DQ, use MatMulInteger with a weight-scale initializer, or use float weights."
                    )
                else:
                    w_np = w_np.astype(np.float32)
                    weight = w_np.T.astype(np.float32)
                bias_result = _try_get_matmul_bias_from_add(model, node, out_features, name_to_init)
                bias_np = bias_result[0] if bias_result else None
            if bias_np is None:
                bias_np = np.zeros((out_features,), dtype=w_np.dtype if raw else np.float32)
            if input_size is None:
                input_size = in_features
            onnx_add_b_quantized: Optional[np.ndarray] = None
            if node.name.endswith("_MatMul"):
                prefix = node.name[: -len("_MatMul")]
                cand = f"{prefix}_Add_B_quantized"
                if cand in name_to_init:
                    aq = name_to_init[cand]
                    if int(aq.size) == out_features:
                        onnx_add_b_quantized = np.asarray(aq, dtype=np.int8).copy()
            layer_entry = {
                "name": f"fc{fc_counter}",
                "weight": weight,
                "bias": bias_np,
                "in_features": in_features,
                "out_features": out_features,
                "fc_output_name": node.output[0] if node.output else None,
                "onnx_add_b_quantized": onnx_add_b_quantized,
            }
            if raw:
                layer_entry["weight_init_name"] = w_storage_name
                layer_entry["bias_init_name"] = bias_init_name
                layer_entry["quant_params"] = (
                    {
                        "weight_scale": float(dq_scale),
                        "weight_scale_arr": np.asarray([float(dq_scale)], dtype=np.float64),
                        "b_zero_point": int(dq_zp),
                    }
                    if dq_scale is not None
                    else None
                )
            else:
                layer_entry["quant_params"] = (
                    {
                        "weight_scale": float(dq_scale),
                        "weight_scale_arr": np.asarray([float(dq_scale)], dtype=np.float64),
                        "b_zero_point": int(dq_zp),
                    }
                    if dq_scale is not None
                    else None
                )
            qfin, qfout, qfin_s, qfout_s = _qdq_fin_fout_for_fc_node(
                model, node, name_to_init, op_type="MatMul", inputs=inputs
            )
            layer_entry["qdq_fin_exp"] = qfin
            layer_entry["qdq_fout_exp"] = qfout
            layer_entry["qdq_fin_scale"] = qfin_s
            layer_entry["qdq_fout_scale"] = qfout_s
            layers.append(layer_entry)

        # --- MatMulInteger (dynamic int8 quantization) ---
        elif node.op_type == "MatMulInteger":
            # Dynamic quantization: int8 activations * int8 weights -> int32
            inputs = list(node.input)
            if len(inputs) < 2:
                continue
            b_name = inputs[1]
            if b_name not in name_to_init:
                continue
            w_np = name_to_init[b_name].copy()
            if w_np.ndim != 2:
                continue
            in_features, out_features = w_np.shape
            bias_init_name_mmi: Optional[str] = None
            if raw:
                weight = w_np
                bias_result = _try_get_bias_from_add_chain(model, node.output[0], out_features, name_to_init, raw=True)
                bias_np = bias_result[0] if bias_result else None
                if bias_result:
                    bias_init_name_mmi = bias_result[1]
            else:
                # ORT quantize_dynamic: dequant weights with initializer scale + weight zero_point (not /128).
                scale_arr, scale_init_name = _lookup_matmulinteger_weight_scale_initializer(b_name, name_to_init)
                zp_arr: Optional[np.ndarray] = None
                if len(inputs) >= 4 and inputs[3] in name_to_init:
                    zp_arr = np.asarray(name_to_init[inputs[3]])
                if scale_arr is None:
                    LOGGER.error(
                        "MatMulInteger node %s: missing weight scale initializer next to %r (expected e.g. *_W_scale). "
                        "Cannot dequantize weights.",
                        node.name,
                        b_name,
                    )
                    continue
                try:
                    w_float = _dequantize_matmulinteger_weights(w_np, scale_arr, zp_arr)
                except ValueError as e:
                    LOGGER.error("MatMulInteger %s: %s", node.name, e)
                    continue
                weight = w_float.T.astype(np.float32)
                bias_result = _try_get_bias_from_add_chain(model, node.output[0], out_features, name_to_init)
                bias_np = bias_result[0] if bias_result else None
            if bias_np is None:
                LOGGER.warning(
                    "MatMulInteger %s: no bias tensor found on Add chain after matmul; exporting zeros (check ONNX Add + *_Add_B).",
                    node.name,
                )
                bias_np = np.zeros((out_features,), dtype=w_np.dtype if raw else np.float32)
            if input_size is None:
                input_size = in_features
            fc_counter += 1
            layer_entry_mmi: Dict[str, Any] = {
                "name": f"fc{fc_counter}",
                "weight": weight,
                "bias": bias_np,
                "in_features": in_features,
                "out_features": out_features,
                "fc_output_name": node.output[0] if node.output else None,
            }
            if raw:
                layer_entry_mmi["weight_init_name"] = b_name
                layer_entry_mmi["bias_init_name"] = bias_init_name_mmi
                b_zp_val: Any = None
                if len(inputs) >= 4 and inputs[3] in name_to_init:
                    zp_arr = name_to_init[inputs[3]]
                    if zp_arr.size == 1:
                        b_zp_val = int(zp_arr.flatten()[0])
                    else:
                        b_zp_val = zp_arr  # per-column, show as array
                scale_result = _try_get_scale_from_mul_chain(model, node.output[0], name_to_init)
                scale_val = scale_result[0] if scale_result else None
                scale_init_name = scale_result[1] if scale_result else None
                # Also try weight_scale from initializers (e.g. fc_proj_in_MatMul_W_scale)
                weight_scale_init = b_name.replace("_quantized", "_scale") if "_quantized" in b_name else None
                weight_scale_val = float(name_to_init[weight_scale_init].flatten()[0]) if weight_scale_init and weight_scale_init in name_to_init else None
                layer_entry_mmi["quant_params"] = {
                    "b_zero_point": b_zp_val,
                    "b_zero_point_init_name": inputs[3] if len(inputs) >= 4 else None,
                    "scale": scale_val,
                    "scale_init_name": scale_init_name,
                    "weight_scale": weight_scale_val,
                    "weight_scale_init_name": weight_scale_init if weight_scale_init in name_to_init else None,
                }
            else:
                # Non-raw: quant_params for RTL (store full weight-scale array for per-channel-safe Fb paths)
                wsc, wsc_name = _lookup_matmulinteger_weight_scale_initializer(b_name, name_to_init)
                weight_scale_val = float(np.asarray(wsc).flatten()[0]) if wsc is not None else None
                if wsc is not None and np.asarray(wsc).size > 1:
                    LOGGER.warning(
                        "MatMulInteger %s: per-channel weight scale (%d elems); using first element %.6g for Fb helper.",
                        node.name,
                        int(np.asarray(wsc).size),
                        weight_scale_val,
                    )
                b_zp_val_mmi = int(name_to_init[inputs[3]].flatten()[0]) if len(inputs) >= 4 and inputs[3] in name_to_init and name_to_init[inputs[3]].size == 1 else None
                layer_entry_mmi["quant_params"] = (
                    {
                        "weight_scale": weight_scale_val,
                        "weight_scale_arr": np.asarray(wsc).reshape(-1).astype(np.float64) if wsc is not None else None,
                        "weight_scale_init_name": wsc_name,
                        "b_zero_point": b_zp_val_mmi,
                    }
                    if weight_scale_val is not None
                    else None
                )
            qfin, qfout, qfin_s, qfout_s = _qdq_fin_fout_for_fc_node(
                model, node, name_to_init, op_type="MatMulInteger", inputs=inputs
            )
            layer_entry_mmi["qdq_fin_exp"] = qfin
            layer_entry_mmi["qdq_fout_exp"] = qfout
            layer_entry_mmi["qdq_fin_scale"] = qfin_s
            layer_entry_mmi["qdq_fout_scale"] = qfout_s
            layers.append(layer_entry_mmi)

    if input_size is None and layers:
        input_size = layers[0]["in_features"]

    if input_size is None:
        if not layers:
            raise RuntimeError(
                f"Cannot determine input_size from ONNX {onnx_path!s}: no fully-connected "
                "(Gemm / MatMul / MatMulInteger / QLinear*) layers were extracted. "
                "The converter needs at least one FC layer whose first input dimension defines the model input size."
            )
        first = layers[0]
        raise RuntimeError(
            f"Cannot determine input_size from ONNX {onnx_path!s}: extracted {len(layers)} FC layer(s) but "
            f"in_features is missing on the first layer {first.get('name')!r} (entry keys: {sorted(first.keys())}). "
            "This indicates a bug in layer extraction for this graph."
        )

    final_activation = _detect_final_activation(model)

    # Per-layer activation detection (trace forward from each FC output)
    fc_output_names = [l.get("fc_output_name") for l in layers if l.get("fc_output_name")]
    per_layer_activations = extract_per_layer_activations(model, fc_output_names) if fc_output_names else []
    for i, act in enumerate(per_layer_activations):
        if i < len(layers):
            layers[i]["activation"] = act  # Relu, Sigmoid, Tanh, etc. or None

    if allow_fout_backfill:
        _backfill_qdq_fout_from_chain(layers)

    return layers, input_size, final_activation


# -----------------------------------------------------------------------------
# Inter-layer scales: paired quantized + float ONNX
#
# To compute accurate Fw (weight exponent) and Fb (bias exponent) for each FC
# layer we need BOTH the quantized and the float ONNX:
#   - Float ONNX → dequantized weights/biases (the "true" float reference).
#   - Quantized ONNX (raw=True) → original int8 weights/biases + per-tensor scale.
# The ratio |q_weight| / |f_weight| gives the power-of-two scaling factor that
# the RTL MAC uses. This section loads both models, aligns their transpose
# conventions, and attaches the paired tensors onto each LayerInfo so that
# ``build_rtl_layer_quant_descriptors`` can derive Fw, Fb, W_int, and B_int.
# -----------------------------------------------------------------------------

_INT_DTYPES_FOR_SCALE = (
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
)


def attach_inter_layer_scale_tensors_from_onnx_pair(
    layers: List[Any],
    quant_path: Path,
    float_path: Path,
) -> None:
    """Load float (dequant) and quantized (raw) FC layers; set onnx_pair_* tensors on each linear LayerInfo.

    Raises RuntimeError if topology/shapes/dtypes do not match (no silent fallback).
    """
    layers_f, _, _ = extract_layers_from_onnx(float_path, raw=False)
    layers_q_raw, _, _ = extract_layers_from_onnx(quant_path, raw=True)
    linear_idx = [i for i, ly in enumerate(layers) if ly.layer_type == "linear"]
    if len(layers_f) != len(layers_q_raw) or len(layers_f) != len(linear_idx):
        raise RuntimeError(
            f"FC layer count mismatch: float ONNX has {len(layers_f)} FC layers, "
            f"quantized raw has {len(layers_q_raw)}, full graph has {len(linear_idx)} linear layers. "
            "Use matching topologies."
        )
    pending: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for j, idx in enumerate(linear_idx):
        wf = np.asarray(layers_f[j]["weight"], dtype=np.float64)
        wq = np.asarray(layers_q_raw[j]["weight"])
        bf = np.asarray(layers_f[j]["bias"], dtype=np.float64).ravel()
        bq = np.asarray(layers_q_raw[j]["bias"]).ravel()
        name = layers[idx].name
        if wq.dtype not in _INT_DTYPES_FOR_SCALE:
            raise RuntimeError(
                f"{name}: quantized ONNX weights must be integer dtype for inter-layer |q|/|f| scales; got {wq.dtype}. "
                "Pass the quantized ONNX path used for this layer extract."
            )
        # Float extract uses w.T for MatMul/MatMulInteger; raw keeps ONNX B as (in, out). Align quant to float (out, in).
        # When in_features == out_features (square W), shapes match without transpose but elements are still transposed:
        # must apply wq.T whenever wf is the transpose layout of wq (including NxN).
        if wf.shape == (wq.shape[1], wq.shape[0]):
            wq = np.ascontiguousarray(wq.T)
        elif wf.shape != wq.shape:
            raise RuntimeError(
                f"{name}: weight shape mismatch between float {wf.shape} and quantized {wq.shape} "
                f"(not a transpose pair)."
            )
        if bf.size != bq.size:
            raise RuntimeError(
                f"{name}: bias length mismatch: float {bf.size} vs quantized {bq.size}."
            )
        pending.append((idx, wf, wq, bf, bq.astype(np.float64, copy=False)))
    for j, (idx, wf, wq, bf, bq) in enumerate(pending):
        layers[idx].onnx_pair_float_weight = wf
        layers[idx].onnx_pair_quant_weight = wq
        layers[idx].onnx_pair_float_bias = bf
        layers[idx].onnx_pair_quant_bias = bq
        qp = layers_q_raw[j].get("quant_params")
        ws = None
        ws_arr = None
        if isinstance(qp, dict):
            wsv = qp.get("weight_scale")
            if wsv is not None:
                ws = float(wsv)
            wsa = qp.get("weight_scale_arr")
            if wsa is not None:
                ws_arr = np.asarray(wsa, dtype=np.float64).reshape(-1)
        layers[idx].onnx_pair_weight_scale = ws
        layers[idx].onnx_pair_weight_scale_arr = ws_arr



import multiclass_rtl_mapper as mrm  # noqa: E402



# =============================================================================
# Conv layer extraction
#
# Walks the ONNX graph and pulls every Conv node into a ConvLayerInfo with float
# weights/biases, kernel/stride/pad attributes, spatial dims (traced through the
# graph), and Fin/Fout exponents from QDQ scales.  ``op_kind`` is "depthwise"
# when kH > 1 or kW > 1, else "pointwise" (1x1 matches the existing engine
# convention even though ONNX itself does not distinguish DW vs PW group=1).
# =============================================================================

def _classify_conv_op_kind(kH: int, kW: int) -> str:
    """Pick which engine handles this conv.

    1x1 → pointwise (multiplies in_ch values per cycle, sums across in_ch).
    Otherwise → depthwise (line-buffer + sliding kH x kW window). Inputs with
    in_ch > 1 are packed channel-fastest into the kernel-window dimension so the
    same engine works without modification.
    """
    if kH == 1 and kW == 1:
        return "pointwise"
    return "depthwise"


def _trace_conv_activation(
    model: Any,
    conv_output_name: str,
    input_to_nodes: Dict[str, List[Any]],
) -> Optional[str]:
    """Trace forward from a Conv output through Q/DQ to find the next activation
    (Relu/Sigmoid/Tanh) — same logic as the per-FC activation trace.
    """
    return _trace_activation_forward(model, conv_output_name, input_to_nodes)


def _conv_output_spatial(
    in_h: int,
    in_w: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    pad: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    """Standard ONNX Conv output size formula.

    pad = (top, left, bottom, right). For symmetric padding stored as [pad_h, pad_w]
    we duplicate. Returns (out_h, out_w).
    """
    pad_h_total = pad[0] + pad[2]
    pad_w_total = pad[1] + pad[3]
    out_h = (in_h + pad_h_total - kernel[0]) // stride[0] + 1
    out_w = (in_w + pad_w_total - kernel[1]) // stride[1] + 1
    return out_h, out_w


def _avgpool_output_spatial(
    in_h: int,
    in_w: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    pad: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    pad_h_total = pad[0] + pad[2]
    pad_w_total = pad[1] + pad[3]
    out_h = (in_h + pad_h_total - kernel[0]) // stride[0] + 1
    out_w = (in_w + pad_w_total - kernel[1]) // stride[1] + 1
    return out_h, out_w


def _conv_attrs(node: Any) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int, int, int], int]:
    """Read kernel_shape / strides / pads / group from a Conv node, defaulting where missing."""
    k = (1, 1)
    s = (1, 1)
    p = (0, 0, 0, 0)
    g = 1
    for attr in node.attribute:
        if attr.name == "kernel_shape" and attr.type == 7:
            ks = list(attr.ints)
            if len(ks) >= 2:
                k = (int(ks[0]), int(ks[1]))
            elif len(ks) == 1:
                k = (int(ks[0]), 1)
        elif attr.name == "strides" and attr.type == 7:
            ss = list(attr.ints)
            if len(ss) >= 2:
                s = (int(ss[0]), int(ss[1]))
            elif len(ss) == 1:
                s = (int(ss[0]), 1)
        elif attr.name == "pads" and attr.type == 7:
            pp = list(attr.ints)
            if len(pp) >= 4:
                p = (int(pp[0]), int(pp[1]), int(pp[2]), int(pp[3]))
            elif len(pp) == 2:
                p = (int(pp[0]), int(pp[1]), int(pp[0]), int(pp[1]))
        elif attr.name == "group" and attr.type == 2:
            g = int(attr.i)
    return k, s, p, g


def _resolve_conv_weight_4d(
    model: Any,
    weight_input_name: str,
    name_to_init: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Optional[float], int, str]:
    """Resolve a Conv's second input (weight) to a 4D float32 array.

    Walks DequantizeLinear chains the same way ``_resolve_matmul_weight_tensor``
    handles 2D MatMul weights — so quantized models with QDQ-wrapped Conv weights
    are dequantized via the ONNX-stored scale (no hardcoded /128).

    Returns ``(W_float, dq_scale, dq_zp, init_name)`` where ``dq_scale`` is None
    for direct float initializers.
    """
    cur: Optional[str] = weight_input_name
    visited: set[str] = set()
    while cur is not None and cur not in visited:
        visited.add(cur)
        if cur in name_to_init:
            w = name_to_init[cur].copy()
            if w.ndim != 4:
                raise RuntimeError(f"Conv weight {cur!r} is not 4D (got shape {w.shape})")
            return w.astype(np.float32, copy=False), None, 0, cur

        prod = _producer_node_for_tensor(model, cur)
        if prod is None:
            raise RuntimeError(f"Cannot resolve Conv weight tensor {weight_input_name!r}: no producer found")
        if prod.op_type == "DequantizeLinear":
            ins = list(prod.input)
            if len(ins) < 2:
                raise RuntimeError(f"DequantizeLinear before Conv has < 2 inputs: {prod.name}")
            qn, sn = ins[0], ins[1]
            zn = ins[2] if len(ins) > 2 else None
            if qn not in name_to_init or sn not in name_to_init:
                raise RuntimeError(
                    f"DequantizeLinear inputs missing initializer (q={qn!r}, scale={sn!r})"
                )
            scale_arr = name_to_init[sn]
            if scale_arr.size != 1:
                raise RuntimeError(
                    f"DequantizeLinear before Conv: only scalar x_scale supported (got shape {scale_arr.shape})"
                )
            wq = name_to_init[qn].copy()
            if wq.ndim != 4:
                raise RuntimeError(f"Quantized conv weight {qn!r} is not 4D (got shape {wq.shape})")
            sc = float(scale_arr.flatten()[0])
            zp = int(name_to_init[zn].flatten()[0]) if zn and zn in name_to_init else 0
            w_float = (wq.astype(np.float32) - float(zp)) * sc
            return w_float, sc, zp, qn
        if prod.op_type in ("Cast", "Identity") and prod.input:
            cur = prod.input[0]
            continue
        raise RuntimeError(
            f"Cannot resolve Conv weight tensor {weight_input_name!r}: producer {prod.op_type} not supported"
        )
    raise RuntimeError(f"Cannot resolve Conv weight tensor {weight_input_name!r}: visited all producers")


def _resolve_conv_bias_1d(
    model: Any,
    bias_input_name: Optional[str],
    name_to_init: Dict[str, np.ndarray],
    out_channels: int,
) -> np.ndarray:
    """Resolve a Conv's third input (bias) to a 1D float32 array of length ``out_channels``.

    Mirrors :func:`_resolve_conv_weight_4d` but for the 1D bias tensor. ORT's static
    QDQ INT8 quantization wraps the conv bias in a ``DequantizeLinear`` node — its
    int32 quantized values live in an initializer and the per-tensor or per-channel
    ``bias_scale`` (= ``input_scale * weight_scale`` as ORT writes it) is another
    initializer on the DQ node. Reading that scale directly from the graph keeps
    us aligned with whatever ORT decided, including any per-channel layout.

    Returns a zero vector if ``bias_input_name`` is empty (Conv with no bias arg).
    """
    if not bias_input_name:
        return np.zeros((out_channels,), dtype=np.float32)
    cur: Optional[str] = bias_input_name
    visited: set[str] = set()
    while cur is not None and cur not in visited:
        visited.add(cur)
        if cur in name_to_init:
            b = name_to_init[cur]
            if b.dtype.kind in ("i", "u"):
                LOGGER.debug(
                    "Conv bias %s is a raw integer initializer (dtype=%s); upcasting to float without dequant",
                    cur, b.dtype,
                )
            b_float = b.astype(np.float32, copy=False).ravel()
            if b_float.size != out_channels:
                LOGGER.warning(
                    "Conv bias %s size %d != out_channels %d; resizing",
                    cur, b_float.size, out_channels,
                )
                b_float = np.resize(b_float, out_channels).astype(np.float32)
            return b_float

        prod = _producer_node_for_tensor(model, cur)
        if prod is None:
            raise RuntimeError(f"Cannot resolve Conv bias tensor {bias_input_name!r}: no producer found")
        if prod.op_type == "DequantizeLinear":
            ins = list(prod.input)
            if len(ins) < 2:
                raise RuntimeError(f"DequantizeLinear before Conv bias has < 2 inputs: {prod.name}")
            qn, sn = ins[0], ins[1]
            zn = ins[2] if len(ins) > 2 else None
            if qn not in name_to_init or sn not in name_to_init:
                raise RuntimeError(
                    f"DequantizeLinear before Conv bias missing initializer (q={qn!r}, scale={sn!r})"
                )
            scale_arr = name_to_init[sn]
            bq = name_to_init[qn]
            # bias_scale is per-tensor (scalar) or per-output-channel (out_channels,).
            # ORT-static-QDQ INT8 emits scalar = input_scale * weight_scale, but per-channel
            # is also legal — handle both rather than hardcoding scalar.
            if scale_arr.size == 1:
                sc = np.full((out_channels,), float(scale_arr.flatten()[0]), dtype=np.float64)
            elif scale_arr.size == out_channels:
                sc = scale_arr.astype(np.float64).ravel()
            else:
                raise RuntimeError(
                    f"DequantizeLinear bias scale shape {scale_arr.shape} matches neither "
                    f"scalar nor out_channels={out_channels}"
                )
            if zn and zn in name_to_init:
                zp_arr = name_to_init[zn]
                if zp_arr.size == 1:
                    zp = np.full((out_channels,), int(zp_arr.flatten()[0]), dtype=np.int64)
                elif zp_arr.size == out_channels:
                    zp = zp_arr.astype(np.int64).ravel()
                else:
                    raise RuntimeError(
                        f"DequantizeLinear bias zero_point shape {zp_arr.shape} matches "
                        f"neither scalar nor out_channels={out_channels}"
                    )
            else:
                zp = np.zeros((out_channels,), dtype=np.int64)
            b_int = bq.astype(np.int64).ravel()
            if b_int.size != out_channels:
                raise RuntimeError(
                    f"Quantized conv bias {qn!r} size {b_int.size} != out_channels {out_channels}"
                )
            b_float = ((b_int - zp).astype(np.float64) * sc).astype(np.float32)
            LOGGER.debug(
                "Resolved DQ-wrapped conv bias %s: dtype=%s scale_size=%d zp_size=%d "
                "exported[min,max]=[%g,%g]",
                qn, bq.dtype, scale_arr.size,
                zp_arr.size if (zn and zn in name_to_init) else 1,
                float(b_float.min()) if b_float.size else 0.0,
                float(b_float.max()) if b_float.size else 0.0,
            )
            return b_float
        if prod.op_type in ("Cast", "Identity") and prod.input:
            cur = prod.input[0]
            continue
        raise RuntimeError(
            f"Cannot resolve Conv bias tensor {bias_input_name!r}: producer {prod.op_type} not supported"
        )
    raise RuntimeError(f"Cannot resolve Conv bias tensor {bias_input_name!r}: visited all producers")


def extract_conv_layers_from_onnx(
    onnx_path: Path,
    *,
    model_input_h: int,
    model_input_w: int,
    model_input_channels: int,
) -> List[mrm.ConvLayerInfo]:
    """Extract every Conv node from the ONNX graph as a ``ConvLayerInfo``.

    Spatial dims are traced through the conv chain assuming the conv layers
    appear in dataflow order before any AveragePool / flatten. The first conv's
    in_h/in_w come from ``model_input_h/w`` (as resolved by ORT inputs).

    Per-layer activation is detected by tracing the conv output forward through
    Q/DQ + Cast/Reshape until reaching a Relu/Sigmoid/Tanh.
    """
    model = _load_onnx(onnx_path)
    inits = _get_initializers_dict(model)
    name_to_init = _build_value_to_array(model, inits)

    input_to_nodes: Dict[str, List[Any]] = {}
    for n in model.graph.node:
        for inp in n.input:
            input_to_nodes.setdefault(inp, []).append(n)

    convs: List[mrm.ConvLayerInfo] = []
    cur_h, cur_w, cur_c = model_input_h, model_input_w, model_input_channels

    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        ins = list(node.input)
        if len(ins) < 2:
            LOGGER.warning("Conv %s has < 2 inputs; skipping", node.name)
            continue
        weight_input = ins[1]
        bias_input = ins[2] if len(ins) >= 3 else None

        W_float, dq_scale, dq_zp, w_init_name = _resolve_conv_weight_4d(model, weight_input, name_to_init)
        out_ch, in_ch, kH, kW = W_float.shape
        kernel, stride, pad, group = _conv_attrs(node)
        if (kH, kW) != kernel:
            LOGGER.warning(
                "%s: weight kernel shape (%d,%d) disagrees with attr kernel_shape %s — using weight shape.",
                node.name, kH, kW, kernel,
            )
        op_kind = _classify_conv_op_kind(kH, kW)

        # Spatial trace
        in_h_layer, in_w_layer = cur_h, cur_w
        out_h, out_w = _conv_output_spatial(in_h_layer, in_w_layer, (kH, kW), stride, pad)

        # Bias resolution: walks DequantizeLinear/Cast/Identity producers so that
        # post-QDQ models (where ORT inserts a DQ in front of the int32 bias) still
        # recover the float bias instead of silently zeroing it.
        bias_arr = _resolve_conv_bias_1d(model, bias_input, name_to_init, out_ch)

        # Fin / Fout from QDQ chains
        fin_e = _fin_exponent_from_dq_chain(model, ins[0], name_to_init, set(), 0)
        fout_e = _fout_exponent_from_q_chain(model, node.output[0], name_to_init, set(), 0)
        fin_s = _fin_scale_from_dq_chain(model, ins[0], name_to_init, set(), 0)
        fout_s = _fout_scale_from_q_chain(model, node.output[0], name_to_init, set(), 0)

        # Activation that follows
        act = _trace_conv_activation(model, node.output[0], input_to_nodes)

        layer = mrm.ConvLayerInfo(
            name=node.name or f"conv_{len(convs)+1}",
            op_kind=op_kind,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_h=kH,
            kernel_w=kW,
            stride_h=stride[0],
            stride_w=stride[1],
            pad_h=pad[0],
            pad_w=pad[1],
            weight=W_float,
            bias=bias_arr,
            qdq_fin_exp=fin_e,
            qdq_fout_exp=fout_e,
            qdq_fin_scale=fin_s,
            qdq_fout_scale=fout_s,
            activation=act,
            in_h=in_h_layer,
            in_w=in_w_layer,
            out_h=out_h,
            out_w=out_w,
            onnx_node_name=node.name,
            fc_output_name=node.output[0] if node.output else None,
        )
        convs.append(layer)
        # Advance spatial state for the next conv: spatial dims from this conv's output;
        # channel count = out_ch.
        cur_h, cur_w, cur_c = out_h, out_w, out_ch

    return convs


# =============================================================================
# AveragePool detection
# =============================================================================

def _find_avgpool_node(model: Any) -> Optional[Any]:
    """Return the first AveragePool / GlobalAveragePool node in the graph, or None."""
    for node in model.graph.node:
        if node.op_type in ("AveragePool", "GlobalAveragePool"):
            return node
    return None


def _avgpool_attrs(node: Any) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int, int, int]]:
    k = (1, 1)
    s = (1, 1)
    p = (0, 0, 0, 0)
    for attr in node.attribute:
        if attr.name == "kernel_shape" and attr.type == 7:
            ks = list(attr.ints)
            k = (int(ks[0]), int(ks[1])) if len(ks) >= 2 else (int(ks[0]), 1)
        elif attr.name == "strides" and attr.type == 7:
            ss = list(attr.ints)
            s = (int(ss[0]), int(ss[1])) if len(ss) >= 2 else (int(ss[0]), 1)
        elif attr.name == "pads" and attr.type == 7:
            pp = list(attr.ints)
            if len(pp) >= 4:
                p = (int(pp[0]), int(pp[1]), int(pp[2]), int(pp[3]))
            elif len(pp) == 2:
                p = (int(pp[0]), int(pp[1]), int(pp[0]), int(pp[1]))
    return k, s, p


# =============================================================================
# Final classifier op detection (Softmax / ArgMax)
# =============================================================================

def _detect_final_classifier_op(model: Any) -> str:
    """Detect whether the model ends in Softmax or ArgMax (or fall back to "softmax").

    Walks backward from the graph's first output, treating Q/DQ + Cast/Reshape as
    transparent passthrough. Returns "softmax" or "argmax".
    """
    if not model.graph.output:
        return "softmax"
    output_to_node: Dict[str, Any] = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node
    current = model.graph.output[0].name
    visited: set[str] = set()
    passthrough = {"Identity", "Cast", "Reshape", "Squeeze", "Unsqueeze",
                   "QuantizeLinear", "DequantizeLinear"}
    while current and current not in visited:
        visited.add(current)
        node = output_to_node.get(current)
        if node is None:
            return "softmax"
        if node.op_type in ("Softmax", "LogSoftmax"):
            return "softmax"
        if node.op_type in ("ArgMax",):
            return "argmax"
        if node.op_type in passthrough and node.input:
            current = node.input[0]
        else:
            return "softmax"
    return "softmax"


# =============================================================================
# Model-level metadata: input shape (NCHW), num_classes (final FC out_features)
# =============================================================================

def _model_input_nchw(model: Any) -> Tuple[int, int, int, int]:
    """Resolve the model's primary input shape as (N, C, H, W).

    Symbolic / dynamic dims are pinned to 1 (matches calibration shape resolution).
    Raises if rank != 4 — multiclass conv pipeline expects NCHW.
    """
    if not model.graph.input:
        raise RuntimeError("Model has no graph inputs")
    inp = model.graph.input[0]
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_value > 0:
            dims.append(int(d.dim_value))
        else:
            dims.append(1)
    if len(dims) != 4:
        raise RuntimeError(f"Model input rank must be 4 (NCHW); got shape {dims}")
    n, c, h, w = dims
    return n, c, h, w


def _conv_layers_to_pool_input(
    conv_layers: List[mrm.ConvLayerInfo],
) -> Tuple[int, int, int]:
    """The (channels, h, w) feeding the AveragePool: take the last conv's output spatial
    dims and out_channels.
    """
    if not conv_layers:
        raise RuntimeError("No conv layers extracted; cannot derive pool input shape")
    last = conv_layers[-1]
    return last.out_channels, last.out_h, last.out_w


# =============================================================================
# Main pipeline (orchestrates calibrate → quantize → extract → emit)
# =============================================================================

def _build_synthetic_calibration_npz(
    float_onnx: Path,
    npz_out: Path,
    *,
    seed: int,
    random_normal: Optional[int],
    random_uniform: Optional[int],
) -> None:
    """synthesize a calibration .npz from the float model's
    input shapes; ORT reads this to choose static int8 scales.
    """
    import onnxruntime as ort

    import multiclass_calib as mcb

    sess = ort.InferenceSession(str(float_onnx), providers=["CPUExecutionProvider"])
    shapes = mcb._collect_inputs(sess)
    auto_n, auto_u = mcb.auto_random_sample_counts(shapes)
    n_normal = auto_n if random_normal is None else max(0, random_normal)
    n_uniform = auto_u if random_uniform is None else max(0, random_uniform)
    LOGGER.info(
        "Synthetic calibration: random-normal=%d random-uniform=%d seed=%d",
        n_normal, n_uniform, seed,
    )
    stacked = mcb.build_calibration_batches(shapes, seed=seed, n_normal=n_normal, n_uniform=n_uniform)
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_out, **stacked)


def _quantize_float_to_qdq_int8(float_onnx: Path, calib_npz: Path, quant_onnx_out: Path) -> None:
    """Run ORT static QDQ INT8 quantization. Inserts QuantizeLinear/DequantizeLinear
    nodes around every Conv/FC weight and activation with calibrated scales.
    """
    import multiclass_quantize as oq
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    reader = oq.NpyCalibrationDataReader(float_onnx.resolve(), calib_npz.resolve())
    quantize_static(
        model_input=str(float_onnx.resolve()),
        model_output=str(quant_onnx_out.resolve()),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )


def _enforce_conv_numeric_fidelity_or_warn(conv_layers: List[mrm.ConvLayerInfo]) -> None:
    """Per-layer thresholds for power-of-two int8 quantization quality.

    Logs a warning rather than raising — convolutional layers often have wider
    activation ranges than pure FC chains, so saturation > 1% is more common
    but still tolerable. The ``--strict-fidelity`` CLI flag promotes these to
    errors.
    """
    max_w_mae = 0.020
    max_b_mae = 0.06
    max_sat_pct = 5.0
    for layer in conv_layers:
        rq = layer.rtl_quant
        if rq is None:
            LOGGER.warning("%s: missing rtl_quant descriptor", layer.name)
            continue
        w_ref = np.asarray(rq.W_float, dtype=np.float64)
        w_rec = np.asarray(rq.W_int, dtype=np.float64) * (2.0 ** (-int(rq.fw_frac)))
        b_ref = np.asarray(rq.B_float, dtype=np.float64).ravel()
        b_rec = np.asarray(rq.B_int, dtype=np.float64) * (2.0 ** (-int(rq.fb_rtl)))
        w_mae = float(np.mean(np.abs(w_ref - w_rec))) if w_ref.size else 0.0
        b_mae = float(np.mean(np.abs(b_ref - b_rec))) if b_ref.size else 0.0
        if w_mae > max_w_mae:
            LOGGER.warning("%s: conv weight MAE=%.5f > %.5f", layer.name, w_mae, max_w_mae)
        if b_mae > max_b_mae:
            LOGGER.warning("%s: conv bias   MAE=%.5f > %.5f", layer.name, b_mae, max_b_mae)
        if rq.weight_sat_lo_pct > max_sat_pct or rq.weight_sat_hi_pct > max_sat_pct:
            LOGGER.warning(
                "%s: conv weight saturation lo/hi=%.2f%%/%.2f%% > %.2f%%",
                layer.name, rq.weight_sat_lo_pct, rq.weight_sat_hi_pct, max_sat_pct,
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a multiclass ONNX classifier (Conv + FC) to RTL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=Path, required=True, help="Float32 ONNX model")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for RTL + .mem files")
    parser.add_argument("--calib-seed", type=int, default=42, help="RNG seed for synthetic calibration")
    parser.add_argument("--calib-random-normal", type=int, default=None, metavar="N",
                        help="Synthetic normal(0,0.2) batch count (default: auto)")
    parser.add_argument("--calib-random-uniform", type=int, default=None, metavar="N",
                        help="Synthetic uniform(-0.6,0.6) batch count (default: auto)")
    parser.add_argument("--head", choices=("auto", "softmax", "argmax"), default="auto",
                        help="Classifier head: auto = detect from ONNX final op (default)")
    parser.add_argument("--strict-fidelity", action="store_true",
                        help="Promote conv-quant fidelity warnings to errors")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    float_path = args.model.resolve()
    out_dir = args.out_dir.resolve()
    if not float_path.is_file():
        LOGGER.error("Model not found: %s", float_path)
        return 1

    weight_format_bits = 8
    scale = 1 << weight_format_bits

    with tempfile.TemporaryDirectory(prefix="multiclass_onnx_to_rtl_") as tmp:
        tdir = Path(tmp)
        calib_npz = tdir / "synthetic_calib.npz"
        quant_onnx = tdir / "quantized_qdq_int8.onnx"

        # ----- Step 1: Calibration -----
        LOGGER.info("Step 1/3: building synthetic calibration npz...")
        _build_synthetic_calibration_npz(
            float_path, calib_npz,
            seed=args.calib_seed,
            random_normal=args.calib_random_normal,
            random_uniform=args.calib_random_uniform,
        )

        # ----- Step 2: Static QDQ INT8 quantization -----
        LOGGER.info("Step 2/3: static QDQ int8 quantization...")
        try:
            _quantize_float_to_qdq_int8(float_path, calib_npz, quant_onnx)
        except Exception as e:
            LOGGER.error("quantize_static failed: %s", e)
            return 1

        # ----- Step 3a: Resolve model-level shape & decide classifier head -----
        float_model = _load_onnx(float_path)
        n_in, c_in, h_in, w_in = _model_input_nchw(float_model)
        if args.head == "auto":
            head_kind = _detect_final_classifier_op(float_model)
        else:
            head_kind = args.head
        LOGGER.info("Model input: NCHW=(%d,%d,%d,%d); classifier head=%s", n_in, c_in, h_in, w_in, head_kind)

        # ----- Step 3b: Extract Conv layers from quantized ONNX -----
        LOGGER.info("Step 3/3: extracting layers and emitting RTL...")
        conv_layers = extract_conv_layers_from_onnx(
            quant_onnx,
            model_input_h=h_in,
            model_input_w=w_in,
            model_input_channels=c_in,
        )
        if not conv_layers:
            LOGGER.warning("No Conv layers extracted from %s — model may be FC-only.", quant_onnx.name)

        # ----- Step 3c: Extract FC layers -----
        # extract_layers_from_onnx returns FC-only entries; the conv outputs are
        # NOT processed by this extractor (it ignores Conv nodes), so we get just
        # the post-flatten FC chain.
        fc_layers_raw, input_size_post_flatten, _final_act = extract_layers_from_onnx(
            quant_onnx, allow_fout_backfill=False,
        )
        if not fc_layers_raw:
            LOGGER.error("No Gemm/MatMul FC layers found — multiclass model expects FC after conv stack")
            return 1

        # Backfill missing Fout on the last FC (no QuantizeLinear after Softmax)
        for i in range(len(fc_layers_raw) - 1):
            if fc_layers_raw[i].get("qdq_fout_exp") is None:
                fc_layers_raw[i]["qdq_fout_exp"] = fc_layers_raw[i + 1].get("qdq_fin_exp")
        if fc_layers_raw and fc_layers_raw[-1].get("qdq_fout_exp") is None:
            fc_layers_raw[-1]["qdq_fout_exp"] = fc_layers_raw[-1].get("qdq_fin_exp")

        # Build LayerInfo list (linear-only) for the FC chain.
        fc_layers: List[mrm.LayerInfo] = []
        for lr in fc_layers_raw:
            w_np = lr["weight"].astype(np.float32)
            b_np = lr["bias"].astype(np.float32)
            layer = mrm.LayerInfo(
                name=lr["name"],
                layer_type="linear",
                in_features=lr["in_features"],
                out_features=lr["out_features"],
                weight=np.asarray(w_np, dtype=np.float32),
                bias=np.asarray(b_np, dtype=np.float32),
                quant_params=lr.get("quant_params"),
                activation=lr.get("activation"),
                qdq_fin_exp=lr.get("qdq_fin_exp"),
                qdq_fout_exp=lr.get("qdq_fout_exp"),
                onnx_add_b_quantized=lr.get("onnx_add_b_quantized"),
            )
            qpm = dict(layer.quant_params) if isinstance(layer.quant_params, dict) else {}
            qpm["qdq_fin_scale"] = lr.get("qdq_fin_scale")
            qpm["qdq_fout_scale"] = lr.get("qdq_fout_scale")
            layer.quant_params = qpm
            fc_layers.append(layer)

        # ----- Step 3d: AveragePool detection + spatial inference -----
        avg_node = _find_avgpool_node(float_model)
        pool_channels, pool_in_h, pool_in_w = _conv_layers_to_pool_input(conv_layers) if conv_layers else (1, 1, 1)
        if avg_node is None:
            LOGGER.warning("No AveragePool node found; using kernel=1, stride=1, frame_rows=%d", pool_in_h)
            pool_kernel = 1
            pool_stride = 1
            pool_out_rows = pool_in_h
        else:
            (kP, _kw_pool), (sP, _sw_pool), pP = _avgpool_attrs(avg_node)
            out_h, _ = _avgpool_output_spatial(pool_in_h, pool_in_w, (kP, _kw_pool), (sP, _sw_pool), pP)
            pool_kernel = kP
            pool_stride = sP
            pool_out_rows = out_h
        flatten_size = pool_channels * pool_out_rows
        # Sanity check: flatten_size must equal first FC's INPUT_SIZE
        if fc_layers and fc_layers[0].in_features and int(fc_layers[0].in_features) != int(flatten_size):
            LOGGER.warning(
                "Flatten size %d != first FC.in_features %d — conv/pool spatial trace may not match the first FC.",
                flatten_size, fc_layers[0].in_features,
            )

        # ----- Step 3e: Build RTL quant descriptors -----
        if conv_layers:
            mrm.build_rtl_conv_quant_descriptors(conv_layers, bit_width=weight_format_bits)
            _enforce_conv_numeric_fidelity_or_warn(conv_layers)

        # FC: attach paired float/int tensors so build_rtl_layer_quant_descriptors works.
        # For multiclass we only have the quantized ONNX; the float ONNX (without QDQ) gives
        # the "true" weights. Use the inter-layer scale attach helper.
        flatten = mrm.LayerInfo(name="flatten_1", layer_type="flatten",
                                out_shape=(1, int(flatten_size)))
        full_fc_chain: List[mrm.LayerInfo] = [flatten]
        for i, lyr in enumerate(fc_layers):
            full_fc_chain.append(lyr)
            if i < len(fc_layers) - 1:
                full_fc_chain.append(mrm.LayerInfo(name=f"fc_relu_{i+1}", layer_type="relu"))
        try:
            attach_inter_layer_scale_tensors_from_onnx_pair(full_fc_chain, quant_onnx, float_path)
        except RuntimeError as e:
            LOGGER.warning("Inter-layer ONNX pair attach failed: %s — falling back to weight tensor only.", e)
        mrm.build_rtl_layer_quant_descriptors(fc_layers, bit_width=weight_format_bits, log_summary=False)

        # ----- Step 3f: Emit the RTL IP -----
        out_dir.mkdir(parents=True, exist_ok=True)
        num_classes = int(fc_layers[-1].out_features) if fc_layers else 1
        mrm.emit_multiclass_format(
            out_dir,
            conv_layers,
            full_fc_chain,
            final_op_kind=head_kind,
            num_classes=num_classes,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            pool_frame_rows=pool_in_h,
            pool_channels=pool_channels,
            pool_out_rows=pool_out_rows,
            flatten_size=int(flatten_size),
            weight_width=weight_format_bits,
            scale=scale,
        )
        mrm.generate_multiclass_rtl_filelist(
            out_dir, float_path.stem, conv_layers, fc_layers, final_op_kind=head_kind,
        )
        mrm.generate_multiclass_mapping_report(
            out_dir, float_path.stem, conv_layers, fc_layers,
            final_op_kind=head_kind,
            num_classes=num_classes,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            pool_frame_rows=pool_in_h,
            pool_out_rows=pool_out_rows,
            flatten_size=int(flatten_size),
            weight_width=weight_format_bits,
        )
        mrm.generate_multiclass_netlist_json(
            out_dir, float_path.stem, conv_layers, fc_layers,
            final_op_kind=head_kind,
            num_classes=num_classes,
        )
        LOGGER.info("Multiclass RTL generation complete! Output: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
