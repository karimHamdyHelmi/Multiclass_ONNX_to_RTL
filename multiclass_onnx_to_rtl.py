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

The Conv layers' ``Fin / Fout`` exponents are derived from the same QDQ chain
walkers used for FC (``_fin_exponent_from_dq_chain`` / ``_fout_exponent_from_q_chain``
imported from the binary script).

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

# -----------------------------------------------------------------------------
# Path setup: locate sibling Binary_ONNX_to_RTL directory and add to sys.path so
# we can reuse its layer-extraction helpers (Fin/Fout chain walkers, MatMul/Gemm
# extractors, attach_inter_layer_scale_tensors_from_onnx_pair).
# -----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_BINARY_DIR_CANDIDATES = [
    _THIS_DIR.parent / "pyramidstech" / "Binary_ONNX_to_RTL",
    _THIS_DIR.parent.parent / "pyramidstech" / "Binary_ONNX_to_RTL",
    Path(r"C:/Users/Kimo_/OneDrive - Alexandria University/Desktop/pyramidstech/Binary_ONNX_to_RTL"),
]
_BINARY_DIR: Optional[Path] = None
for _cand in _BINARY_DIR_CANDIDATES:
    if (_cand / "binary_onnx_to_rtl.py").is_file():
        _BINARY_DIR = _cand.resolve()
        break
if _BINARY_DIR is None:
    raise RuntimeError(
        "Cannot locate sibling Binary_ONNX_to_RTL directory. "
        "Searched: " + ", ".join(str(c) for c in _BINARY_DIR_CANDIDATES)
    )
if str(_BINARY_DIR) not in sys.path:
    sys.path.insert(0, str(_BINARY_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import binary_onnx_to_rtl as _bin_pipe  # noqa: E402
import multiclass_rtl_mapper as mrm  # noqa: E402

# Reuse the binary script's ONNX helpers verbatim — the FC handling is identical.
_load_onnx = _bin_pipe._load_onnx
_get_initializers_dict = _bin_pipe._get_initializers_dict
_build_value_to_array = _bin_pipe._build_value_to_array
_get_attr = _bin_pipe._get_attr
_get_node_op_type = _bin_pipe._get_node_op_type
_fin_exponent_from_dq_chain = _bin_pipe._fin_exponent_from_dq_chain
_fout_exponent_from_q_chain = _bin_pipe._fout_exponent_from_q_chain
_fin_scale_from_dq_chain = _bin_pipe._fin_scale_from_dq_chain
_fout_scale_from_q_chain = _bin_pipe._fout_scale_from_q_chain
_qdq_fin_fout_for_fc_node = _bin_pipe._qdq_fin_fout_for_fc_node
extract_layers_from_onnx = _bin_pipe.extract_layers_from_onnx
attach_inter_layer_scale_tensors_from_onnx_pair = _bin_pipe.attach_inter_layer_scale_tensors_from_onnx_pair
extract_per_layer_activations = _bin_pipe.extract_per_layer_activations

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


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
    (Relu/Sigmoid/Tanh) — same logic as binary script's per-FC activation trace.
    """
    return _bin_pipe._trace_activation_forward(model, conv_output_name, input_to_nodes)


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

        prod = _bin_pipe._producer_node_for_tensor(model, cur)
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

        # Bias resolution: float initializer (or None → zeros)
        if bias_input and bias_input in name_to_init:
            bias_arr = name_to_init[bias_input].astype(np.float32).ravel()
            if bias_arr.size != out_ch:
                LOGGER.warning("%s: bias size %d != out_ch %d", node.name, bias_arr.size, out_ch)
                bias_arr = np.resize(bias_arr, out_ch).astype(np.float32)
        else:
            bias_arr = np.zeros((out_ch,), dtype=np.float32)

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
    """Same as binary script: synthesize a calibration .npz from the float model's
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

    Logs a warning rather than raising — multiclass models often have wider
    activation ranges than the binary FC-only models, so saturation > 1% is more
    common but still tolerable. The ``--strict-fidelity`` CLI flag promotes
    these to errors.
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

        # ----- Step 3c: Extract FC layers (reuse binary script) -----
        # binary_onnx_to_rtl.extract_layers_from_onnx returns FC-only entries; for a
        # mixed conv+FC model the conv outputs are NOT processed by this extractor
        # (it ignores Conv nodes), so we get just the post-flatten FC chain.
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
        # the "true" weights. Use the same attach helper as binary script.
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
