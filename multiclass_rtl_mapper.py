#!/usr/bin/env python3
"""
RTL generation engine for ``multiclass_onnx_to_rtl.py``.
========================================================

Takes per-layer Conv + FC weight/bias arrays (with quantization metadata) and emits
synthesizable SystemVerilog modules plus ``.mem`` weight/bias ROM files for a complete
multiclass classifier IP.

**Pipeline overview:**

1. **Fixed-point quantization** (``build_rtl_conv_quant_descriptors`` and the FC
   chain helpers) — convert float32 weights/biases to power-of-two int8
   (W_int, B_int) with exponents Fw, Fb.

2. **Scale derivation** — derive per-layer ``LAYER_SCALE = (Fin + Fw) - Fout`` and
   ``BIAS_SCALE = (Fin + Fw) - Fb``, expressed as signed integer right-/left-shifts.

3. **Memory file generation** — emit packed ``.mem`` files in the layout required by:
     - depthwise_conv_engine.sv    : single ROM row, ``NUM_FILTERS * (K_H * K_W * K_C) * Q_WIDTH`` bits
     - pointwise_conv_engine.sv    : single ROM row, ``NUM_FILTERS * INPUTS_PER_CYCLE * Q_WIDTH`` bits
     - conv bias ROMs              : single ROM row, ``NUM_FILTERS * BIAS_WIDTH`` bits (sign-extended int32)
     - fc_in / fc_out ROMs         : LSB-first packing shared across the FC chain.

4. **SystemVerilog emission** (``emit_multiclass_format``) — write the entire IP:
     - Shared infrastructure: quant_pkg.sv, mac.sv, fc_in.sv, fc_out.sv,
       fc_in_layer.sv, fc_out_layer.sv, relu_layer.sv, sync_fifo.sv
     - Conv-side modules: line_buffers.sv, depthwise_conv_engine.sv,
       pointwise_conv_engine.sv, avg_pool_kx1.sv, flatten_unit.sv,
       softmax_layer.sv / argmax_layer.sv (chosen from ONNX final op),
       multiclass_NN.sv (top module), multiclass_NN_wrapper.sv

This module is fully self-contained: every helper, SV template, and ROM packer it
needs to convert a multiclass Conv + FC ONNX classifier into RTL lives in this
file.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# =============================================================================
# PyramidTech Header Utilities
# Every generated .sv file starts with a proprietary header block.  These helpers
# produce the header text and wrap module bodies with `begin_keywords / end_keywords.
# =============================================================================

def _pyramidtech_header(file_name: str, description: str, *, author: str = "Ahmed Abou-Auf", quant_pkg_style: bool = False) -> str:
    """Generate PyramidTech proprietary header block."""
    desc_lines = description.strip().split("\n")
    if len(desc_lines) == 1:
        desc_block = desc_lines[0].strip()
    else:
        first = desc_lines[0].strip()
        rest = "\n".join("                " + line.strip() for line in desc_lines[1:] if line.strip())
        desc_block = first + "\n" + rest if rest else first
    closing = "***************************************************************************************/" if quant_pkg_style else "****************************************************************************************/"
    return f'''/****************************************************************************************
"PYRAMIDTECH CONFIDENTIAL

Copyright (c) 2026 PyramidTech LLC. All rights reserved.

This file contains proprietary and confidential information of PyramidTech LLC.
The information contained herein is unpublished and subject to trade secret
protection. No part of this file may be reproduced, modified, distributed,
transmitted, disclosed, or used in any form or by any means without the
prior written permission of PyramidTech LLC.

This material must be returned immediately upon request by PyramidTech LLC"
/****************************************************************************************
File name:      {file_name}
  
Description:    {desc_block}
  
Author:         {author}
  
Change History:
02-25-2026     AA  Initial Release
  
{closing}
'''


# -----------------------------------------------------------------------------
# RTL content wrapping (PyramidTech headers)
# -----------------------------------------------------------------------------

def _pyramidtech_wrap(content: str, file_name: str, description: str, *, author: str = "Ahmed Abou-Auf", quant_pkg_style: bool = False) -> str:
    """Prepend PyramidTech header, wrap with begin_keywords/end_keywords, and return RTL content."""
    header = _pyramidtech_header(file_name, description, author=author, quant_pkg_style=quant_pkg_style)
    body = content.rstrip()
    # Strip existing begin_keywords/end_keywords to avoid duplication
    body = re.sub(r'^`begin_keywords\s+"[^"]*"\s*\n?', '', body)
    body = re.sub(r'\n?`end_keywords\s*$', '', body)
    body = body.strip()
    return header + '`begin_keywords "1800-2012"\n\n' + body + '\n\n`end_keywords'


# Header descriptions for embedded reusable .sv bodies
_FILE_DESCRIPTIONS = {
    "mac.sv": "Pipelined Multiply-Accumulate (MAC) unit with output saturation",
    "fc_in.sv": "Fully-connected (FC) input layer module. Instantiates MAC units, adds biases, and applies layer scaling with output saturation.",
    "fc_out.sv": "Fully-connected (FC) output layer module with a 3-stage pipeline: Stage 1 (Multiplication), Stage 2 (Addition/Scaling), and Stage 3 (Saturation/Output).",
    "relu_layer.sv": "ReLU activation function applied element-wise to a single input. Passes input directly if positive; outputs zero if negative.",
    "sync_fifo.sv": "Synchronous FIFO module for data buffering. Uses a standard pointer-based implementation with full/empty flags",
}


def _write_embedded_sv(sv_dir: Path, filename: str, body: str) -> None:
    """Write one SystemVerilog file from embedded body with PyramidTech header."""
    desc = _FILE_DESCRIPTIONS.get(filename, "")
    content = _pyramidtech_wrap(body, filename, desc)
    (sv_dir / filename).write_text(content.rstrip() + "\n", encoding="utf-8")


# =============================================================================
# Quantization Helpers and Core Data Types
#
# LayerInfo — carrier for everything the generator needs about one FC layer:
#   weights, biases, ONNX quantization metadata, paired float/int tensors,
#   and the computed RTL descriptor (set after build_rtl_layer_quant_descriptors).
#
# RtlLayerQuantDescriptor — per-layer output of the fixed-point conversion:
#   int8 W_int / B_int arrays, exponents Fw / Fb / Fin / Fout, saturation stats,
#   and the optional ONNX-pair Fb used for bias scale alignment.
#
# The quantization approach is "power-of-two gain": we find the largest integer
# exponent F such that round(x * 2^F) fits entirely within [-128, 127] with no
# clipping.  This makes the hardware shift-friendly (LAYER_SCALE / BIAS_SCALE
# are simple arithmetic right-/left-shifts).
# =============================================================================

def float_to_int(val: np.ndarray, scale: int, bit_width: int) -> np.ndarray:
    """Quantize float tensors with integer ``scale``: round(val * scale), clip (fallback path).

    Primary ONNX→RTL path uses :func:`build_rtl_layer_quant_descriptors` + power-of-two
    :func:`quantize_weight_for_rtl` / :func:`quantize_bias_for_rtl`; :func:`mem_export_weight_matrix`
    reads ``layer.rtl_quant`` when present.
    """
    if bit_width == 4:
        min_val, max_val = -8, 7
        dtype = np.int8
    elif bit_width == 8:
        min_val, max_val = -128, 127
        dtype = np.int8
    elif bit_width == 16:
        min_val, max_val = -32768, 32767
        dtype = np.int16
    else:
        raise ValueError(f"Unsupported bit width: {bit_width}. Supported: 4, 8, 16")

    quantized = np.clip(np.round(val.astype(np.float32) * float(scale)), min_val, max_val)
    return quantized.astype(dtype)


@dataclass
class LayerInfo:
    """Everything the RTL generator needs about one layer in the model.

    For linear (FC) layers the key fields are:
      - weight / bias: float32 arrays (dequantized from ONNX or original float).
      - qdq_fin_exp / qdq_fout_exp: activation exponents from ORT QDQ scales.
      - onnx_pair_*: matched float+int tensors from both the float and quantized
        ONNX — used to derive Fw/Fb exponents for the RTL shift parameters.
      - rtl_quant: the final RtlLayerQuantDescriptor (set after
        build_rtl_layer_quant_descriptors runs).

    Non-linear layers (flatten, relu) are placeholders that carry only name + type.
    """
    name: str
    layer_type: str  # 'flatten', 'linear', 'relu'
    module_qualname: Optional[str] = None
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    weight: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    in_shape: Optional[Tuple[int, ...]] = None
    out_shape: Optional[Tuple[int, ...]] = None
    quant_params: Optional[Dict[str, Any]] = None  # ONNX: weight_scale, b_zero_point, etc.
    activation: Optional[str] = None  # ONNX: Relu, Sigmoid, Tanh, etc. (from graph trace)
    # Static QDQ / QLinear: exponents from ONNX quant scales (see `_qdq_fin_fout_for_fc_node`).
    qdq_fin_exp: Optional[int] = None
    qdq_fout_exp: Optional[int] = None
    # Optional: float vs integer tensors 
    onnx_pair_float_weight: Optional[np.ndarray] = None
    onnx_pair_float_bias: Optional[np.ndarray] = None
    onnx_pair_quant_weight: Optional[np.ndarray] = None
    onnx_pair_quant_bias: Optional[np.ndarray] = None
    onnx_pair_weight_scale: Optional[float] = None  # scalar view (legacy compatibility)
    onnx_pair_weight_scale_arr: Optional[np.ndarray] = None  # full ONNX weight scale tensor (supports per-channel)
    # Static QDQ: int8 bias initializer ``{MatMulNodePrefix}_Add_B_quantized`` (same shape as bias vector).
    onnx_add_b_quantized: Optional[np.ndarray] = None
    # Unified RTL fixed-point (built after attach): .mem and LAYER_SCALE/BIAS_SCALE use this one descriptor.
    rtl_quant: Optional["RtlLayerQuantDescriptor"] = None


@dataclass
class RtlLayerQuantDescriptor:
    """Per-layer fixed-point descriptor: the bridge between float ONNX and hardware ROM.

    Created by ``build_rtl_layer_quant_descriptors`` for every FC layer.  Contains:
      - W_int / B_int: the actual int8 values written to .mem ROM files.
      - fw_frac / fb_rtl: power-of-two exponents (W ≈ W_int * 2^-fw_frac).
      - fin_qdq / fout_qdq: activation exponents from ORT QDQ scales.
      - fb_pair: alternative Fb from the ONNX float/int tensor pair (when present).
      - saturation percentages: how many elements hit the int8 clip limits.

    The shift values that parameterise fc_in/fc_out are derived from these exponents
    by ``compute_layer_scales_from_rtl_descriptors``.
    """

    layer_name: str
    layer_index: int
    W_float: np.ndarray  # (out_features, in_features)
    B_float: np.ndarray
    onnx_weight_scale: Optional[float]
    onnx_weight_zp: Optional[Any]  # from quant_params b_zero_point (ONNX metadata; not ROM zp)
    weight_init_name: Optional[str]  # e.g. MatMulInteger weight_scale initializer name
    # Chosen RTL convention: W_int ≈ round(W_float * 2^fw_frac), clip to int8
    fw_frac: int
    W_int: np.ndarray
    weight_sat_lo_pct: float
    weight_sat_hi_pct: float
    # Bias: B_int uses fb_rtl (F_mac = Fin + fw_frac).
    fin_qdq: int
    fb_rtl: int
    B_int: np.ndarray
    bias_sat_lo_pct: float
    bias_sat_hi_pct: float
    fout_qdq: int
    bias_clipped_for_int8: bool
    # ONNX-pair Fb from ONNX float/quant pair when present (scales / Add_B shift); None if no pair.
    fb_pair: Optional[int] = None

    @property
    def fb_frac(self) -> int:
        """Effective Fb for ONNX-pair scale rows: pair Fb when present, else RTL bias exponent (matches ``B_int`` domain)."""
        return int(self.fb_pair) if self.fb_pair is not None else int(self.fb_rtl)


def _saturation_pct_int(arr: np.ndarray, lo: int, hi: int) -> Tuple[float, float]:
    a = np.asarray(arr, dtype=np.int32).ravel()
    n = max(a.size, 1)
    return (100.0 * float(np.sum(a <= lo)) / n, 100.0 * float(np.sum(a >= hi)) / n)


def quantize_weight_for_rtl(
    W_float: np.ndarray,
    *,
    lo: int = -128,
    hi: int = 127,
) -> Tuple[int, np.ndarray, float, float]:
    """
    Pick the largest integer fw_frac >= 0 such that round(W_float * 2^fw_frac) fits in [lo, hi]
    with **no clipping** (same as handwritten-style fixed-point: power-of-two gain, shift-friendly).

    Returns (fw_frac, W_int, sat_lo_pct, sat_hi_pct) — sat should be 0.0 when unclipped.
    """
    W = np.asarray(W_float, dtype=np.float64)
    max_abs = float(np.max(np.abs(W)))
    if max_abs < 1e-15:
        z = np.zeros(W.shape, dtype=np.int32)
        return 0, z, 0.0, 0.0
    F = int(np.floor(np.log2(hi / max_abs)))
    F = max(0, min(F, 24))
    while F >= 0:
        raw = np.rint(W * (2.0**F))
        if np.all((raw >= lo) & (raw <= hi)):
            wq = raw.astype(np.int32)
            sl, sh = _saturation_pct_int(wq, lo, hi)
            return F, wq, sl, sh
        F -= 1
    raw = np.rint(W)
    wq = np.clip(raw, lo, hi).astype(np.int32)
    sl, sh = _saturation_pct_int(wq, lo, hi)
    LOGGER.warning(
        "quantize_weight_for_rtl: could not fit without clip at any power-of-two F; clipped at F=0 (layer max_abs=%.6g)",
        max_abs,
    )
    return 0, wq, sl, sh


def quantize_bias_for_rtl(
    B_float: np.ndarray,
    F_mac: int,
    *,
    lo: int = -128,
    hi: int = 127,
    layer_name: str = "",
) -> Tuple[int, np.ndarray, bool]:
    """
    Quantize float bias (ONNX, post-matmul add) to int8 for ROM.

    **RTL (fc_in / fc_out):** ``bias_*`` is sign-extended int8 in ``acc_t``. ``BIAS_SCALE`` shifts the bias
    so it matches the MAC sum domain (``F_mac = Fin + Fw``). ``LAYER_SCALE`` is applied after bias + MAC.

    **Quantization rule (reference-style):** At each candidate ``Fb``, use **per-element**
    conversion: **nearest integer** (``rint``) if it lies in int8 range, otherwise **truncate toward zero**
    (``trunc``) if that lies in range. This avoids lowering **global** ``Fb`` (which would force
    ``BIAS_SCALE = F_mac - F_b != 0``) when only a few elements would round to ±128 under ``rint`` but
    fit with ``trunc`` at ``Fb = F_mac`` — the usual cause of ``FC_*_OUT_BIAS_SCALE = 1`` vs reference ``0``.

    Returns (fb_frac, B_int, was_clipped).
    """
    B = np.asarray(B_float, dtype=np.float64).ravel()
    if B.size == 0:
        return F_mac, np.zeros(0, dtype=np.int32), False

    f_hi = max(0, min(int(F_mac), 24))
    for Fb in range(f_hi, -1, -1):
        scaled = B * (2.0**Fb)
        rr = np.rint(scaled)
        rt = np.trunc(scaled)
        in_r = (rr >= lo) & (rr <= hi)
        in_t = (rt >= lo) & (rt <= hi)
        if np.all(in_r | in_t):
            out = np.where(in_r, rr, rt).astype(np.int32)
            return Fb, out, False

    LOGGER.warning(
        "%s: no Fb in [0..%s] fits per-element rint/trunc; clipping at Fb=0.",
        layer_name or "layer",
        f_hi,
    )
    scaled = B * 1.0
    rr = np.rint(scaled)
    rt = np.trunc(scaled)
    in_r = (rr >= lo) & (rr <= hi)
    pre = np.where(in_r, rr, rt)
    clipped = np.clip(pre, lo, hi).astype(np.int32)
    return 0, clipped, True


def build_rtl_layer_quant_descriptors(
    linear_layers: List[LayerInfo],
    *,
    bit_width: int = 8,
    log_summary: bool = True,
    log_detail: bool = False,
) -> List[RtlLayerQuantDescriptor]:
    """
    Build per-layer RTL quantization from **float** weights/biases (dequant ONNX or float model).
    Weights: ``quantize_weight_for_rtl`` (power-of-two int8, unchanged).

    **proj_in** bias ``.mem``: ``B_int`` from ``quantize_bias_for_rtl(B_float, Fin+Fw_rtl)``, sign-extended.

    **proj_out** bias ``.mem``: export ONNX ``*_Add_B_quantized`` sign-extended to ``acc_t`` when present
    (graph truth); otherwise use descriptor-domain ``rq.B_int`` sign-extended to ``acc_t``.

    **Descriptor**: ``fb_rtl`` quantizes ``B_int``; ``fb_pair`` is ONNX-pair Fb when ONNX pair exists.
    ``fb_frac`` property is ``fb_pair`` if set else ``fb_rtl`` (for ``_qdq_pair_bias_alignment`` / scales).

    ``FC_*_*_LAYER_SCALE``: ``compute_layer_scales_from_rtl_descriptors`` uses ``_qdq_pair_alignment`` on ONNX
    QDQ ``Fin``/``Fout`` and ONNX-pair ``Fw``/``Fb`` for all blocks (including odd-tail last block).

    ``log_detail``: per-layer Fin/Fw/Fb and min/max logs.
    """
    if bit_width != 8:
        raise NotImplementedError("build_rtl_layer_quant_descriptors currently supports bit_width=8 only")
    lo, hi = -128, 127
    out: List[RtlLayerQuantDescriptor] = []
    for i, layer in enumerate(linear_layers):
        if layer.qdq_fin_exp is None or layer.qdq_fout_exp is None:
            raise RuntimeError(
                f"{layer.name}: missing Fin/Fout exponents from quantized ONNX (qdq_fin_exp / qdq_fout_exp). "
                "Export with static QDQ (``onnx_quantize.py``: quantize_static, QuantFormat.QDQ) or use "
                "QLinearMatMul/QLinearGemm so activation scales appear as graph initializers."
            )
        fin = abs(int(layer.qdq_fin_exp))
        fout = abs(int(layer.qdq_fout_exp))
        if layer.onnx_pair_float_weight is not None:
            Wf = np.asarray(layer.onnx_pair_float_weight, dtype=np.float64)
        elif layer.weight is not None:
            Wf = np.asarray(layer.weight, dtype=np.float64)
        else:
            raise RuntimeError(f"{layer.name}: need onnx_pair_float_weight or weight for RTL quant")
        if layer.onnx_pair_float_bias is not None:
            Bf = np.asarray(layer.onnx_pair_float_bias, dtype=np.float64).ravel()
        elif layer.bias is not None:
            Bf = np.asarray(layer.bias, dtype=np.float64).ravel()
        else:
            Bf = np.zeros((layer.out_features or 0,), dtype=np.float64)

        qp = layer.quant_params if isinstance(layer.quant_params, dict) else {}
        onnx_ws = qp.get("weight_scale")
        if onnx_ws is None:
            onnx_ws = layer.onnx_pair_weight_scale
        onnx_zp = qp.get("b_zero_point")
        w_init = qp.get("weight_scale_init_name")

        fw_frac, W_int, wsl, wsh = quantize_weight_for_rtl(Wf, lo=lo, hi=hi)
        f_mac = fin + fw_frac

        fb_rtl, B_int, b_clip = quantize_bias_for_rtl(Bf, f_mac, lo=lo, hi=hi, layer_name=layer.name)
        bsl, bsh = _saturation_pct_int(B_int, lo, hi)
        fb_pair: Optional[int] = None
        if _layer_has_full_onnx_pair(layer):
            _, fb_pair = _fw_fb_pair_exponents_from_onnx_pair(layer, fin)

        desc = RtlLayerQuantDescriptor(
            layer_name=layer.name,
            layer_index=i,
            W_float=Wf.astype(np.float64),
            B_float=Bf.astype(np.float64),
            onnx_weight_scale=float(onnx_ws) if onnx_ws is not None else None,
            onnx_weight_zp=onnx_zp,
            weight_init_name=str(w_init) if w_init is not None else None,
            fw_frac=fw_frac,
            W_int=W_int,
            weight_sat_lo_pct=wsl,
            weight_sat_hi_pct=wsh,
            fin_qdq=fin,
            fb_rtl=fb_rtl,
            B_int=B_int,
            bias_sat_lo_pct=bsl,
            bias_sat_hi_pct=bsh,
            fout_qdq=fout,
            bias_clipped_for_int8=b_clip,
            fb_pair=fb_pair,
        )
        out.append(desc)
        layer.rtl_quant = desc

        if log_detail:
            pair_note = ""
            if _layer_has_full_onnx_pair(layer):
                fwp, _ = _fw_fb_pair_exponents_from_onnx_pair(layer, fin)
                pair_note = f" | ONNX-pair_Fw(ONNX)={fwp} for BIAS_SCALE rows (ROM weight Fw={fw_frac})"
            LOGGER.info(
                "rtl_quant %s: Fin=%d Fw=%d F_mac=%d Fb_rtl=%d Fb_pair=%s Fout=%d | W_float[min,max]=[%.6g,%.6g] W_int[min,max]=[%d,%d] "
                "w_sat(lo,hi)=(%.2f%%,%.2f%%) | B_float[min,max]=[%.6g,%.6g] B_int[min,max]=[%d,%d] b_sat=(%.2f%%,%.2f%%)%s%s",
                layer.name,
                fin,
                fw_frac,
                f_mac,
                fb_rtl,
                str(fb_pair) if fb_pair is not None else "—",
                fout,
                float(Wf.min()),
                float(Wf.max()),
                int(W_int.min()),
                int(W_int.max()),
                wsl,
                wsh,
                float(Bf.min()) if Bf.size else 0.0,
                float(Bf.max()) if Bf.size else 0.0,
                int(B_int.min()) if B_int.size else 0,
                int(B_int.max()) if B_int.size else 0,
                bsl,
                bsh,
                " ONNX_scale=%s" % (onnx_ws,) if onnx_ws is not None else "",
                pair_note,
            )
        if wsl > 1.0 or wsh > 1.0 or bsl > 1.0 or bsh > 1.0:
            LOGGER.warning(
                "rtl_quant %s: saturation > 1%% after power-of-two quant — inspect float range.",
                layer.name,
            )

    if log_summary and log_detail:
        LOGGER.info("--- rtl_quant summary (power-of-two ROM, unified with Fw/Fb for shifts) ---")
        for d in out:
            LOGGER.info(
                "  %s: 2^Fw weights Fw=%d | bias Fb_rtl=%d Fb_pair=%s | W_int range [%d,%d]",
                d.layer_name,
                d.fw_frac,
                d.fb_rtl,
                str(d.fb_pair) if d.fb_pair is not None else "—",
                int(d.W_int.min()),
                int(d.W_int.max()),
            )
    elif log_summary and not log_detail:
        LOGGER.debug(
            "rtl_quant: built %d layer descriptor(s) (power-of-two ROM; use --verbose for Fin/Fw/Fb details).",
            len(out),
        )
    return out


def compute_layer_scales_from_rtl_descriptors(
    linear_layers: List[LayerInfo],
    descriptors: List[RtlLayerQuantDescriptor],
) -> Dict[str, int]:
    """LAYER_SCALE/BIAS_SCALE from RTL ROM exponents (fw_frac / fb_frac).

    Policy:
    - Fin/Fout from QDQ exponents on each layer descriptor.
    - Fw from ``quantize_weight_for_rtl`` (``d.fw_frac``): the actual ROM weight exponent.
    - Fb from ``d.fb_frac`` (ONNX-pair Fb when present, else ``fb_rtl``).
    - Single alignment rule for all rows/layers (no topology-specific exceptions).
    """
    if len(descriptors) != len(linear_layers):
        raise ValueError("descriptor count must match linear layer count")
    n = len(linear_layers)
    rows: List[Dict[str, Any]] = []
    for i, (layer, d) in enumerate(zip(linear_layers, descriptors)):
        fw_rom = int(d.fw_frac)
        fb_rom = int(d.fb_frac)
        bsc, bdir = _qdq_pair_bias_alignment(d.fin_qdq, fw_rom, fb_rom)
        lsc, ldir = _qdq_pair_layer_alignment(int(d.fin_qdq), fw_rom, int(d.fout_qdq))
        rows.append(
            {
                "Fin": d.fin_qdq,
                "Fw": fw_rom,
                "Fb": fb_rom,
                "Fout": d.fout_qdq,
                "biasScale": bsc,
                "biasDir": bdir,
                "layerScale": lsc,
                "layerDir": ldir,
            }
        )
    num_blocks = (n + 1) // 2
    result: Dict[str, int] = {}
    for b in range(num_blocks):
        in_idx = 2 * b
        out_idx = 2 * b + 1 if 2 * b + 1 < n else None
        rin = rows[in_idx]
        rout = rows[out_idx] if out_idx is not None else rin
        prefix = f"FC_{b + 1}"
        result[f"{prefix}_IN_LAYER_SCALE"] = _rtl_signed_layer_scale(rin["layerScale"], rin["layerDir"])
        result[f"{prefix}_IN_BIAS_SCALE"] = _rtl_signed_bias_scale(rin["biasScale"], rin["biasDir"])
        result[f"{prefix}_OUT_LAYER_SCALE"] = _rtl_signed_layer_scale(rout["layerScale"], rout["layerDir"])
        result[f"{prefix}_OUT_BIAS_SCALE"] = _rtl_signed_bias_scale(rout["biasScale"], rout["biasDir"])
    if n % 2 == 1 and num_blocks >= 1:
        last_i = n - 1
        dlast = descriptors[last_i]
        prev = linear_layers[last_i - 1] if last_i > 0 else None
        if prev is not None and prev.qdq_fout_exp is not None:
            fin_o = abs(int(prev.qdq_fout_exp))
        else:
            fin_o = int(dlast.fin_qdq)
        fw_tail_out = 0
        lsc_o, ldir_o = _qdq_pair_layer_alignment(fin_o, fw_tail_out, dlast.fout_qdq)
        p_last = f"FC_{num_blocks}"
        result[f"{p_last}_OUT_LAYER_SCALE"] = _rtl_signed_layer_scale(lsc_o, ldir_o)
        result[f"{p_last}_OUT_BIAS_SCALE"] = 0
    for k in list(result.keys()):
        raw_v = int(result[k])
        if raw_v < -24 or raw_v > 24:
            raise RuntimeError(f"Scale parameter {k} out of supported range [-24,24]: {raw_v}")
        result[k] = raw_v
    return result


def _fc_layer_scales_unified_or_legacy(
    linear_layers: List[LayerInfo],
    weight_width: int,
    python_scale: int,
) -> Tuple[Dict[str, int], str]:
    """Return (scales dict, tag) using ``rtl_quant`` when present, else ONNX-pair inter-layer rows."""
    descs_rtl = [L.rtl_quant for L in linear_layers if L.rtl_quant is not None]
    if len(descs_rtl) == len(linear_layers) and linear_layers:
        return compute_layer_scales_from_rtl_descriptors(linear_layers, descs_rtl), "rtl_quant_unified"
    return compute_fc_layer_scales(linear_layers, weight_width, python_scale), "onnx_mode_legacy"


def _clip_to_bitwidth(w: np.ndarray, bit_width: int) -> np.ndarray:
    w = np.asarray(w, dtype=np.int32)
    if bit_width == 4:
        return np.clip(w, -8, 7)
    if bit_width == 8:
        return np.clip(w, -128, 127)
    if bit_width == 16:
        return np.clip(w, -32768, 32767)
    raise ValueError(f"Unsupported bit_width {bit_width}")


def mem_export_weight_matrix(
    layer: LayerInfo,
    bit_width: int,
    rom_bias_quantize_scale: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    ROM integers for .mem. Prefer ``layer.rtl_quant`` (power-of-two float→int8, same Fw/Fb as
    ``compute_layer_scales_from_rtl_descriptors``). Missing descriptor is treated as an error to
    avoid silent fallback quantization policy drift.
    """
    if layer.weight is None:
        raise RuntimeError(f"Layer {layer.name} missing weight")

    rq = layer.rtl_quant
    if rq is not None:
        wq = _clip_to_bitwidth(np.asarray(rq.W_int, dtype=np.int32), bit_width)
        bq = _clip_to_bitwidth(np.asarray(rq.B_int, dtype=np.int32), bit_width)
        return wq, bq, "rtl_power2_unified"
    raise RuntimeError(
        f"{layer.name}: missing rtl_quant descriptor for .mem export. "
        "Run build_rtl_layer_quant_descriptors before generation."
    )


def log_mem_export_debug(
    layer_name: str,
    wq: np.ndarray,
    bq: np.ndarray,
    source_tag: str,
    *,
    bit_width: int = 8,
    onnx_weight_scale: Optional[float] = None,
    onnx_zp: Optional[Any] = None,
) -> None:
    """Saturation and range stats for exported .mem tensors (weights + biases)."""
    wq = np.asarray(wq, dtype=np.int32)
    bq = np.asarray(bq, dtype=np.int32).ravel()
    lo, hi = (-128, 127) if bit_width == 8 else (-8, 7) if bit_width == 4 else (-32768, 32767)
    nw = wq.size
    pct_lo = 100.0 * float(np.sum(wq <= lo)) / max(nw, 1)
    pct_hi = 100.0 * float(np.sum(wq >= hi)) / max(nw, 1)
    nb = max(bq.size, 1)
    pct_b_lo = 100.0 * float(np.sum(bq <= lo)) / nb
    pct_b_hi = 100.0 * float(np.sum(bq >= hi)) / nb
    extra = ""
    if onnx_weight_scale is not None:
        extra += f" onnx_w_scale={onnx_weight_scale}"
    if onnx_zp is not None:
        extra += f" onnx_w_zp={onnx_zp}"
    LOGGER.info(
        "mem_debug %s: src=%s w[%d] min/max=%s/%s sat_lo=%.2f%% sat_hi=%.2f%% | b[%d] min/max=%s/%s sat_lo=%.2f%% sat_hi=%.2f%%%s",
        layer_name,
        source_tag,
        nw,
        int(wq.min()) if nw else 0,
        int(wq.max()) if nw else 0,
        pct_lo,
        pct_hi,
        bq.size,
        int(bq.min()) if bq.size else 0,
        int(bq.max()) if bq.size else 0,
        pct_b_lo,
        pct_b_hi,
        extra,
    )
    if pct_lo > 1.0 or pct_hi > 1.0 or pct_b_lo > 1.0 or pct_b_hi > 1.0:
        LOGGER.warning(
            "mem_debug %s: saturation > 1%% on weights or biases — check ONNX vs ROM quantization path.",
            layer_name,
        )


def mem_saturation_summary(wq: np.ndarray, bq: np.ndarray, bit_width: int = 8) -> Dict[str, Any]:
    """Structured stats for validation scripts (no file I/O)."""
    wq = np.asarray(wq, dtype=np.int32).ravel()
    bq = np.asarray(bq, dtype=np.int32).ravel()
    lo, hi = (-128, 127) if bit_width == 8 else (-8, 7) if bit_width == 4 else (-32768, 32767)
    nw = max(wq.size, 1)
    nb = max(bq.size, 1)
    return {
        "weight_n": int(wq.size),
        "weight_min": int(wq.min()) if wq.size else 0,
        "weight_max": int(wq.max()) if wq.size else 0,
        "weight_sat_lo_pct": 100.0 * float(np.sum(wq <= lo)) / nw,
        "weight_sat_hi_pct": 100.0 * float(np.sum(wq >= hi)) / nw,
        "bias_n": int(bq.size),
        "bias_min": int(bq.min()) if bq.size else 0,
        "bias_max": int(bq.max()) if bq.size else 0,
        "bias_sat_lo_pct": 100.0 * float(np.sum(bq <= lo)) / nb,
        "bias_sat_hi_pct": 100.0 * float(np.sum(bq >= hi)) / nb,
    }


# =============================================================================
# Memory File Generation (.mem format for weights and biases)
#
# .mem files are plain-text hex dumps loaded by $readmemh in the SV ROMs.
# Two packing strategies exist:
#
#   proj_in (fc_in_layer):
#     - weights: one hex line per input-feature index; each line packs all
#       neurons' int8 weights for that input (neuron 0 in LSBs).
#     - bias:    single hex line packing all neurons' biases, each sign-extended
#       from int8 to acc_t (4*Q_WIDTH bits).
#
#   proj_out (fc_out_layer):
#     - weights: one hex line per output neuron; each line packs all input
#       weights for that neuron (input 0 in LSBs).
#     - bias:    one hex line per output neuron; each value is a full acc_t
#       (32-bit) integer, quantized directly from the float bias into the
#       accumulator domain (F_acc = Fin_act + Fw_rom).
#
# The asymmetry between proj_in and proj_out bias formats is intentional:
# proj_in uses int8 biases with BIAS_SCALE shifts, while proj_out bypasses
# BIAS_SCALE by quantizing the bias at the exact accumulator exponent.
# =============================================================================

def generate_quant_pkg_style_weight_mem(
    weight_matrix: np.ndarray,
    out_path: Path,
    layer_name: str,
    in_features: int,
    out_features: int,
    bit_width: int
) -> None:
    """Generate weight .mem in quant_pkg format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weight_matrix = np.asarray(weight_matrix, dtype=np.int32)
    mask = (1 << bit_width) - 1

    with out_path.open("w", encoding="utf-8") as wf:
        if layer_name == "fc1":
            num_neurons = out_features
            total_bits = num_neurons * bit_width
            hex_width = (total_bits + 3) // 4
            for j in range(in_features):
                packed = 0
                for neuron_idx in range(num_neurons):
                    val = int(weight_matrix[neuron_idx, j]) & mask
                    packed |= val << (neuron_idx * bit_width)
                wf.write(f"{packed:0{hex_width}X}\n")
        else:
            num_inputs = in_features
            total_bits = num_inputs * bit_width
            hex_width = (total_bits + 3) // 4
            for neuron_idx in range(out_features):
                packed = 0
                for inp_idx in range(num_inputs):
                    val = int(weight_matrix[neuron_idx, inp_idx]) & mask
                    packed |= val << (inp_idx * bit_width)
                wf.write(f"{packed:0{hex_width}X}\n")


def generate_quant_pkg_style_bias_mem(
    bias_vector: np.ndarray,
    out_path: Path,
    num_neurons: int,
    bit_width: int
) -> None:
    """Generate bias .mem in quant_pkg format: single line, all biases packed (neuron 0 in LSBs)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bias_vector = np.asarray(bias_vector, dtype=np.int32)
    mask = (1 << bit_width) - 1
    total_bits = num_neurons * bit_width
    hex_width = (total_bits + 3) // 4
    packed = 0
    for neuron_idx in range(num_neurons):
        val = int(bias_vector[neuron_idx]) & mask
        packed |= val << (neuron_idx * bit_width)
    with out_path.open("w", encoding="utf-8") as bf:
        bf.write(f"{packed:0{hex_width}X}\n")


def bias_int8_rom_to_acc_per_neuron(bias_int8: np.ndarray, bit_width: int) -> np.ndarray:
    """
    Sign-extend int8 ``B_int`` (``quantize_bias_for_rtl``) to ``acc_t`` width for **proj_in** packed bias
    and for **proj_out** when using the descriptor ``B_int`` path.
    """
    bias_int8 = np.asarray(bias_int8, dtype=np.int32).ravel()
    mask = (1 << bit_width) - 1
    out = np.zeros(bias_int8.size, dtype=np.int64)
    for i in range(bias_int8.size):
        val = int(bias_int8[i]) & mask
        if val >= (1 << (bit_width - 1)):
            val = val - (1 << bit_width)
        out[i] = val
    return out


def _acc_bitpattern_to_signed(v: int, acc_width: int) -> int:
    """Interpret ``v`` as ``acc_width``-bit two's complement (handles uint32-style -1 == 0xffffffff)."""
    mask_u = (1 << acc_width) - 1
    u = int(v) & mask_u
    sign = 1 << (acc_width - 1)
    return u - (1 << acc_width) if (u & sign) else u


def quantize_bias_to_acc_domain(
    bias_float: np.ndarray,
    acc_exp: int,
    acc_width: int,
) -> np.ndarray:
    """Quantize float bias vector to the full acc_t domain: round(B * 2^acc_exp), clip to signed acc_width."""
    raw = np.rint(np.asarray(bias_float, dtype=np.float64).ravel() * (2.0 ** int(acc_exp)))
    min_acc = -(1 << (int(acc_width) - 1))
    max_acc = (1 << (int(acc_width) - 1)) - 1
    return np.clip(raw, min_acc, max_acc).astype(np.int64)


def proj_out_bias_rom_values(
    layer: LayerInfo,
    bit_width: int,
    *,
    fin_act_exp: int,
    mem_label: str = "",
) -> np.ndarray:
    """Per-neuron ``acc_t`` integers for ``*_proj_out_bias.mem``.

    Rebuilds bias from ``rq.B_float`` into the emitted accumulator domain
    (``F_acc = fin_act_exp + rq.fw_frac``), using the full ``acc_t`` width.
    """
    rq = layer.rtl_quant
    if rq is None:
        raise RuntimeError(f"{layer.name}: rtl_quant required for proj_out bias export")
    acc_w = 4 * bit_width
    f_acc = int(fin_act_exp) + int(rq.fw_frac)
    b_float = np.asarray(rq.B_float, dtype=np.float64).ravel()
    out = quantize_bias_to_acc_domain(b_float, f_acc, acc_w)
    LOGGER.debug(
        "proj_out_bias %s%s: rebuilt_from_float fin_act=%d fw_rom=%d f_acc=%d exported[min,max]=[%s,%s]",
        layer.name,
        f" ({mem_label})" if mem_label else "",
        fin_act_exp,
        rq.fw_frac,
        f_acc,
        int(out.min()) if out.size else "—",
        int(out.max()) if out.size else "—",
    )
    return out


def quantize_output_bias_for_acc_rom(
    layer: LayerInfo,
    bit_width: int,
    *,
    fin_act_exp: int,
    mem_label: str = "",
) -> np.ndarray:
    """Same as :func:`proj_out_bias_rom_values` (optional ``mem_label`` for log context)."""
    return proj_out_bias_rom_values(layer, bit_width, fin_act_exp=fin_act_exp, mem_label=mem_label)


def generate_proj_bias_mem_acc(
    bias_vector: np.ndarray,
    out_path: Path,
    num_neurons: int,
    bit_width: int,
) -> None:
    """Generate bias .mem for fc_in (acc_t packed): one row, each bias sign-extended to 4*bit_width."""
    acc_width = 4 * bit_width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bias_vector = np.asarray(bias_vector, dtype=np.int32)
    mask = (1 << bit_width) - 1
    total_bits = num_neurons * acc_width
    hex_width = (total_bits + 3) // 4
    packed = 0
    for neuron_idx in range(num_neurons):
        val = int(bias_vector[neuron_idx]) & mask
        if val >= (1 << (bit_width - 1)):
            val = val - (1 << bit_width)
        val_acc = val & ((1 << acc_width) - 1)
        packed |= val_acc << (neuron_idx * acc_width)
    with out_path.open("w", encoding="utf-8") as bf:
        bf.write(f"{packed:0{hex_width}X}\n")


def generate_proj_out_bias_mem(
    bias_vector: np.ndarray,
    out_path: Path,
    bit_width: int,
) -> None:
    """Write fc_out bias ROM: one hex line per address, full ``acc_t`` width (``proj_out_bias_rom_values``)."""
    acc_width = 4 * bit_width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bias_vector = np.asarray(bias_vector, dtype=np.int64).ravel()
    min_acc = -(1 << (acc_width - 1))
    max_acc = (1 << (acc_width - 1)) - 1
    hex_width = (acc_width + 3) // 4
    mask_u = (1 << acc_width) - 1
    with out_path.open("w", encoding="utf-8") as bf:
        for v in bias_vector:
            vi_s = _acc_bitpattern_to_signed(int(v), acc_width)
            if vi_s < min_acc or vi_s > max_acc:
                LOGGER.warning(
                    "generate_proj_out_bias_mem: bias %s outside signed acc_t range [%s,%s]; clipping",
                    vi_s,
                    min_acc,
                    max_acc,
                )
                vi_s = max(min_acc, min(max_acc, vi_s))
            val_acc = vi_s & mask_u
            bf.write(f"{val_acc:0{hex_width}X}\n")


def _proj_prefix(layer_idx: int) -> str:
    """ROM/file prefix: fc_proj for block 0, fc_2_proj for block 1, fc_3_proj for block 2, ..."""
    if layer_idx == 0:
        return "fc_proj"
    return f"fc_{layer_idx + 1}_proj"


def _fc_proj_mem_relpaths(linear_layer_index: int) -> Tuple[str, str]:
    """Paths (from RTL root) for paired FC blocks: even idx → *_in_*, odd → *_out_*."""
    b = linear_layer_index // 2
    prefix = _proj_prefix(b)
    if linear_layer_index % 2 == 0:
        return (
            f"mem_files/{prefix}_in_weights.mem",
            f"mem_files/{prefix}_in_bias.mem",
        )
    return (
        f"mem_files/{prefix}_out_weights.mem",
        f"mem_files/{prefix}_out_bias.mem",
    )


def generate_proj_mem_files(
    layers: List[LayerInfo],
    mem_dir: Path,
    scale: int,
    bit_width: int,
    *,
    debug_mem: bool = False,
) -> None:
    """Write all .mem files for the FC chain block structure.

    Each block b produces four ROM files:
      - fc[_N]_proj_in_weights.mem  — packed int8 weights for the first half (fc_in)
      - fc[_N]_proj_in_bias.mem     — packed acc_t biases for fc_in (sign-extended int8)
      - fc[_N]_proj_out_weights.mem — packed int8 weights for the second half (fc_out)
      - fc[_N]_proj_out_bias.mem    — per-neuron acc_t biases for fc_out (quantized from float)

    Layer pairing: block 0 = (fc1, fc2), block 1 = (fc3, fc4), ...
    If N is odd the last block pairs fcN with a 1x1 identity weight matrix.
    """
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    mem_dir.mkdir(parents=True, exist_ok=True)
    # Pair consecutively: (fc1,fc2), (fc3,fc4), ...; odd N: last block has (fcN, None) -> 1x1 proj_out
    n = len(linear_layers)
    num_blocks = (n + 1) // 2
    block_indices = [
        (2 * b, 2 * b + 1) if 2 * b + 1 < n else (2 * b, None)
        for b in range(num_blocks)
    ]
    for block_idx, (layer_idx, next_idx) in enumerate(block_indices):
        layer = linear_layers[layer_idx]
        next_layer = linear_layers[next_idx] if next_idx is not None else None
        if next_layer is not None and int(layer.out_features or 0) != int(next_layer.in_features or 0):
            raise RuntimeError(
                f"Block interface mismatch: {layer.name}.out_features={layer.out_features} "
                f"!= {next_layer.name}.in_features={next_layer.in_features}"
            )
        prefix = _proj_prefix(block_idx)
        wq, bq, wtag = mem_export_weight_matrix(layer, bit_width, scale)
        in_f = layer.in_features or 0
        out_f = layer.out_features or 0

        if debug_mem:
            qp = layer.quant_params or {}
            log_mem_export_debug(
                f"{layer.name}->{prefix}_in",
                wq,
                bq,
                wtag,
                bit_width=bit_width,
                onnx_weight_scale=qp.get("weight_scale") if isinstance(qp, dict) else None,
                onnx_zp=qp.get("b_zero_point") if isinstance(qp, dict) else None,
            )

        # proj_in: weights + bias (per layer block, as in the FC chain)
        generate_quant_pkg_style_weight_mem(wq, mem_dir / f"{prefix}_in_weights.mem", "fc1", in_f, out_f, bit_width)
        generate_proj_bias_mem_acc(bq, mem_dir / f"{prefix}_in_bias.mem", out_f, bit_width)

        # proj_out: next layer weights + bias (per layer block, as in the FC chain)
        if next_layer is not None:
            next_wq, next_bq, ntag = mem_export_weight_matrix(next_layer, bit_width, scale)
            next_in = next_layer.in_features or 0
            next_out = next_layer.out_features or 0
            if debug_mem:
                nqp = next_layer.quant_params or {}
                log_mem_export_debug(
                    f"{next_layer.name}->{prefix}_out",
                    next_wq,
                    next_bq,
                    ntag,
                    bit_width=bit_width,
                    onnx_weight_scale=nqp.get("weight_scale") if isinstance(nqp, dict) else None,
                    onnx_zp=nqp.get("b_zero_point") if isinstance(nqp, dict) else None,
                )
            generate_quant_pkg_style_weight_mem(next_wq, mem_dir / f"{prefix}_out_weights.mem", "fc2", next_in, next_out, bit_width)
            if layer.qdq_fout_exp is None:
                raise RuntimeError(
                    f"{layer.name}: qdq_fout_exp required for proj_out bias (paired block fin_act_exp)"
                )
            fin_act_exp = abs(int(layer.qdq_fout_exp))
            next_bq_acc = proj_out_bias_rom_values(
                next_layer,
                bit_width,
                fin_act_exp=fin_act_exp,
                mem_label=f"{prefix}_out_bias.mem",
            )
            generate_proj_out_bias_mem(next_bq_acc, mem_dir / f"{prefix}_out_bias.mem", bit_width)
        else:
            next_out = 1
            w_out = np.zeros((1, out_f), dtype=np.int32)
            if out_f >= 1:
                w_out[0, 0] = 1
            generate_quant_pkg_style_weight_mem(w_out, mem_dir / f"{prefix}_out_weights.mem", "fc2", out_f, next_out, bit_width)
            if layer.qdq_fout_exp is None:
                raise RuntimeError(
                    f"{layer.name}: qdq_fout_exp required for proj_out bias (tail block fin_act_exp)"
                )
            fin_act_exp = abs(int(layer.qdq_fout_exp))
            bq_out_acc = proj_out_bias_rom_values(
                layer,
                bit_width,
                fin_act_exp=fin_act_exp,
                mem_label=f"{prefix}_out_bias.mem",
            )
            generate_proj_out_bias_mem(bq_out_acc, mem_dir / f"{prefix}_out_bias.mem", bit_width)


def generate_quant_pkg_style_mems(
    layer_name: str,
    weight_matrix: np.ndarray,
    bias_vector: np.ndarray,
    bit_width: int,
    rtl_root: Path,
) -> None:
    """Generate mem files for legacy flow."""
    if bit_width not in (4, 8, 16):
        raise ValueError(f"Unsupported bit width: {bit_width}. Supported: 4, 8, 16")

    mem_dir = rtl_root / "mem"
    mem_dir.mkdir(parents=True, exist_ok=True)

    weight_matrix = np.asarray(weight_matrix, dtype=np.int32)
    bias_vector = np.asarray(bias_vector, dtype=np.int32)

    out_features, in_features = weight_matrix.shape
    mask = (1 << bit_width) - 1

    weights_path = mem_dir / f"{layer_name}_weights_{bit_width}.mem"
    with weights_path.open("w", encoding="utf-8") as wf:
        if layer_name == "fc1":
            num_neurons = out_features
            total_bits = num_neurons * bit_width
            hex_width = total_bits // 4
            for j in range(in_features):
                packed = 0
                for neuron_idx in range(num_neurons):
                    val = int(weight_matrix[neuron_idx, j]) & mask
                    packed |= val << (neuron_idx * bit_width)
                wf.write(f"{packed:0{hex_width}X}\n")
        else:
            num_inputs = in_features
            total_bits = num_inputs * bit_width
            hex_width = total_bits // 4
            for neuron_idx in range(out_features):
                packed = 0
                for inp_idx in range(num_inputs):
                    val = int(weight_matrix[neuron_idx, inp_idx]) & mask
                    packed |= val << (inp_idx * bit_width)
                wf.write(f"{packed:0{hex_width}X}\n")

    biases_path = mem_dir / f"{layer_name}_biases_{bit_width}.mem"
    num_neurons = bias_vector.shape[0]
    total_bits = num_neurons * bit_width
    hex_width = (total_bits + 3) // 4
    packed_biases = 0
    for neuron_idx in range(num_neurons):
        val = int(bias_vector[neuron_idx]) & mask
        packed_biases |= val << (neuron_idx * bit_width)
    with biases_path.open("w", encoding="utf-8") as bf:
        bf.write(f"{packed_biases:0{hex_width}X}\n")


def emit_legacy_rtl_outputs(
    legacy_rtl_dir: Path,
    layers: List[LayerInfo],
    scale: int,
    bits_list: Sequence[int] = (4, 8, 16),
    write_sv: bool = True,
) -> None:
    """Emit legacy rtl/ flow outputs (mem files only; write_sv not supported)."""
    legacy_rtl_dir = legacy_rtl_dir.resolve()

    if write_sv:
        LOGGER.warning("Legacy SV file writing not supported in minimal rtl_mapper; emitting mem files only.")

    for layer in layers:
        if layer.layer_type != "linear":
            continue

        if layer.weight is None:
            raise RuntimeError(f"Linear layer {layer.name} missing weights")

        weight_np = np.asarray(layer.weight, dtype=np.float32)
        bias_np = (
            np.asarray(layer.bias, dtype=np.float32)
            if layer.bias is not None
            else np.zeros((weight_np.shape[0],), dtype=np.float32)
        )

        for bits in bits_list:
            wq = float_to_int(weight_np, scale, bits)
            bq = float_to_int(bias_np, scale, bits)
            generate_quant_pkg_style_mems(layer.name, wq, bq, bits, legacy_rtl_dir)


# =============================================================================
# Embedded SystemVerilog Module Templates
#
# The SV source for reusable building-block modules is stored as Python string
# constants (MAC_SV, FC_IN_SV, etc.) rather than
# external template files.  This keeps the generator self-contained — no
# separate template directory is needed.
#
# Each string is written to disk by ``_write_embedded_sv`` with a PyramidTech
# header prepended.  The quant_pkg is generated programmatically (to
# parameterise Q_WIDTH at compile time).
#
# Module overview (hardware datapath, in order):
#   quant_pkg  → widths, types, saturation constants shared by all modules
#   mac        → pipelined multiply-accumulate (3-stage: multiply, accumulate, saturate)
#   fc_in      → instantiates N MACs, adds bias (<<< BIAS_SCALE), shifts (>>> LAYER_SCALE)
#   fc_out     → single-output FC with 3-stage pipeline (mult, add+scale, saturate)
#   relu_layer → max(0, x), registered output
#   sync_fifo  → pointer-based FIFO for AXI4-Stream back-pressure
# =============================================================================

def _get_quant_pkg_content(weight_width: int) -> str:
    """Generate quant_pkg.sv."""
    if weight_width not in (4, 8, 16):
        weight_width = 16
    body = '''package quant_pkg;

  timeunit 1ns;
  timeprecision 1ps;

  // =============================================================
  // Quantization mode (select ONE at compile time)
  // =============================================================
  `ifdef Q_INT4
    localparam int Q_WIDTH = 32'd4;
  `elsif Q_INT8
    localparam int Q_WIDTH = 32'd8;
  `elsif Q_INT16
    localparam int Q_WIDTH = 32'd16;
  `else
    // Default (safety)
    localparam int Q_WIDTH = 32'd8;
  `endif

  // =============================================================
  // Common fixed-point types
  // =============================================================
  typedef logic signed [Q_WIDTH-1:0]       q_data_t;   // Quantized data
  typedef logic signed [2*Q_WIDTH-1:0]     q_mult_t;   // Multiply result
  typedef logic signed [4*Q_WIDTH-1:0]     acc_t;      // Accumulator

  //Widths of prediction axi4 stream
  localparam int DATA_WIDTH = 32'd32;
  localparam int KEEP_WIDTH = 32'd4;
  

  // =============================================================
  // Method A: Narrow limits (Q-width), then cast to acc_t
  // Use when accumulator must saturate to Q range
  // =============================================================
  localparam q_data_t Q_MAX = {1'b0, {Q_WIDTH-1{1'b1}}};
  localparam q_data_t Q_MIN = {1'b1, {Q_WIDTH-1{1'b0}}};

  localparam acc_t ACC_Q_MAX = acc_t'(Q_MAX);
  localparam acc_t ACC_Q_MIN = acc_t'(Q_MIN);

  // =============================================================
  // Method B: Native accumulator limits (full acc_t width)
  // Use when accumulator keeps full dynamic range
  // =============================================================
  localparam acc_t ACC_FULL_MAX = {1'b0, {4*Q_WIDTH-1{1'b1}}};
  localparam acc_t ACC_FULL_MIN = {1'b1, {4*Q_WIDTH-1{1'b0}}};

  // =============================================================
  // Activation Function Limits
  // =============================================================
  localparam q_data_t SIGMOID_MAX = 1 << (Q_WIDTH - 2); 
  localparam q_data_t SIGMOID_MIN = 8'h0;

endpackage: quant_pkg
'''
    desc = "Package defining quantization widths, fixed-point data types,\n                and saturation limits for accumulation and activation functions."
    return _pyramidtech_wrap(body, "quant_pkg.sv", desc, quant_pkg_style=True)
MAC_SV = '''module mac
  import quant_pkg::*;
#(
  parameter int INPUT_SIZE    = 16
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  logic enable_i,      // Enable MAC operation

  input  q_data_t a_i,        // Signed multiplicand
  input  q_data_t b_i,        // Signed multiplier

  output acc_t acc_o,         // Saturated accumulator output
  output logic valid_o        // Output valid (aligned with acc)
);

  timeunit 1ns;
  timeprecision 1ps;

  // ------------------------------------------------------------
  // Internal signals
  // ------------------------------------------------------------
  q_mult_t mult_q;            // Registered multiplication result (Stage 1)
  acc_t    acc_q;             // Internal accumulator register (Stage 2)

  logic    enable_q;          // Pipeline stage 1 enable
  logic    enable_q2;         // Pipeline stage 2 enable

  logic [$clog2(INPUT_SIZE + 1) - 1:0] count_q;   //Counter of input data
  logic clear;
  logic clear_q;             // Pipeline stage 1 clear
  logic clear_q2;            // Pipeline stage 2 clear


  assign clear = (count_q == INPUT_SIZE) ? 1'b1 : 1'b0;

  // ------------------------------------------------------------
  // Stage 1: Multiply
  // ------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin: multiply_stage
    if (!rst_n_i) begin
      mult_q  <= '0;
      count_q <= '0;
    end
    else if (clear) begin
      mult_q  <= '0;
      count_q <= '0;
    end
    else if (enable_i) begin
      mult_q  <= $signed(a_i) * $signed(b_i);
      count_q <= count_q + 1;
    end
  end: multiply_stage

  // ------------------------------------------------------------
  // Stage 2: Accumulate
  // ------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin: accumulate_stage
    if (!rst_n_i) begin
      acc_q <= '0;
    end
    else if (clear_q) begin
      acc_q <= '0;
    end
    else if (enable_q) begin
      acc_q <= acc_q + mult_q;
    end
  end: accumulate_stage

  // ------------------------------------------------------------
  // Stage 3: Output saturation to ACC full range
  // ------------------------------------------------------------
  always_ff @(posedge clk_i) begin: saturation_stage
    if (acc_q > $signed(ACC_FULL_MAX)) begin
      acc_o <= ACC_FULL_MAX;
    end
    else if (acc_q < $signed(ACC_FULL_MIN)) begin
      acc_o <= ACC_FULL_MIN;
    end
    else begin
      // Truncate/slice using explicit width literals
      acc_o <= acc_q[4*Q_WIDTH-1:0];
    end
  end: saturation_stage

  // ------------------------------------------------------------
  // Valid signal pipeline (matches MAC latency)
  // ------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin: valid_pipeline
    if (!rst_n_i) begin
      enable_q   <= 1'b0;
      enable_q2  <= 1'b0;
      valid_o    <= 1'b0;
      clear_q    <= 1'b0;
      clear_q2   <= 1'b0;
    end
    else if (clear_q2) begin
      valid_o    <= 1'b0;
      enable_q   <= enable_i;
      enable_q2  <= enable_q;
      clear_q2   <= 1'b0;
    end
    else begin
      enable_q   <= enable_i;
      enable_q2  <= enable_q;
      valid_o    <= enable_q2;
      clear_q    <= clear;
      clear_q2   <= clear_q;
    end
  end: valid_pipeline

endmodule: mac
'''

FC_IN_SV = '''// MAC (F_mac) + bias: bias_aligned = biases_i <<< BIAS_SCALE (see quantize_bias_for_rtl).
module fc_in
  import quant_pkg::*;
#(
  parameter int    NUM_NEURONS = 8,
  parameter int    INPUT_SIZE  = 16,
  parameter signed BIAS_SCALE  = 0,
  parameter signed LAYER_SCALE = 12
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  logic valid_i,                  // Input data valid
  input  q_data_t data_i,                // Input data
  input  q_data_t weights_i[NUM_NEURONS], // Weight vector per neuron
  input  acc_t    biases_i[NUM_NEURONS],  // Biases vector per neuron

  output q_data_t data_o[NUM_NEURONS],    // FC layer output
  output logic    valid_o                // Output valid pulse
);

  timeunit 1ns;
  timeprecision 1ps;

  // ------------------------------------------------------------
  // Internal signals
  // ------------------------------------------------------------
  acc_t mac_acc_q[NUM_NEURONS];          // Registered MAC outputs
  acc_t acc_tmp_s[NUM_NEURONS];          // Combinational bias addition
  acc_t bias_aligned_s[NUM_NEURONS];     // Combinational bias alignment
  acc_t data_out_temp_s[NUM_NEURONS];    // Combinational layer scaling

  logic [$clog2(INPUT_SIZE + 1) - 1:0] count_q;

  logic mac_enable_s;
  logic [NUM_NEURONS- 1:0] mac_valid_q;
  logic all_mac_valid_s;

  // ------------------------------------------------------------
  // Instantiate MAC units
  // ------------------------------------------------------------
  genvar i;
  generate
    for (i = 0; i < NUM_NEURONS; i++) begin : gen_mac_units
      mac #(
          .INPUT_SIZE(INPUT_SIZE)
      ) u_mac (
        .clk_i     (clk_i),
        .rst_n_i     (rst_n_i),
        .enable_i  (mac_enable_s),
        .a_i       (data_i),
        .b_i       (weights_i[i]),
        .valid_o   (mac_valid_q[i]),
        .acc_o     (mac_acc_q[i])
      );
    end
  endgenerate

  assign all_mac_valid_s = &mac_valid_q;
  assign mac_enable_s    = valid_i;

  // ------------------------------------------------------------
  // Count valid inputs and generate valid_o for last input
  // ------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin : count_logic
    if (!rst_n_i) begin
      count_q <= '0;
      valid_o <= 1'b0;
    end
    else begin
      valid_o <= 1'b0;
      if (all_mac_valid_s) begin
        if (count_q == (INPUT_SIZE - 1)) begin
          count_q <= '0;
          valid_o <= 1'b1;
        end
        else begin
          count_q <= count_q + 1'b1;
        end
      end
    end
  end : count_logic

  // ------------------------------------------------------------
  // Bias addition, layer scaling, and output saturation
  // ------------------------------------------------------------
  generate
    for (i = 0; i < NUM_NEURONS; i++) begin : gen_output_path
      // Align bias to accumulator width
      assign bias_aligned_s[i] = (BIAS_SCALE >= 0) ?
                                 (biases_i[i] <<< BIAS_SCALE) :
                                 (biases_i[i] >>> -BIAS_SCALE);

      // Add bias to MAC output
      assign acc_tmp_s[i] = mac_acc_q[i] + bias_aligned_s[i];

      // Apply layer scaling
      assign data_out_temp_s[i] = (LAYER_SCALE >= 0) ?
                                  (acc_tmp_s[i] >>> LAYER_SCALE) :
                                  (acc_tmp_s[i] <<< -LAYER_SCALE);

      // Register output with saturation logic
      always_ff @(posedge clk_i or negedge rst_n_i) begin : saturation_reg
        if (!rst_n_i) begin
          data_o[i] <= '0;
        end
        else if (count_q == (INPUT_SIZE-1)) begin
          if (data_out_temp_s[i] > ACC_Q_MAX) begin
            data_o[i] <= Q_MAX;
          end
          else if (data_out_temp_s[i] < ACC_Q_MIN) begin
            data_o[i] <= Q_MIN;
          end
          else begin
            // Cast to narrow quantized data type
            data_o[i] <= data_out_temp_s[i][Q_WIDTH-1:0];
          end
        end
      end : saturation_reg
    end
  endgenerate

endmodule : fc_in
'''

FC_OUT_SV = '''module fc_out 
  import quant_pkg::*;
#(
  parameter int    NUM_NEURONS = 2,
  parameter signed LAYER_SCALE = 5,
  parameter signed BIAS_SCALE  = 1
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  logic    valid_i,                   // Input valid
  input  q_data_t data_i[NUM_NEURONS],       // Inputs to this FC layer
  input  q_data_t weights_i[NUM_NEURONS],    // Weight vector
  input  acc_t    bias_i,                    // Bias

  output q_data_t data_o,                    // FC output
  output logic    valid_o                    // Output valid
);

  timeunit 1ns;
  timeprecision 1ps;

  // ------------------------------------------------------------
  // Internal signals
  // ------------------------------------------------------------
  acc_t mult_res_s[NUM_NEURONS];
  acc_t bias_q;

  // Pipeline Stage 1 Registers
  acc_t mult_q[NUM_NEURONS];
  logic valid_q;

  // Stage 2 Combinational Signals
  acc_t sum_stage2_s;
  acc_t sum_stage2_tmp_s;

  // Pipeline Stage 2 Registers
  acc_t sum_q2;
  logic valid_q2;

  // ============================================================
  // Stage 0/1: Multipliers and Input Registration
  // ============================================================
  genvar i;
  generate
    for (i = 0; i < NUM_NEURONS; i = i + 1) begin : gen_mult
      assign mult_res_s[i] = $signed(data_i[i]) * $signed(weights_i[i]);
    end
  endgenerate

  always_ff @(posedge clk_i or negedge rst_n_i) begin : stage1_regs
    if (!rst_n_i) begin
      for (int j = 0; j < NUM_NEURONS; j++) begin
        mult_q[j] <= '0;
      end
      valid_q <= 1'b0;
      bias_q  <= '0;
    end 
    else begin
      for (int j = 0; j < NUM_NEURONS; j++) begin
        mult_q[j] <= mult_res_s[j];
      end
      valid_q <= valid_i;
      bias_q  <= bias_i;
    end
  end : stage1_regs

  // ============================================================
  // Stage 2: Adder + Bias + Layer scaling
  // ============================================================
  always_comb begin : stage2_logic
    if (BIAS_SCALE >= 0) begin
      sum_stage2_tmp_s = bias_q <<< BIAS_SCALE; 
    end
    else begin
      sum_stage2_tmp_s = bias_q >>> -BIAS_SCALE;
    end

    for (int j = 0; j < NUM_NEURONS; j++) begin
      sum_stage2_tmp_s += mult_q[j];
    end

    if (LAYER_SCALE >= 0) begin
      sum_stage2_s = sum_stage2_tmp_s >>> LAYER_SCALE;
    end
    else begin
      sum_stage2_s = sum_stage2_tmp_s <<< -LAYER_SCALE;
    end
  end : stage2_logic

  always_ff @(posedge clk_i or negedge rst_n_i) begin : stage2_regs
    if (!rst_n_i) begin
      sum_q2   <= '0;
      valid_q2 <= 1'b0;
    end 
    else begin
      sum_q2   <= sum_stage2_s;
      valid_q2 <= valid_q;
    end
  end : stage2_regs

  // ============================================================
  // Stage 3: Saturation + Output Register
  // ============================================================
  always_ff @(posedge clk_i or negedge rst_n_i) begin : stage3_regs
    if (!rst_n_i) begin
      data_o  <= '0;
      valid_o <= 1'b0;
    end 
    else begin
      valid_o <= valid_q2;
      if (sum_q2 > ACC_Q_MAX) begin
        data_o <= Q_MAX;
      end
      else if (sum_q2 < ACC_Q_MIN) begin
        data_o <= Q_MIN;
      end
      else begin
        data_o <= sum_q2[Q_WIDTH- 1:0];
      end
    end
  end : stage3_regs

endmodule : fc_out
'''

RELU_LAYER_SV = '''module relu_layer 
  import quant_pkg::*;
(
  input  logic    clk_i,
  input  logic    rst_n_i,

  input  logic    valid_i,               // Input data valid
  input  q_data_t data_i,                // Input quantized data

  output q_data_t data_o,                // ReLU activated output
  output logic    valid_o                // Output valid signal
);

  timeunit 1ns;
  timeprecision 1ps;

  // ============================================================
  // ReLU logic with pipelined valid signal
  // ============================================================
  always_ff @(posedge clk_i or negedge rst_n_i) begin : relu_pipeline
    if (!rst_n_i) begin
      data_o  <= '0;
      valid_o <= 1'b0;
    end 
    else begin
      valid_o <= valid_i;
      if (valid_i) begin
        // Rectified Linear Unit logic: output = max(0, input)
        data_o <= (data_i < 0) ? 0 : data_i;
      end
    end
  end : relu_pipeline

endmodule : relu_layer
'''

SYNC_FIFO_SV = '''module sync_fifo #(
  parameter int DATA_WIDTH = 32,
  parameter int DEPTH      = 1024
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  logic                  write_en_i,
  input  logic [DATA_WIDTH-1:0] write_data_i,
  output logic                  full_o,

  input  logic                  read_en_i,
  output logic [DATA_WIDTH-1:0] read_data_o,
  output logic                  empty_o
);

  timeunit 1ns;
  timeprecision 1ps;

  localparam int ADDR_WIDTH = $clog2(DEPTH);

  logic [DATA_WIDTH-1:0] fifo_mem_q [0:DEPTH-1];
  logic [ADDR_WIDTH:0] write_ptr_q;
  logic [ADDR_WIDTH:0] read_ptr_q;

  always_ff @(posedge clk_i or negedge rst_n_i) begin : write_ptr_logic
    if (!rst_n_i) begin
      write_ptr_q <= '0;
    end
    else if (write_en_i && !full_o) begin
      write_ptr_q <= write_ptr_q + 1'b1;
    end
  end : write_ptr_logic

  always_ff @(posedge clk_i) begin : write_mem_logic
    if (write_en_i && !full_o) begin
      fifo_mem_q[write_ptr_q[ADDR_WIDTH-1:0]] <= write_data_i;
    end
  end : write_mem_logic

  always_ff @(posedge clk_i or negedge rst_n_i) begin : read_logic
    if (!rst_n_i) begin
      read_ptr_q <= '0;
      read_data_o <= '0;
    end
    else if (read_en_i && !empty_o) begin
      read_data_o <= fifo_mem_q[read_ptr_q[ADDR_WIDTH-1:0]];
      read_ptr_q <= read_ptr_q + 1'b1;
    end
  end : read_logic

  assign empty_o = (write_ptr_q == read_ptr_q);
  assign full_o  = (write_ptr_q[ADDR_WIDTH] != read_ptr_q[ADDR_WIDTH]) &&
                   (write_ptr_q[ADDR_WIDTH-1:0] == read_ptr_q[ADDR_WIDTH-1:0]);

endmodule : sync_fifo
'''

def _linear_layers_have_onnx_pair_qf_tensors(linear_layers: List[LayerInfo]) -> bool:
    if not linear_layers:
        return False
    for layer in linear_layers:
        if (
            layer.onnx_pair_float_weight is None
            or layer.onnx_pair_quant_weight is None
            or layer.onnx_pair_float_bias is None
            or layer.onnx_pair_quant_bias is None
        ):
            return False
    return True


def _layer_has_full_onnx_pair(layer: LayerInfo) -> bool:
    return (
        layer.onnx_pair_float_weight is not None
        and layer.onnx_pair_quant_weight is not None
        and layer.onnx_pair_float_bias is not None
        and layer.onnx_pair_quant_bias is not None
    )


def _fw_fb_pair_exponents_from_onnx_pair(layer: LayerInfo, fin_qdq: int) -> Tuple[int, int]:
    """Fw/Fb exponents for ``_qdq_pair_alignment`` / ``FC_*_*_BIAS_SCALE`` — same as ``_build_qdq_pair_interlayer_rows``."""
    fw = _fw_exponent_from_onnx(layer.onnx_pair_quant_weight, layer.onnx_pair_float_weight)
    Bf = np.asarray(layer.onnx_pair_float_bias, dtype=np.float64).ravel()
    ws_eff: Optional[np.ndarray]
    if layer.onnx_pair_weight_scale_arr is not None:
        ws_eff = np.asarray(layer.onnx_pair_weight_scale_arr, dtype=np.float64).reshape(-1)
    elif layer.onnx_pair_weight_scale is not None:
        ws_eff = np.asarray([float(layer.onnx_pair_weight_scale)], dtype=np.float64)
    else:
        ws_eff = None
    bq = np.asarray(layer.onnx_pair_quant_bias, dtype=np.float64).ravel()
    fb = _fb_exponent_from_onnx(Bf, bq, fin_qdq, fw, w_scale=ws_eff)
    return fw, fb


def _align_quant_float_weights(qw: np.ndarray, fw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    qw = np.asarray(qw)
    fw = np.asarray(fw)
    if qw.shape == fw.shape:
        return qw, fw
    if qw.shape == (fw.shape[1], fw.shape[0]):
        return np.ascontiguousarray(qw.T), fw
    if fw.shape == (qw.shape[1], qw.shape[0]):
        return qw, np.ascontiguousarray(fw.T)
    raise ValueError(f"Weight shape mismatch for Fw extraction: quant {qw.shape} vs float {fw.shape}")


def _mode_round_log2_ratio(q: np.ndarray, f: np.ndarray) -> Optional[int]:
    """Find the dominant power-of-two ratio between quantized and float arrays.

    For each element where both q and f are non-zero, compute round(log2(|q|/|f|))
    and return the statistical mode.  This gives the exponent F such that
    q ≈ f * 2^F for the majority of elements — i.e. the quantization gain.
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    f = np.asarray(f, dtype=np.float64).ravel()
    if q.size != f.size:
        return None
    m = (f != 0) & (q != 0)
    if not np.any(m):
        return None
    s = np.abs(q[m]) / np.abs(f[m])
    r = np.rint(np.log2(s)).astype(np.int64)
    uniq, counts = np.unique(r, return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _fw_exponent_from_onnx(qw: np.ndarray, fw: np.ndarray) -> int:
    """Weight exponent from quant vs float ONNX tensors (mode of round(log2(|q|/|f|)))."""
    qw, fw = _align_quant_float_weights(qw, fw)
    m = _mode_round_log2_ratio(qw, fw)
    return 0 if m is None else m


def _fb_exponent_from_onnx(
    fb: np.ndarray,
    bq: np.ndarray,
    fin: int,
    fw: int,
    *,
    w_scale: Optional[np.ndarray],
) -> int:
    fb = np.asarray(fb, dtype=np.float64).ravel()
    bq = np.asarray(bq, dtype=np.float64).ravel()
    m_pair = _mode_round_log2_ratio(bq, fb)
    if m_pair is not None:
        return int(m_pair)
    ws = None if w_scale is None else np.asarray(w_scale, dtype=np.float64).reshape(-1)
    if ws is None or ws.size == 0:
        return fin + fw
    ws = ws[np.isfinite(ws) & (ws > 0)]
    if ws.size == 0:
        return fin + fw
    if ws.size == 1:
        ws_mode = float(ws[0])
    else:
        l2 = np.rint(np.log2(ws)).astype(np.int64)
        uniq, counts = np.unique(l2, return_counts=True)
        ws_mode = float(2.0 ** int(uniq[int(np.argmax(counts))]))
    s_act = 2.0 ** (-fin)
    s_mac = s_act * ws_mode
    b_int = np.rint(fb / s_mac).astype(np.float64)
    if not np.any(fb != 0) and not np.any(b_int != 0):
        return fin + fw
    if np.all(np.abs(b_int) < 1e-12) and np.any(fb != 0):
        return fin + fw
    m = _mode_round_log2_ratio(b_int, fb)
    if m is None:
        return fin + fw
    return m


def _qdq_pair_bias_alignment(fin: int, fw: int, fb: int) -> Tuple[int, str]:
    """BIAS_SCALE row: compare F_mac = Fin+Fw with Fb."""
    fmac = fin + fw
    if fmac > fb:
        return fmac - fb, "Left"
    if fmac < fb:
        return fb - fmac, "Right"
    return 0, "NA"


def _qdq_pair_layer_alignment(fin: int, fw: int, fout: int) -> Tuple[int, str]:
    """LAYER_SCALE row: F_mac = Fin+Fw vs Fout. Use Fw from ``quantize_weight_for_rtl`` (ROM weight plane)."""
    fmac = fin + fw
    if fmac > fout:
        return fmac - fout, "Right"
    if fmac < fout:
        return fout - fmac, "Left"
    return 0, "NA"


def _qdq_pair_alignment(
    fin: int, fw: int, fb: int, fout: int
) -> Tuple[int, str, int, str]:
    """Single Fw for both rows (legacy / interlayer rows where ONNX Fw drives everything)."""
    bs, bd = _qdq_pair_bias_alignment(fin, fw, fb)
    ls, ld = _qdq_pair_layer_alignment(fin, fw, fout)
    return bs, bd, ls, ld


def _rtl_signed_bias_scale(bias_scale: int, bias_dir: str) -> int:
    if bias_dir == "NA" or bias_scale == 0:
        return 0
    if bias_dir == "Left":
        return bias_scale
    if bias_dir == "Right":
        return -bias_scale
    return 0


def _rtl_signed_layer_scale(layer_scale: int, layer_dir: str) -> int:
    if layer_dir == "NA" or layer_scale == 0:
        return 0
    if layer_dir == "Right":
        return layer_scale
    if layer_dir == "Left":
        return -layer_scale
    return 0


def _build_qdq_pair_interlayer_rows(
    linear_layers: List[LayerInfo],
) -> List[Dict[str, Any]]:
    """Build per-layer alignment rows using ONNX-pair Fw/Fb and QDQ Fin/Fout.

    Each row contains the four exponents (Fin, Fw, Fb, Fout) plus the derived
    BIAS_SCALE and LAYER_SCALE shift values.  Used by the legacy
    ``compute_fc_layer_scales`` path.
    """
    rows: List[Dict[str, Any]] = []
    for i, layer in enumerate(linear_layers):
        if layer.qdq_fin_exp is None or layer.qdq_fout_exp is None:
            raise RuntimeError(
                f"{layer.name}: missing qdq_fin_exp / qdq_fout_exp for inter-layer scale rows."
            )
        fin = abs(int(layer.qdq_fin_exp))
        fout = abs(int(layer.qdq_fout_exp))
        assert layer.onnx_pair_quant_weight is not None and layer.onnx_pair_float_weight is not None
        assert layer.onnx_pair_float_bias is not None
        fw = _fw_exponent_from_onnx(layer.onnx_pair_quant_weight, layer.onnx_pair_float_weight)
        if layer.onnx_pair_weight_scale_arr is not None:
            w_scale_arr = np.asarray(layer.onnx_pair_weight_scale_arr, dtype=np.float64).reshape(-1)
        elif layer.onnx_pair_weight_scale is not None:
            w_scale_arr = np.asarray([float(layer.onnx_pair_weight_scale)], dtype=np.float64)
        else:
            w_scale_arr = None
        fb_exp = _fb_exponent_from_onnx(
            layer.onnx_pair_float_bias,
            layer.onnx_pair_quant_bias,
            fin,
            fw,
            w_scale=w_scale_arr,
        )
        bsc, bdir, lsc, ldir = _qdq_pair_alignment(fin, fw, fb_exp, fout)
        rows.append(
            {
                "Fin": fin,
                "Fw": fw,
                "Fb": fb_exp,
                "Fout": fout,
                "biasScale": bsc,
                "biasDir": bdir,
                "layerScale": lsc,
                "layerDir": ldir,
            }
        )
    return rows


def compute_fc_layer_scales(
    linear_layers: List[LayerInfo],
    weight_width: int,
    python_scale: int = 256,
) -> Dict[str, int]:
    """Compute LAYER_SCALE and BIAS_SCALE per FC block.

    Fw and Fb exponents: only from ONNX pair (_fw_exponent_from_onnx / _fb_exponent_from_onnx).
    Fin/Fout: ``LayerInfo.qdq_fin_exp`` / ``qdq_fout_exp`` from quantized ONNX QDQ scales.

    ``python_scale`` is unused (API compatibility).

    Requires ``attach_inter_layer_scale_tensors_from_onnx_pair``.
    """
    del weight_width  # unused; kept for call-site compatibility
    del python_scale
    if not _linear_layers_have_onnx_pair_qf_tensors(linear_layers):
        raise RuntimeError(
            "Inter-layer scales require both ONNX models: set onnx_pair_float_weight/bias and "
            "onnx_pair_quant_weight/bias on each linear layer (quantized + float ONNX pair)."
        )
    rows = _build_qdq_pair_interlayer_rows(linear_layers)
    n = len(linear_layers)
    num_blocks = (n + 1) // 2
    result: Dict[str, int] = {}
    for b in range(num_blocks):
        in_idx = 2 * b
        out_idx = 2 * b + 1 if 2 * b + 1 < n else None
        rin = rows[in_idx]
        rout = rows[out_idx] if out_idx is not None else rin
        prefix = f"FC_{b + 1}"
        result[f"{prefix}_IN_LAYER_SCALE"] = _rtl_signed_layer_scale(rin["layerScale"], rin["layerDir"])
        result[f"{prefix}_IN_BIAS_SCALE"] = _rtl_signed_bias_scale(rin["biasScale"], rin["biasDir"])
        result[f"{prefix}_OUT_LAYER_SCALE"] = _rtl_signed_layer_scale(rout["layerScale"], rout["layerDir"])
        result[f"{prefix}_OUT_BIAS_SCALE"] = _rtl_signed_bias_scale(rout["biasScale"], rout["biasDir"])

    for k in list(result.keys()):
        raw_v = int(result[k])
        if raw_v < -24 or raw_v > 24:
            raise RuntimeError(f"Scale parameter {k} out of supported range [-24,24]: {raw_v}")
        result[k] = raw_v

    return result


# =============================================================================
# FC Chain Parameter Computation
#
# FC layers are grouped into "blocks" of two consecutive layers:
#   Block b = (fc_{2b+1}, fc_{2b+2})  →  fc_in_layer + fc_out_layer
# If the total number of FC layers N is odd, the last block pairs the final
# layer with a 1x1 identity projection so every block still has an in + out half.
#
# Per-block parameters (FC_b_NEURONS, FC_b_INPUT_SIZE, FC_b_ROM_DEPTH,
# FC_b_IN/OUT_LAYER_SCALE, FC_b_IN/OUT_BIAS_SCALE) are computed from ONNX
# layer shapes and the scale engine for substitution into the SV templates.
# =============================================================================

def _compute_fc_chain_params(linear_layers: List[LayerInfo], weight_width: int = 8, python_scale: int = 256) -> dict:
    """Compute FC chain parameters from ONNX layers.
    Pairs layers consecutively: block b = (fc_{2b+1}, fc_{2b+2}); odd N: last block has (fcN, 1x1).
    Returns dict with FC_1_*, FC_2_*, ... for num_blocks = ceil(N/2)."""
    n = len(linear_layers)
    num_blocks = (n + 1) // 2
    scales, _scale_tag = _fc_layer_scales_unified_or_legacy(
        linear_layers, weight_width, python_scale
    )
    result: Dict[str, Any] = {"FIFO_DEPTH": 1024}
    for b in range(num_blocks):
        in_idx = 2 * b
        out_idx = 2 * b + 1 if 2 * b + 1 < n else None
        in_layer = linear_layers[in_idx]
        out_layer = linear_layers[out_idx] if out_idx is not None else None
        in_f = in_layer.in_features or 0
        in_out = in_layer.out_features or 0
        rom_depth = (out_layer.out_features or 1) if out_layer else 1
        prefix = f"FC_{b + 1}"
        result[f"{prefix}_NEURONS"] = in_out
        result[f"{prefix}_INPUT_SIZE"] = in_f
        result[f"{prefix}_ROM_DEPTH"] = rom_depth
        result[f"{prefix}_IN_LAYER_SCALE"] = scales[f"{prefix}_IN_LAYER_SCALE"]
        result[f"{prefix}_IN_BIAS_SCALE"] = scales[f"{prefix}_IN_BIAS_SCALE"]
        result[f"{prefix}_OUT_LAYER_SCALE"] = scales[f"{prefix}_OUT_LAYER_SCALE"]
        result[f"{prefix}_OUT_BIAS_SCALE"] = scales[f"{prefix}_OUT_BIAS_SCALE"]
    return result


def _generate_fc_rom(
    out_dir: Path,
    rom_name: str,
    depth: int,
    width_neurons: int,
    weight_width: int,
) -> None:
    """Generate one ROM .sv module that loads its data from a .mem file via $readmemh.

    Each ROM is a simple synchronous read-port wrapper around a memory array.
    The WIDTH parameter is set to the packed bit-width of one ROM row (all
    neurons * Q_WIDTH for weights, or acc_t for biases).  DEPTH is the number
    of addresses (= input features for proj_in weights, or output neurons for
    proj_out weights).
    """
    base = rom_name.replace("_weights_rom", "").replace("_bias_rom", "")
    if "bias" in rom_name:
        mem_basename = f"{base}_bias.mem"
    else:
        mem_basename = f"{base}_weights.mem"
    packed_width = width_neurons * weight_width
    if "bias" in rom_name and "in" in rom_name:
        packed_width = width_neurons * weight_width * 4  # acc_t per neuron for fc_in bias
    elif "bias" in rom_name and "out" in rom_name:
        packed_width = weight_width * 4  # acc_t for fc_out bias
    body = f"""module {rom_name} #(
  parameter int DEPTH  = {depth},
  parameter int WIDTH  = {packed_width},
  parameter int ADDR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
  input  logic              clk_i,
  input  logic [ADDR_W-1:0] addr_i,
  output logic [WIDTH-1:0]  data_o
);

  timeunit 1ns;
  timeprecision 1ps;

  (* rom_style = "block" *)
  logic [WIDTH-1:0] mem [0:DEPTH-1];

  initial begin
    $readmemh({{`__FILE__, "/../mem_files/{mem_basename}"}}, mem);
  end

  always_ff @(posedge clk_i) begin : read_port
    data_o <= mem[addr_i];
  end : read_port

endmodule : {rom_name}
"""
    rom_type = "biases" if "bias" in rom_name else "weights"
    rom_desc = f"rom of {rom_type} of {base} layer"
    (out_dir / f"{rom_name}.sv").write_text(_pyramidtech_wrap(body, f"{rom_name}.sv", rom_desc), encoding="utf-8")


def _generate_fc_in_layer_module(out_dir: Path, input_sizes: List[int], weight_width: int) -> None:
    """Generate fc_in_layer.sv — the ROM + compute wrapper for the first half of each block.

    Uses Verilog generate-if blocks keyed on ``INPUT_SIZE`` to instantiate the
    correct weight and bias ROMs for each block.  A single fc_in_layer module
    serves all blocks; the top module passes different INPUT_SIZE values to
    select the matching ROM at elaboration time.
    """
    branches = []
    for b, in_sz in enumerate(input_sizes):
        prefix = _proj_prefix(b)
        cond = "if" if b == 0 else "else if"
        branches.append(f"""    {cond} (INPUT_SIZE == {in_sz}) begin : gen_rom_{in_sz}
      {prefix}_in_weights_rom #(
        .DEPTH(INPUT_SIZE),
        .WIDTH(NUM_NEURONS * Q_WIDTH)
      ) u_weights_rom (
        .clk_i  (clk_i),
        .addr_i (weight_addr_q),
        .data_o (weight_rom_row_s)
      );

      {prefix}_in_bias_rom #(
        .DEPTH(1),
        .WIDTH(NUM_NEURONS * Q_WIDTH * 4)
      ) u_bias_rom (
        .clk_i  (clk_i),
        .addr_i (1'b0),
        .data_o (bias_rom_row_s)
      );
    end""")
    gen_block = " \n".join(branches)
    body = f"""`begin_keywords "1800-2012"
module fc_in_layer 
  import quant_pkg::*;
#(
  parameter int    NUM_NEURONS = 8,
  parameter int    INPUT_SIZE  = 16,
  parameter signed LAYER_SCALE = 12,
  parameter signed BIAS_SCALE  = 0
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  logic    valid_i,               // Input valid signal
  input  q_data_t data_i,                // Input data

  output q_data_t data_o[NUM_NEURONS],   // FC layer output
  output logic    valid_o                // Output valid pulse
);

  timeunit 1ns;
  timeprecision 1ps;

  // ------------------------------------------------------------
  // Internal signals and registers
  // ------------------------------------------------------------
  logic [$clog2(INPUT_SIZE)- 1:0] weight_addr_q;

  logic [NUM_NEURONS * Q_WIDTH - 1:0]   weight_rom_row_s;
  logic [NUM_NEURONS * Q_WIDTH*4 - 1:0] bias_rom_row_s;

  q_data_t weights_rom_data_s[NUM_NEURONS];
  acc_t    bias_rom_data_s[NUM_NEURONS];

  logic    valid_i_q;
  q_data_t data_i_q;

  // ------------------------------------------------------------
  // Input stage registers
  // ------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin : input_regs
    if (!rst_n_i) begin
      valid_i_q <= 1'b0;
      data_i_q  <= '0;
    end 
    else begin
      valid_i_q <= valid_i;
      data_i_q  <= data_i;
    end
  end : input_regs

  // ------------------------------------------------------------
  // Address counter for sequential ROM access
  // ------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin : addr_counter
    if (!rst_n_i) begin
      weight_addr_q <= '0;
    end
    else if (valid_i) begin
      weight_addr_q <= (weight_addr_q == (INPUT_SIZE - 1)) ? 
                       '0 : weight_addr_q + 1'b1;
    end
  end : addr_counter

  // ------------------------------------------------------------
  // Weight and Bias ROM Instantiations
  // ------------------------------------------------------------
  generate
{gen_block}
  endgenerate

  // ------------------------------------------------------------
  // ROM data unpacking
  // ------------------------------------------------------------
  genvar n;
  generate
    for (n = 0; n < NUM_NEURONS; n++) begin : gen_unpack
      assign weights_rom_data_s[n] = weight_rom_row_s[n * Q_WIDTH +: Q_WIDTH];
      assign bias_rom_data_s[n]    = bias_rom_row_s[n * Q_WIDTH * 4 +: Q_WIDTH * 4];
    end
  endgenerate

  // ------------------------------------------------------------
  // FC Compute Block Instance
  // ------------------------------------------------------------
  fc_in #(
    .NUM_NEURONS  (NUM_NEURONS),
    .INPUT_SIZE   (INPUT_SIZE),
    .LAYER_SCALE  (LAYER_SCALE),
    .BIAS_SCALE   (BIAS_SCALE)
  ) u_fc_in (
    .clk_i     (clk_i),
    .rst_n_i     (rst_n_i),
    .valid_i   (valid_i_q),
    .data_i    (data_i_q),
    .weights_i (weights_rom_data_s),
    .biases_i  (bias_rom_data_s),
    .data_o    (data_o),
    .valid_o   (valid_o)
  );

endmodule : fc_in_layer
`end_keywords"""
    desc = "Fully-connected (FC) input layer with sequential ROM access.\n                Reads weights and biases from ROM and feeds the compute block."
    (out_dir / "fc_in_layer.sv").write_text(_pyramidtech_wrap(body, "fc_in_layer.sv", desc), encoding="utf-8")


def _generate_fc_out_layer_module(out_dir: Path, rom_depths: List[int], weight_width: int) -> None:
    """Generate fc_out_layer.sv — the ROM + compute wrapper for the second half of each block.

    Uses Verilog generate-if blocks keyed on ``ROM_DEPTH`` to instantiate the
    correct weight and bias ROMs.  The fc_out compute block processes one
    output neuron at a time (sequential ROM reads), unlike fc_in which feeds
    all neurons in parallel.
    """
    branches = []
    for b, rom_d in enumerate(rom_depths):
        prefix = _proj_prefix(b)
        cond = "if" if b == 0 else "else if"
        addr_arg = "1'b0" if rom_d <= 1 else "weights_rom_addr_s"
        bias_addr = "1'b0" if rom_d <= 1 else "bias_rom_addr_s"
        branches.append(f"""    {cond} (ROM_DEPTH == {rom_d}) begin : gen_rom_{rom_d}
      {prefix}_out_weights_rom #(
        .DEPTH(ROM_DEPTH),
        .WIDTH(NUM_NEURONS * Q_WIDTH)
      ) u_weights_rom (
        .clk_i  (clk_i),
        .addr_i ({addr_arg}),
        .data_o (weights_rom_data_raw_s)
      );

      {prefix}_out_bias_rom #(
        .DEPTH(ROM_DEPTH),
        .WIDTH(Q_WIDTH*4)
      ) u_bias_rom (
        .clk_i  (clk_i),
        .addr_i ({bias_addr}),
        .data_o (bias_rom_data_s)
      );
    end""")
    gen_block = " \n".join(branches)
    body = f"""`begin_keywords "1800-2012"
module fc_out_layer 
  import quant_pkg::*;
#(
  parameter int    NUM_NEURONS = 2,
  parameter int    ROM_DEPTH   = 45,
  parameter signed LAYER_SCALE = 5,
  parameter signed BIAS_SCALE  = 1
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  logic    valid_i,                   // Input valid
  input  q_data_t data_i[NUM_NEURONS],       // FC inputs

  output q_data_t data_o,                    // FC output
  output logic    valid_o                    // Output valid
);

  timeunit 1ns;
  timeprecision 1ps;

  // ------------------------------------------------------------
  // Counters and flags
  // ------------------------------------------------------------
  logic [$clog2(ROM_DEPTH) - 1:0] addr_cnt_q;
  logic addr_last_s;

  // Valid pipeline signals
  logic valid_pipeline_q;
  logic valid_pipeline_q2;
  logic valid_out_engine_s;
  logic valid_out_engine_q;

  // ROM addresses
  logic [$clog2(ROM_DEPTH) - 1:0] weights_rom_addr_s;
  logic [$clog2(ROM_DEPTH) - 1:0] bias_rom_addr_s;

  // ------------------------------------------------------------
  // ROM data
  // ------------------------------------------------------------
  logic [NUM_NEURONS * Q_WIDTH - 1:0] weights_rom_data_raw_s;
  q_data_t weights_rom_data_s[NUM_NEURONS];
  acc_t    bias_rom_data_s;

  // Registered versions (pipeline suffixes)
  q_data_t weights_rom_data_q[NUM_NEURONS];
  acc_t    bias_rom_data_q;
  q_data_t data_i_q[NUM_NEURONS];

  // ------------------------------------------------------------
  // Address counter
  // ------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin : addr_counter
    if (!rst_n_i) begin
      addr_cnt_q <= '0;
    end
    else if (valid_pipeline_q) begin
      addr_cnt_q <= (addr_cnt_q == (ROM_DEPTH - 1)) ? 
                    '0 : addr_cnt_q + 1'b1;
    end
  end : addr_counter

  assign addr_last_s = (addr_cnt_q == (ROM_DEPTH - 1));

  // ------------------------------------------------------------
  // Valid pipeline
  // ------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin : pipeline_control
    if (!rst_n_i) begin
      valid_pipeline_q <= 1'b0;
    end
    else if (valid_i) begin
      valid_pipeline_q <= 1'b1;
    end
    else if (addr_last_s) begin
      valid_pipeline_q <= 1'b0;
    end
  end : pipeline_control

  // Extra stage to align with ROM pipeline
  always_ff @(posedge clk_i or negedge rst_n_i) begin : alignment_regs
    if (!rst_n_i) begin
      valid_pipeline_q2  <= 1'b0;
      valid_out_engine_q <= 1'b0;
    end
    else begin
      valid_pipeline_q2  <= valid_pipeline_q;
      valid_out_engine_q <= valid_out_engine_s;
    end
  end : alignment_regs

  assign valid_o           = valid_out_engine_q;
  assign weights_rom_addr_s = addr_cnt_q;
  assign bias_rom_addr_s    = addr_cnt_q;

  // ------------------------------------------------------------
  // Weight and Bias ROM Instantiations
  // ------------------------------------------------------------
  generate
{gen_block}
  endgenerate

  // Split packed weights
  genvar i;
  generate
    for (i = 0; i < NUM_NEURONS; i++) begin : gen_split_weights
      assign weights_rom_data_s[i] = weights_rom_data_raw_s[i * Q_WIDTH +: Q_WIDTH];
    end
  endgenerate

  // ------------------------------------------------------------
  // Register ROM outputs and data_i
  // ------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin : rom_out_regs
    if (!rst_n_i) begin
      for (int k = 0; k < NUM_NEURONS; k++) begin
        weights_rom_data_q[k] <= '0;
        data_i_q[k]           <= '0;
      end
      bias_rom_data_q <= '0;
    end 
    else begin
      for (int k = 0; k < NUM_NEURONS; k++) begin
        weights_rom_data_q[k] <= weights_rom_data_s[k];
        data_i_q[k]           <= data_i[k];
      end
      bias_rom_data_q <= bias_rom_data_s;
    end
  end : rom_out_regs

  // ------------------------------------------------------------
  // FC Compute Block Instance
  // ------------------------------------------------------------
  fc_out #(
    .NUM_NEURONS  (NUM_NEURONS),
    .LAYER_SCALE  (LAYER_SCALE),
    .BIAS_SCALE   (BIAS_SCALE)
  ) u_fc_out (
    .clk_i     (clk_i),
    .rst_n_i     (rst_n_i),
    .valid_i   (valid_pipeline_q2),
    .data_i    (data_i_q),
    .weights_i (weights_rom_data_q),
    .bias_i    (bias_rom_data_q),
    .data_o    (data_o),
    .valid_o   (valid_out_engine_s)
  );

endmodule : fc_out_layer
`end_keywords"""
    desc = "Fully-connected output layer with sequential ROM access. \n                Reads weights and biases from ROM and feeds the fc_out compute block."
    (out_dir / "fc_out_layer.sv").write_text(_pyramidtech_wrap(body, "fc_out_layer.sv", desc, author="AA"), encoding="utf-8")




# =============================================================================
# ConvLayerInfo dataclass
#
# Carries everything the RTL generator needs about one Conv layer in the model:
# weights (4D float32: out_ch, in_ch, kH, kW), bias (out_ch,), kernel/stride/pad,
# spatial in/out HxW (traced through the ONNX graph), Fin/Fout exponents, optional
# activation that follows, and the computed RTL descriptor (set after
# build_rtl_conv_quant_descriptors runs).
# =============================================================================

@dataclass
class ConvLayerInfo:
    """One Conv layer's full description for RTL generation.

    ``op_kind``:
      - "depthwise"  : the engine instantiates ``depthwise_conv_engine`` (line buffer +
        kH*kW window). For multi-channel inputs (in_ch > 1) the engine packs all
        ``in_ch`` channels into the kernel-window dimension (channel-fastest).
      - "pointwise"  : 1x1 conv → instantiates ``pointwise_conv_engine`` (in_ch parallel
        multiplies, sum across in_ch, one output position per cycle).

    ``in_h`` / ``in_w`` / ``out_h`` / ``out_w`` are spatial dims traced through the
    ONNX graph; the top module uses ``out_h`` of the last conv (before AvgPool) as the
    AvgPool ``FRAME_ROWS`` parameter and the post-flatten size as the first FC's INPUT_SIZE.
    """

    name: str                                 # e.g. "conv2d_proj_conv"
    op_kind: str                              # "depthwise" or "pointwise"
    in_channels: int
    out_channels: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_h: int = 0
    pad_w: int = 0
    weight: Optional[np.ndarray] = None       # (out_ch, in_ch, kH, kW) float
    bias: Optional[np.ndarray] = None         # (out_ch,) float
    quant_params: Optional[Dict[str, Any]] = None
    qdq_fin_exp: Optional[int] = None
    qdq_fout_exp: Optional[int] = None
    qdq_fin_scale: Optional[float] = None
    qdq_fout_scale: Optional[float] = None
    activation: Optional[str] = None          # ONNX activation that follows (Relu, etc.)
    rtl_quant: Optional[RtlLayerQuantDescriptor] = None
    in_h: int = 1
    in_w: int = 1
    out_h: int = 1
    out_w: int = 1
    onnx_node_name: str = ""
    fc_output_name: Optional[str] = None      # ONNX output tensor name (for QDQ chain trace)


# =============================================================================
# Conv weight / bias quantization
#
# Conv weights are 4D ``(out_ch, in_ch, kH, kW)`` while ``quantize_weight_for_rtl``
# treats them as a single tensor for choosing Fw — i.e. one Fw exponent per layer
# applied uniformly across all weights.  This is the same power-of-two convention
# used for FC weights.
# =============================================================================

def quantize_conv_weight_for_rtl(
    W4d: np.ndarray,
    *,
    lo: int = -128,
    hi: int = 127,
) -> Tuple[int, np.ndarray, float, float]:
    """Quantize 4D conv weights to int8 with a single power-of-two exponent.

    Reshapes (out_ch, in_ch, kH, kW) → flat → calls ``quantize_weight_for_rtl`` →
    reshapes back. Returns (fw_frac, W_int_4d, sat_lo_pct, sat_hi_pct).
    """
    W = np.asarray(W4d, dtype=np.float64)
    flat = W.reshape(-1)
    fw, wq_flat, sl, sh = quantize_weight_for_rtl(flat, lo=lo, hi=hi)
    return fw, wq_flat.reshape(W.shape), sl, sh


def build_rtl_conv_quant_descriptors(
    conv_layers: List[ConvLayerInfo],
    *,
    bit_width: int = 8,
    log_summary: bool = True,
) -> List[RtlLayerQuantDescriptor]:
    """Build per-conv-layer RTL quant descriptors using the same Fin/Fw/Fb/Fout
    methodology as ``build_rtl_layer_quant_descriptors``.

    For Conv layers we don't have an ONNX-pair quant_weight tensor (Conv weights
    pass through DequantizeLinear in static QDQ but we use the float weight as the
    truth source, identical to FC). ``fb_pair`` is therefore always None and the
    descriptor's ``fb_frac`` falls back to ``fb_rtl``.
    """
    if bit_width != 8:
        raise NotImplementedError("build_rtl_conv_quant_descriptors currently supports bit_width=8 only")
    lo, hi = -128, 127
    out: List[RtlLayerQuantDescriptor] = []
    for i, layer in enumerate(conv_layers):
        if layer.qdq_fin_exp is None or layer.qdq_fout_exp is None:
            raise RuntimeError(
                f"{layer.name}: missing Fin/Fout exponents from quantized ONNX (qdq_fin_exp / qdq_fout_exp). "
                "Export with static QDQ (multiclass_quantize.py) so activation scales appear as graph initializers."
            )
        fin = abs(int(layer.qdq_fin_exp))
        fout = abs(int(layer.qdq_fout_exp))

        if layer.weight is None:
            raise RuntimeError(f"{layer.name}: missing weight array for RTL quant")
        Wf = np.asarray(layer.weight, dtype=np.float64)
        if Wf.ndim != 4:
            raise RuntimeError(f"{layer.name}: conv weight must be 4D (out_ch, in_ch, kH, kW); got {Wf.shape}")
        Bf = (
            np.asarray(layer.bias, dtype=np.float64).ravel()
            if layer.bias is not None
            else np.zeros((layer.out_channels,), dtype=np.float64)
        )

        fw_frac, W_int, wsl, wsh = quantize_conv_weight_for_rtl(Wf, lo=lo, hi=hi)
        f_mac = fin + fw_frac
        fb_rtl, B_int, b_clip = quantize_bias_for_rtl(Bf, f_mac, lo=lo, hi=hi, layer_name=layer.name)
        bsl = 100.0 * float(np.sum(B_int <= lo)) / max(B_int.size, 1)
        bsh = 100.0 * float(np.sum(B_int >= hi)) / max(B_int.size, 1)

        desc = RtlLayerQuantDescriptor(
            layer_name=layer.name,
            layer_index=i,
            W_float=Wf.astype(np.float64),
            B_float=Bf.astype(np.float64),
            onnx_weight_scale=None,
            onnx_weight_zp=None,
            weight_init_name=None,
            fw_frac=fw_frac,
            W_int=W_int,
            weight_sat_lo_pct=wsl,
            weight_sat_hi_pct=wsh,
            fin_qdq=fin,
            fb_rtl=fb_rtl,
            B_int=B_int,
            bias_sat_lo_pct=bsl,
            bias_sat_hi_pct=bsh,
            fout_qdq=fout,
            bias_clipped_for_int8=b_clip,
            fb_pair=None,
        )
        layer.rtl_quant = desc
        out.append(desc)
        if wsl > 1.0 or wsh > 1.0 or bsl > 1.0 or bsh > 1.0:
            LOGGER.warning(
                "rtl_quant conv %s: saturation > 1%% after power-of-two quant — inspect float range.",
                layer.name,
            )

    if log_summary:
        LOGGER.debug(
            "rtl_quant conv: built %d conv descriptor(s) (power-of-two ROM).",
            len(out),
        )
    return out


def compute_conv_layer_scale(layer: ConvLayerInfo) -> int:
    """Conv ``LAYER_SCALE`` from the layer's RTL descriptor (signed shift).

    Mirrors the FC ``compute_layer_scales_from_rtl_descriptors`` rule:
      F_mac = Fin + Fw, then layer_shift = F_mac - Fout (right-shift if positive).
    Engines apply this as ``shifted = pre_shift >>> LAYER_SCALE`` for positive values;
    we always emit positive shifts here because conv outputs typically have lower
    fractional precision than the MAC (Fin+Fw > Fout).
    """
    rq = layer.rtl_quant
    if rq is None:
        raise RuntimeError(f"{layer.name}: missing rtl_quant for LAYER_SCALE computation")
    lsc, ldir = _qdq_pair_layer_alignment(int(rq.fin_qdq), int(rq.fw_frac), int(rq.fout_qdq))
    return _rtl_signed_layer_scale(lsc, ldir)


def compute_conv_bias_scale(layer: ConvLayerInfo) -> int:
    """Conv ``BIAS_SCALE`` (unused by the existing engines — bias is stored in the
    accumulator domain via ``BIAS_WIDTH=32``-bit ROM and added directly).

    Returned for parity with FC; engines may ignore it. Computed as F_mac - Fb.
    """
    rq = layer.rtl_quant
    if rq is None:
        raise RuntimeError(f"{layer.name}: missing rtl_quant for BIAS_SCALE computation")
    bsc, bdir = _qdq_pair_bias_alignment(int(rq.fin_qdq), int(rq.fw_frac), int(rq.fb_frac))
    return _rtl_signed_bias_scale(bsc, bdir)


# =============================================================================
# Conv .mem packing
#
# Existing RTL engines decode bits as:
#   depthwise: weights_rom_row[DATA_WIDTH*(f*WINDOW_SIZE + k + 1)-1 -: DATA_WIDTH]
#       where WINDOW_SIZE = K_H * K_W * K_C (channel-fastest within each spatial pos)
#   pointwise: weights_rom_row[DATA_WIDTH*(f*INPUTS_PER_CYCLE + i + 1)-1 -: DATA_WIDTH]
#       where INPUTS_PER_CYCLE = in_channels
#   conv bias:  bias_rom_row[BIAS_WIDTH*(f+1)-1 -: BIAS_WIDTH]
#       where BIAS_WIDTH = 32 (acc domain), per-filter sign-extended int32
#
# All three pack into a single ROM row (depth=1). The rightmost bits hold filter 0.
# =============================================================================

def _pack_lsb_first(values: List[int], elem_bits: int) -> int:
    """Pack ``values`` (LSB-first) as a single integer of len*elem_bits bits.

    Values are first masked to ``elem_bits`` (two's complement for signed ints),
    then OR'd at offset i*elem_bits.  This matches how SystemVerilog's bit-slice
    indexing reads packed words (slot 0 lives in the lowest bits).
    """
    mask = (1 << elem_bits) - 1
    packed = 0
    for i, v in enumerate(values):
        packed |= (int(v) & mask) << (i * elem_bits)
    return packed


def generate_dw_conv_weight_mem(
    W_int_4d: np.ndarray,
    out_path: Path,
    bit_width: int = 8,
) -> Tuple[int, int, int, int]:
    """Write depthwise/proj_conv weight .mem (single hex line).

    Layout: NUM_FILTERS=out_ch outer; within each filter, iterate kernel positions
    in (kH, kW, in_ch) order (channel-fastest). Total bits = out_ch * (kH * kW * in_ch) * bit_width.

    Returns (NUM_FILTERS, K_H, K_W, K_C) for the ROM .sv generator.
    """
    W = np.asarray(W_int_4d, dtype=np.int32)
    if W.ndim != 4:
        raise ValueError(f"DW conv weight must be 4D, got {W.shape}")
    out_ch, in_ch, kH, kW = W.shape
    window_size = kH * kW * in_ch
    out_path.parent.mkdir(parents=True, exist_ok=True)
    values: List[int] = []
    for f in range(out_ch):
        for i in range(kH):
            for j in range(kW):
                for c in range(in_ch):
                    values.append(int(W[f, c, i, j]))
    packed = _pack_lsb_first(values, bit_width)
    total_bits = out_ch * window_size * bit_width
    hex_width = (total_bits + 3) // 4
    out_path.write_text(f"{packed:0{hex_width}X}\n", encoding="utf-8")
    return out_ch, kH, kW, in_ch


def generate_pw_conv_weight_mem(
    W_int_4d: np.ndarray,
    out_path: Path,
    bit_width: int = 8,
) -> Tuple[int, int]:
    """Write pointwise (1x1) conv weight .mem (single hex line).

    Layout: NUM_FILTERS=out_ch outer; within each filter, iterate over INPUTS_PER_CYCLE=in_ch.
    Total bits = out_ch * in_ch * bit_width. Requires kH==kW==1.

    Returns (NUM_FILTERS, INPUTS_PER_CYCLE) for the ROM .sv generator.
    """
    W = np.asarray(W_int_4d, dtype=np.int32)
    if W.ndim != 4:
        raise ValueError(f"PW conv weight must be 4D, got {W.shape}")
    out_ch, in_ch, kH, kW = W.shape
    if kH != 1 or kW != 1:
        raise ValueError(f"PW (pointwise) weight must have kH=kW=1, got ({kH},{kW})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    values: List[int] = []
    for f in range(out_ch):
        for c in range(in_ch):
            values.append(int(W[f, c, 0, 0]))
    packed = _pack_lsb_first(values, bit_width)
    total_bits = out_ch * in_ch * bit_width
    hex_width = (total_bits + 3) // 4
    out_path.write_text(f"{packed:0{hex_width}X}\n", encoding="utf-8")
    return out_ch, in_ch


def generate_conv_bias_mem(
    B_int: np.ndarray,
    out_path: Path,
    *,
    bias_width: int = 32,
) -> int:
    """Write conv bias .mem in the format expected by ``depthwise_conv_engine`` /
    ``pointwise_conv_engine``: single hex line of ``out_ch * bias_width`` bits with
    each bias sign-extended to int32 and packed filter 0 in LSBs.

    Returns the number of filters written (= len(B_int)).
    """
    B = np.asarray(B_int, dtype=np.int64).ravel()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask = (1 << bias_width) - 1
    values: List[int] = []
    for v in B:
        vi = int(v)
        # sign-extend to int32
        if vi < 0:
            vi = vi & mask
        else:
            vi = vi & mask
        values.append(vi)
    packed = _pack_lsb_first(values, bias_width)
    total_bits = B.size * bias_width
    hex_width = (total_bits + 3) // 4
    out_path.write_text(f"{packed:0{hex_width}X}\n", encoding="utf-8")
    return int(B.size)


# =============================================================================
# Embedded SystemVerilog templates (conv stack)
#
# These are static SV bodies wrapped by ``_pyramidtech_wrap`` into the project's
# header style.  The depthwise / pointwise engines are emitted with a generated
# ``generate-if`` cascade (one branch per detected conv layer) so the same
# engine module serves every layer in the model — the top instantiates with a
# unique LAYER_INDEX per conv to pick the right ROM.
# =============================================================================

MULTICLASS_LINE_BUFFERS_SV = '''// ----------------------------------------------------------------------------
// line_buffers — sliding window buffer for streaming convolution
// ----------------------------------------------------------------------------
// Stores K_H rows, builds K_W-wide rows from streaming input. Supports
// multi-input per cycle via INPUTS_PER_CYCLE; each cycle places that many
// pixels into the current row, advancing the column counter and asserting
// row_complete when K_W pixels have been accumulated.
//
// valid_out pulses for one cycle when the K_H x K_W window is freshly valid.
// ----------------------------------------------------------------------------
module line_buffers #(
    parameter DATA_WIDTH       = 8,
    parameter K_W              = 3,
    parameter K_H              = 3,
    parameter INPUTS_PER_CYCLE = 1
)(
    input  logic                                          clk,
    input  logic                                          rst_n,
    input  logic                                          valid_in,
    input  logic signed [INPUTS_PER_CYCLE*DATA_WIDTH-1:0] data_in,

    output logic signed [DATA_WIDTH-1:0] data_out [0:K_H-1][0:K_W-1],
    output logic                                          valid_out
);

    timeunit 1ns;
    timeprecision 1ps;

    logic signed [DATA_WIDTH-1:0] row_buf     [0:K_H-1][0:K_W-1];
    logic signed [DATA_WIDTH-1:0] current_row [0:K_W-1];

    logic [$clog2(K_W+1)-1:0] col_counter;
    logic [$clog2(K_H+1)-1:0] valid_row_count;
    logic                     row_complete;

    logic [DATA_WIDTH-1:0] input_vals [0:INPUTS_PER_CYCLE-1];

    logic [$clog2(K_W+1)-1:0]              next_col;
    logic [$clog2(INPUTS_PER_CYCLE+1)-1:0] pix_idx;
    logic [$clog2(K_W+1)-1:0]              space_left;
    logic [$clog2(K_W+1)-1:0]              to_place;

    genvar g;
    generate
        for (g = 0; g < INPUTS_PER_CYCLE; g = g + 1)
            assign input_vals[g] = data_in[DATA_WIDTH*(g+1)-1 -: DATA_WIDTH];
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_counter  <= '0;
            row_complete <= 1'b0;
            for (int c = 0; c < K_W; c++) current_row[c] <= '0;
        end
        else begin
            row_complete <= 1'b0;
            if (valid_in) begin
                next_col = col_counter;
                pix_idx  = 0;
                while (pix_idx < INPUTS_PER_CYCLE) begin
                    space_left = K_W - next_col;
                    to_place   = (INPUTS_PER_CYCLE - pix_idx <= space_left) ?
                                 (INPUTS_PER_CYCLE - pix_idx) : space_left;
                    for (int j = 0; j < to_place; j++) begin
                        current_row[next_col] <= input_vals[pix_idx];
                        next_col = next_col + 1;
                        pix_idx  = pix_idx + 1;
                    end
                    if (next_col >= K_W) begin
                        row_complete <= 1'b1;
                        col_counter  <= 0;
                        next_col     = 0;
                    end
                end
                col_counter <= next_col;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_row_count <= '0;
            valid_out       <= 1'b0;
            for (int r = 0; r < K_H; r++)
                for (int c = 0; c < K_W; c++)
                    row_buf[r][c] <= '0;
        end
        else begin
            valid_out <= 1'b0;
            if (row_complete) begin
                for (int r = 0; r < K_H-1; r++)
                    for (int c = 0; c < K_W; c++)
                        row_buf[r][c] <= row_buf[r+1][c];
                for (int c = 0; c < K_W; c++)
                    row_buf[K_H-1][c] <= current_row[c];
                if (valid_row_count < K_H)
                    valid_row_count <= valid_row_count + 1'b1;
                if (valid_row_count >= K_H-1)
                    valid_out <= 1'b1;
            end
        end
    end

    generate
        for (g = 0; g < K_H; g = g + 1)
            for (genvar c = 0; c < K_W; c = c + 1)
                assign data_out[g][c] = row_buf[g][c];
    endgenerate

endmodule
'''


MULTICLASS_AVG_POOL_SV_TEMPLATE = '''// ----------------------------------------------------------------------------
// avg_pool_kx1 — average pooling along the row dimension only (W=1)
// ----------------------------------------------------------------------------
// Implements AvgPool2d(kernel_size=(KERNEL,1), stride=(STRIDE,STRIDE)) for a
// streaming H x 1 x CHANNELS frame. Output rows = floor((FRAME_ROWS-KERNEL)/STRIDE)+1.
//
// FRAME_ROWS is parameterized at instantiation time so the same module serves
// any input height. The row counter wraps at FRAME_ROWS-1 to detect frame end
// without an external "last" pin. STRIDE is assumed power-of-two so cnt naturally
// cycles through 0..STRIDE-1 by bit-select on row_cnt.
// ----------------------------------------------------------------------------
module avg_pool_kx1 #(
    parameter int DATA_W     = 8,
    parameter int CHANNELS   = 8,
    parameter int FRAME_ROWS = 714,
    parameter int KERNEL     = 2,
    parameter int STRIDE     = 4
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [DATA_W-1:0] data_in  [CHANNELS],
    input  logic              valid_in,

    output logic [DATA_W-1:0] data_out [CHANNELS],
    output logic              valid_out
);

    timeunit 1ns;
    timeprecision 1ps;

    localparam int ROW_BITS = $clog2(FRAME_ROWS);
    localparam int CNT_BITS = $clog2(STRIDE);
    localparam int EMIT_AT  = KERNEL - 1;

    logic [ROW_BITS-1:0] row_cnt;
    logic                last_row;
    logic [CNT_BITS-1:0] cnt;
    logic [DATA_W-1:0]   acc [CHANNELS];
    logic [DATA_W:0]     sum [CHANNELS];
    logic [DATA_W-1:0]   avg [CHANNELS];

    assign last_row = (row_cnt == ROW_BITS'(FRAME_ROWS - 1));
    assign cnt      = row_cnt[CNT_BITS-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)        row_cnt <= '0;
        else if (valid_in) row_cnt <= last_row ? '0 : row_cnt + 1'b1;
    end

    always_ff @(posedge clk) begin
        if (valid_in && cnt == '0) acc <= data_in;
    end

    always_comb
        for (int c = 0; c < CHANNELS; c++) begin
            sum[c] = {1'b0, acc[c]} + {1'b0, data_in[c]};
            avg[c] = sum[c][DATA_W:1];
        end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
        end
        else begin
            valid_out <= valid_in && (cnt == CNT_BITS'(EMIT_AT));
            if (valid_in && cnt == CNT_BITS'(EMIT_AT))
                data_out <= avg;
        end
    end

endmodule
'''


MULTICLASS_FLATTEN_UNIT_SV_TEMPLATE = '''// ----------------------------------------------------------------------------
// flatten_unit — Transpose [0,1,3,2] + Reshape to flat vector
// ----------------------------------------------------------------------------
// Buffers POOL_ROWS rows of CHANNELS-wide data, then streams them out in
// channel-major / row-minor order so the FC chain sees [c0_r0, c0_r1, ...,
// c0_r(POOL_ROWS-1), c1_r0, ..., cN-1_r(POOL_ROWS-1)] — a flat vector of
// length CHANNELS * POOL_ROWS = FLATTEN_SIZE.
//
// One small RAM per channel keeps the design simple and parallel-friendly.
// Write phase: POOL_ROWS cycles, valid_in=1, write each cycle into row index r.
// Read phase: FLATTEN_SIZE cycles, output one byte/cycle in channel-major order.
//
// Synchronization: a one-shot start_read pulse fires when the last row is written;
// the read counter then sweeps the full flat output before going idle again.
// ----------------------------------------------------------------------------
module flatten_unit #(
    parameter int DATA_W      = 8,
    parameter int CHANNELS    = 8,
    parameter int POOL_ROWS   = 179
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [DATA_W-1:0] data_in  [CHANNELS],
    input  logic              valid_in,

    output logic [DATA_W-1:0] data_out,
    output logic              valid_out,
    output logic              tlast_o
);

    timeunit 1ns;
    timeprecision 1ps;

    localparam int FLATTEN_SIZE = CHANNELS * POOL_ROWS;
    localparam int ROW_BITS     = (POOL_ROWS > 1) ? $clog2(POOL_ROWS) : 1;
    localparam int CH_BITS      = (CHANNELS  > 1) ? $clog2(CHANNELS)  : 1;
    localparam int FLAT_BITS    = $clog2(FLATTEN_SIZE + 1);

    // One BRAM per channel (depth = POOL_ROWS, width = DATA_W)
    logic [DATA_W-1:0] bank [0:CHANNELS-1][0:POOL_ROWS-1];

    logic [ROW_BITS-1:0] write_row_q;
    logic                write_done_q;

    logic [FLAT_BITS-1:0] read_idx_q;
    logic                 read_active_q;

    // -- Write side: one row per valid_in pulse, store all CHANNELS values --
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_row_q  <= '0;
            write_done_q <= 1'b0;
        end
        else begin
            write_done_q <= 1'b0;
            if (valid_in) begin
                for (int c = 0; c < CHANNELS; c++)
                    bank[c][write_row_q] <= data_in[c];
                if (write_row_q == ROW_BITS'(POOL_ROWS-1)) begin
                    write_row_q  <= '0;
                    write_done_q <= 1'b1;  // one-shot pulse, triggers read phase
                end
                else begin
                    write_row_q <= write_row_q + 1'b1;
                end
            end
        end
    end

    // -- Read side: sweep CHANNELS x POOL_ROWS in channel-major order --
    logic [CH_BITS-1:0]  read_ch;
    logic [ROW_BITS-1:0] read_row;
    assign read_ch  = read_idx_q[FLAT_BITS-1:ROW_BITS];   // upper bits = channel
    assign read_row = read_idx_q[ROW_BITS-1:0];            // lower bits = row

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_active_q <= 1'b0;
            read_idx_q    <= '0;
            valid_out     <= 1'b0;
            data_out      <= '0;
            tlast_o       <= 1'b0;
        end
        else begin
            valid_out <= 1'b0;
            tlast_o   <= 1'b0;
            if (write_done_q) begin
                read_active_q <= 1'b1;
                read_idx_q    <= '0;
            end
            if (read_active_q) begin
                data_out  <= bank[read_ch][read_row];
                valid_out <= 1'b1;
                if (read_idx_q == FLAT_BITS'(FLATTEN_SIZE - 1)) begin
                    read_active_q <= 1'b0;
                    tlast_o       <= 1'b1;
                end
                else begin
                    read_idx_q <= read_idx_q + 1'b1;
                end
            end
        end
    end

endmodule
'''


# Argmax: when the ONNX final op is ArgMax (or when user requests argmax mode),
# pick the index of the largest logit among NUM_CLASSES streamed values.
MULTICLASS_ARGMAX_SV = '''// ----------------------------------------------------------------------------
// argmax_layer — running max + index detector over a stream of NUM_CLASSES logits
// ----------------------------------------------------------------------------
// Consumes one logit per valid_i cycle for NUM_CLASSES cycles, tracks the largest
// value and its position, and asserts valid_o for one cycle when the last logit
// arrives, presenting the predicted class index on data_o.
//
// Index width = $clog2(NUM_CLASSES). Suitable for AXI4-Stream tdata padding to
// the model's prediction word width.
// ----------------------------------------------------------------------------
module argmax_layer
  import quant_pkg::*;
#(
  parameter int NUM_CLASSES = 10
)(
  input  logic    clk_i,
  input  logic    rst_n_i,

  input  logic    valid_i,
  input  q_data_t data_i,

  output logic [$clog2(NUM_CLASSES)-1:0] data_o,
  output logic    valid_o
);

  timeunit 1ns;
  timeprecision 1ps;

  localparam int IDX_W = $clog2(NUM_CLASSES);

  q_data_t                   max_q;
  logic    [IDX_W-1:0]       max_idx_q;
  logic    [IDX_W-1:0]       count_q;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      max_q     <= Q_MIN;
      max_idx_q <= '0;
      count_q   <= '0;
      data_o    <= '0;
      valid_o   <= 1'b0;
    end
    else begin
      valid_o <= 1'b0;
      if (valid_i) begin
        // First sample of a frame: reset max.
        if (count_q == '0) begin
          max_q     <= data_i;
          max_idx_q <= '0;
        end
        else if ($signed(data_i) > $signed(max_q)) begin
          max_q     <= data_i;
          max_idx_q <= count_q;
        end

        if (count_q == IDX_W'(NUM_CLASSES - 1)) begin
          count_q <= '0;
          // Emit the index of the winning class. Use updated value if last logit wins.
          if ($signed(data_i) > $signed(max_q))
            data_o <= count_q;
          else
            data_o <= max_idx_q;
          valid_o <= 1'b1;
        end
        else begin
          count_q <= count_q + 1'b1;
        end
      end
    end
  end

endmodule : argmax_layer
'''


# Softmax (when ONNX has Softmax op): in INT8 hardware a true softmax requires
# exp() + division. For the typical multiclass use case where the user only
# needs the predicted class, the argmax of softmax outputs equals the argmax of
# logits — so we stream the raw logits unchanged and let downstream software
# apply softmax if probability values are needed. Each logit becomes one AXI
# beat; valid_o passes through with no change.
MULTICLASS_SOFTMAX_SV = '''// ----------------------------------------------------------------------------
// softmax_layer — INT8 streaming passthrough (raw logits)
// ----------------------------------------------------------------------------
// In an INT8 fixed-point pipeline a true softmax requires an exp() LUT and a
// division stage; both add significant area and are unnecessary when the
// downstream consumer only needs the predicted class (argmax of softmax ==
// argmax of logits). This module therefore passes the streamed logits through
// untouched, with one logit per valid pulse for NUM_CLASSES cycles per frame.
//
// If the integrating system needs probability values, apply softmax in software
// on the streamed logits. To enable hardware softmax later, replace this body
// with an exp-LUT + sum + divider pipeline.
// ----------------------------------------------------------------------------
module softmax_layer
  import quant_pkg::*;
#(
  parameter int NUM_CLASSES = 10
)(
  input  logic    clk_i,
  input  logic    rst_n_i,

  input  logic    valid_i,
  input  q_data_t data_i,

  output q_data_t data_o,
  output logic    valid_o
);

  timeunit 1ns;
  timeprecision 1ps;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      data_o  <= '0;
      valid_o <= 1'b0;
    end
    else begin
      valid_o <= valid_i;
      if (valid_i) data_o <= data_i;
    end
  end

endmodule : softmax_layer
'''


# =============================================================================
# Conv ROM .sv generator (per-layer single-row ROMs)
#
# Each conv layer has its own pair of ROM modules: one for the packed weights
# (NUM_FILTERS * (K_H*K_W*K_C or in_ch) * Q_WIDTH bits) and one for biases
# (NUM_FILTERS * BIAS_WIDTH bits, BIAS_WIDTH=32). All conv ROMs are depth=1
# (single row) which the engine reads once at start.  The .mem file path uses
# `__FILE__` relative resolution so the simulator finds it next to the ROM .sv.
# =============================================================================

def _generate_conv_rom_sv(
    out_dir: Path,
    rom_module_name: str,
    mem_basename: str,
    packed_width: int,
    *,
    description: str = "",
) -> Path:
    """Emit one conv ROM .sv module that loads ``mem_files/<mem_basename>`` via $readmemh.

    Conv ROMs always have DEPTH=1 (single packed row). WIDTH = packed_width bits;
    for weights this is NUM_FILTERS * WINDOW_SIZE * Q_WIDTH; for biases it's
    NUM_FILTERS * BIAS_WIDTH (= 32 in our convention).
    """
    body = f"""module {rom_module_name} #(
  parameter int DEPTH  = 1,
  parameter int WIDTH  = {packed_width},
  parameter int ADDR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
  input  logic              clk_i,
  input  logic [ADDR_W-1:0] addr_i,
  output logic [WIDTH-1:0]  data_o
);

  timeunit 1ns;
  timeprecision 1ps;

  (* rom_style = "block" *)
  logic [WIDTH-1:0] mem [0:DEPTH-1];

  initial begin
    $readmemh({{`__FILE__, "/../mem_files/{mem_basename}"}}, mem);
  end

  always_ff @(posedge clk_i) begin : read_port
    data_o <= mem[addr_i];
  end : read_port

endmodule : {rom_module_name}
"""
    desc = description or f"ROM for conv layer (loads {mem_basename})"
    out_path = out_dir / f"{rom_module_name}.sv"
    out_path.write_text(_pyramidtech_wrap(body, f"{rom_module_name}.sv", desc), encoding="utf-8")
    return out_path


# =============================================================================
# Depthwise / pointwise conv engine generators
#
# Each engine has a generate-if cascade that selects per-layer ROM modules
# based on a LAYER_INDEX parameter (1, 2, 3, ... for the N detected DW/PW
# layers of this kind).  The branches are emitted at code-gen time so the
# engine module exactly matches the layers in the input ONNX model.
# =============================================================================

def _generate_dw_conv_engine_sv(
    out_dir: Path,
    dw_layer_names: List[str],
) -> Path:
    """Emit ``depthwise_conv_engine.sv`` with one ROM-selection branch per DW layer.

    Layer instantiation ``LAYER_INDEX = i+1`` selects ROMs ``<dw_layer_names[i]>_weights_rom``
    and ``<dw_layer_names[i]>_bias_rom``.  This parallels the existing engine layout
    where DW_CONV_INDEX=1..3 selected hardcoded conv2d_proj_conv / conv2d_1_proj_conv /
    conv2d_2_proj_conv ROMs.
    """
    if not dw_layer_names:
        raise ValueError("Need at least one DW layer to emit depthwise_conv_engine.sv")
    branches: List[str] = []
    for i, lname in enumerate(dw_layer_names):
        cond = "if" if i == 0 else "else if"
        branches.append(f"""    {cond} (LAYER_INDEX == {i+1}) begin : gen_{lname}
      {lname}_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
      ) the_{lname}_weights_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
      );

      {lname}_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
      ) the_{lname}_bias_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
      );
    end""")
    gen_block = "\n".join(branches)
    body = f"""module depthwise_conv_engine #(
    parameter int DATA_WIDTH       = 8,
    parameter int K_H              = 3,
    parameter int K_W              = 3,
    parameter int NUM_FILTERS      = 2,
    parameter int BIAS_WIDTH       = 32,
    parameter int LAYER_SCALE      = 7,
    parameter int LAYER_INDEX      = 1,
    parameter int INPUTS_PER_CYCLE = 1
)(
    input  logic                                          clk_i,
    input  logic                                          rst_n_i,
    input  logic                                          valid_in,
    input  logic signed [INPUTS_PER_CYCLE*DATA_WIDTH-1:0] data_in,
    output logic signed [DATA_WIDTH-1:0]                  conv_out [0:NUM_FILTERS-1],
    output logic                                          valid_out
);

    timeunit 1ns;
    timeprecision 1ps;

    // -------------------- Line buffer + window flatten --------------------
    logic signed [DATA_WIDTH-1:0] lb_out [0:K_H-1][0:K_W-1];
    logic                         lb_valid;

    line_buffers #(
        .DATA_WIDTH(DATA_WIDTH),
        .K_H(K_H),
        .K_W(K_W),
        .INPUTS_PER_CYCLE(INPUTS_PER_CYCLE)
    ) lb_inst (
        .clk(clk_i),
        .rst_n(rst_n_i),
        .valid_in(valid_in),
        .data_in(data_in),
        .data_out(lb_out),
        .valid_out(lb_valid)
    );

    localparam int WINDOW_SIZE      = K_H * K_W;
    localparam int WEIGHTS_ROW_WIDTH = NUM_FILTERS * WINDOW_SIZE * DATA_WIDTH;

    logic signed [DATA_WIDTH-1:0]              window_flat [0:WINDOW_SIZE-1];
    logic signed [WEIGHTS_ROW_WIDTH-1:0]       weights_rom_row;
    logic signed [BIAS_WIDTH*NUM_FILTERS-1:0]  bias_rom_row;

    genvar i, j;
    generate
        for (i = 0; i < K_H; i = i+1) begin : GEN_FLATTEN_ROW
            for (j = 0; j < K_W; j = j+1) begin : GEN_FLATTEN_COL
                assign window_flat[i*K_W + j] = lb_out[i][j];
            end
        end
    endgenerate

    // -------------------- Per-layer ROM selection --------------------
    generate
{gen_block}
    endgenerate

    // -------------------- Pipeline (multiply -> accumulate -> bias + scale) --------------------
    localparam int PIPELINE_DEPTH = 3;
    logic [PIPELINE_DEPTH-1:0] valid_pipe;
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            valid_pipe <= '0;
        else
            valid_pipe <= {{valid_pipe[PIPELINE_DEPTH-2:0], lb_valid}};
    end
    assign valid_out = valid_pipe[PIPELINE_DEPTH-1];

    logic signed [DATA_WIDTH*2-1:0] mult_pipe [0:NUM_FILTERS-1][0:WINDOW_SIZE-1];
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            for (int f = 0; f < NUM_FILTERS; f++)
                for (int k = 0; k < WINDOW_SIZE; k++)
                    mult_pipe[f][k] <= '0;
        end
        else if (lb_valid) begin
            for (int f = 0; f < NUM_FILTERS; f++)
                for (int k = 0; k < WINDOW_SIZE; k++)
                    mult_pipe[f][k] <= $signed(window_flat[k]) *
                        $signed(weights_rom_row[DATA_WIDTH*(f*WINDOW_SIZE + k + 1)-1 -: DATA_WIDTH]);
        end
    end

    logic signed [31:0] sum_pipe [0:NUM_FILTERS-1];
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            for (int f = 0; f < NUM_FILTERS; f++)
                sum_pipe[f] <= '0;
        else if (valid_pipe[0])
            for (int f = 0; f < NUM_FILTERS; f++) begin
                automatic logic signed [31:0] sum_tmp;
                sum_tmp = 0;
                for (int k = 0; k < WINDOW_SIZE; k++)
                    sum_tmp += mult_pipe[f][k];
                sum_pipe[f] <= sum_tmp;
            end
    end

    logic signed [31:0] conv_out_pipe [0:NUM_FILTERS-1];
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            for (int f = 0; f < NUM_FILTERS; f++)
                conv_out_pipe[f] <= '0;
        end
        else if (valid_pipe[1]) begin
            for (int f = 0; f < NUM_FILTERS; f++) begin
                automatic logic signed [31:0] pre_shift;
                automatic logic signed [31:0] shifted;
                pre_shift = sum_pipe[f] + $signed(bias_rom_row[BIAS_WIDTH*(f+1)-1 -: BIAS_WIDTH]);
                shifted   = pre_shift >>> LAYER_SCALE;
                if (shifted > 127)         conv_out_pipe[f] <= 127;
                else if (shifted < -128)   conv_out_pipe[f] <= -128;
                else                        conv_out_pipe[f] <= shifted;
            end
        end
    end

    generate
        for (i = 0; i < NUM_FILTERS; i = i+1)
            assign conv_out[i] = conv_out_pipe[i][DATA_WIDTH-1:0];
    endgenerate

endmodule
"""
    out_path = out_dir / "depthwise_conv_engine.sv"
    desc = "Depthwise / kH x kW projection convolution engine with line buffer, parameterized layer-ROM selection."
    out_path.write_text(_pyramidtech_wrap(body, "depthwise_conv_engine.sv", desc), encoding="utf-8")
    return out_path


def _generate_pw_conv_engine_sv(
    out_dir: Path,
    pw_layer_names: List[str],
) -> Path:
    """Emit ``pointwise_conv_engine.sv`` with one ROM-selection branch per PW layer.

    Layer instantiation ``LAYER_INDEX = i+1`` selects ROMs ``<pw_layer_names[i]>_weights_rom``
    and ``<pw_layer_names[i]>_bias_rom``.  Mirrors the existing engine's PW_CONV_INDEX=1..5 cascade.
    """
    if not pw_layer_names:
        raise ValueError("Need at least one PW layer to emit pointwise_conv_engine.sv")
    branches: List[str] = []
    for i, lname in enumerate(pw_layer_names):
        cond = "if" if i == 0 else "else if"
        branches.append(f"""    {cond} (LAYER_INDEX == {i+1}) begin : gen_{lname}
      {lname}_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
      ) the_{lname}_weights_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
      );

      {lname}_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
      ) the_{lname}_bias_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
      );
    end""")
    gen_block = "\n".join(branches)
    body = f"""module pointwise_conv_engine #(
    parameter int DATA_WIDTH       = 8,
    parameter int NUM_FILTERS      = 2,
    parameter int INPUTS_PER_CYCLE = 2,
    parameter int BIAS_WIDTH       = 32,
    parameter int LAYER_INDEX      = 1,
    parameter int LAYER_SCALE      = 5
)(
    input  logic                                  clk_i,
    input  logic                                  rst_n_i,
    input  logic                                  valid_in,
    input  logic signed [DATA_WIDTH-1:0]          data_in [0:INPUTS_PER_CYCLE-1],
    output logic signed [DATA_WIDTH-1:0]          conv_out [0:NUM_FILTERS-1],
    output logic                                  valid_out
);

    timeunit 1ns;
    timeprecision 1ps;

    logic signed [DATA_WIDTH-1:0] ch_in [0:INPUTS_PER_CYCLE-1];
    genvar k;
    generate
        for (k = 0; k < INPUTS_PER_CYCLE; k = k+1) begin : GEN_CH_IN
            assign ch_in[k] = data_in[k];
        end
    endgenerate

    localparam int WEIGHTS_ROW_WIDTH = NUM_FILTERS * INPUTS_PER_CYCLE * DATA_WIDTH;
    logic signed [WEIGHTS_ROW_WIDTH-1:0]       weights_rom_row;
    logic signed [BIAS_WIDTH*NUM_FILTERS-1:0]  bias_rom_row;

    generate
{gen_block}
    endgenerate

    localparam int PIPELINE_DEPTH = 3;
    logic [PIPELINE_DEPTH-1:0] valid_pipe;
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) valid_pipe <= '0;
        else          valid_pipe <= {{valid_pipe[PIPELINE_DEPTH-2:0], valid_in}};
    end
    assign valid_out = valid_pipe[PIPELINE_DEPTH-1];

    logic signed [DATA_WIDTH*2-1:0] mult_pipe [0:NUM_FILTERS-1][0:INPUTS_PER_CYCLE-1];
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            for (int f = 0; f < NUM_FILTERS; f++)
                for (int i = 0; i < INPUTS_PER_CYCLE; i++)
                    mult_pipe[f][i] <= '0;
        end
        else if (valid_in) begin
            for (int f = 0; f < NUM_FILTERS; f++)
                for (int i = 0; i < INPUTS_PER_CYCLE; i++)
                    mult_pipe[f][i] <= $signed(ch_in[i]) *
                        $signed(weights_rom_row[DATA_WIDTH*(f*INPUTS_PER_CYCLE + i + 1)-1 -: DATA_WIDTH]);
        end
    end

    logic signed [31:0] sum_pipe [0:NUM_FILTERS-1];
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            for (int f = 0; f < NUM_FILTERS; f++)
                sum_pipe[f] <= '0;
        else if (valid_pipe[0])
            for (int f = 0; f < NUM_FILTERS; f++) begin
                automatic logic signed [31:0] sum_tmp = 0;
                for (int i = 0; i < INPUTS_PER_CYCLE; i++)
                    sum_tmp += mult_pipe[f][i];
                sum_pipe[f] <= sum_tmp;
            end
    end

    logic signed [31:0] conv_out_pipe [0:NUM_FILTERS-1];
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            for (int f = 0; f < NUM_FILTERS; f++)
                conv_out_pipe[f] <= '0;
        else if (valid_pipe[1])
            for (int f = 0; f < NUM_FILTERS; f++) begin
                automatic logic signed [31:0] pre_shift;
                automatic logic signed [31:0] shifted;
                pre_shift = sum_pipe[f] + $signed(bias_rom_row[BIAS_WIDTH*(f+1)-1 -: BIAS_WIDTH]);
                shifted   = pre_shift >>> LAYER_SCALE;
                if (shifted > 127)        conv_out_pipe[f] <= 127;
                else if (shifted < -128)  conv_out_pipe[f] <= -128;
                else                       conv_out_pipe[f] <= shifted;
            end
    end

    genvar i;
    generate
        for (i = 0; i < NUM_FILTERS; i = i+1)
            assign conv_out[i] = conv_out_pipe[i][DATA_WIDTH-1:0];
    endgenerate

endmodule
"""
    out_path = out_dir / "pointwise_conv_engine.sv"
    desc = "1x1 pointwise convolution engine with parallel input channels and parameterized layer-ROM selection."
    out_path.write_text(_pyramidtech_wrap(body, "pointwise_conv_engine.sv", desc), encoding="utf-8")
    return out_path


def _generate_avg_pool_sv(out_dir: Path) -> Path:
    """Emit ``avg_pool_kx1.sv`` (parameterised — FRAME_ROWS / KERNEL / STRIDE / CHANNELS)."""
    out_path = out_dir / "avg_pool_kx1.sv"
    desc = "Streaming average pooling along the row dimension (kernel KxKERNEL=Kx1, stride STRIDExSTRIDE)."
    out_path.write_text(_pyramidtech_wrap(MULTICLASS_AVG_POOL_SV_TEMPLATE, "avg_pool_kx1.sv", desc), encoding="utf-8")
    return out_path


def _generate_flatten_unit_sv(out_dir: Path) -> Path:
    """Emit ``flatten_unit.sv`` (parameterised — CHANNELS, POOL_ROWS)."""
    out_path = out_dir / "flatten_unit.sv"
    desc = "Transpose [0,1,3,2] + Reshape buffer: rebuilds POOL_ROWS x CHANNELS into a flat channel-major stream."
    out_path.write_text(_pyramidtech_wrap(MULTICLASS_FLATTEN_UNIT_SV_TEMPLATE, "flatten_unit.sv", desc), encoding="utf-8")
    return out_path


def _generate_line_buffers_sv(out_dir: Path) -> Path:
    """Emit ``line_buffers.sv`` (already fully parameterised)."""
    out_path = out_dir / "line_buffers.sv"
    desc = "Sliding K_H x K_W window line buffer for streaming 2D convolution."
    out_path.write_text(_pyramidtech_wrap(MULTICLASS_LINE_BUFFERS_SV, "line_buffers.sv", desc), encoding="utf-8")
    return out_path


def _generate_softmax_sv(out_dir: Path) -> Path:
    out_path = out_dir / "softmax_layer.sv"
    desc = "Softmax classifier head: passes raw logits through unchanged (downstream applies softmax in software)."
    out_path.write_text(_pyramidtech_wrap(MULTICLASS_SOFTMAX_SV, "softmax_layer.sv", desc), encoding="utf-8")
    return out_path


def _generate_argmax_sv(out_dir: Path) -> Path:
    out_path = out_dir / "argmax_layer.sv"
    desc = "Argmax classifier head: streams the index of the largest logit over NUM_CLASSES samples."
    out_path.write_text(_pyramidtech_wrap(MULTICLASS_ARGMAX_SV, "argmax_layer.sv", desc), encoding="utf-8")
    return out_path


# =============================================================================
# Conv block grouping
#
# Conv layers in the ONNX graph are naturally grouped into "blocks" delimited
# by Relu nodes (the activation between blocks).  ``group_conv_blocks`` walks the
# extracted layer list and returns a list of block specifications: each block
# is a list of conv layers belonging to it, ending at a Relu (or end-of-conv).
# =============================================================================

def group_conv_blocks(conv_layers: List[ConvLayerInfo]) -> List[List[ConvLayerInfo]]:
    """Group consecutive conv layers into blocks separated by ReLU activations.

    A block ends when its last layer's ``activation`` is set (e.g. "Relu"). All
    conv layers in a block share the same input/output streaming clock — the
    block boundary is where the dataflow needs a vector ReLU before continuing.
    """
    blocks: List[List[ConvLayerInfo]] = []
    cur: List[ConvLayerInfo] = []
    for layer in conv_layers:
        cur.append(layer)
        if (layer.activation or "").strip().lower() == "relu":
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def assign_conv_layer_indices(
    conv_layers: List[ConvLayerInfo],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Assign 1-based LAYER_INDEX values to DW and PW conv layers in occurrence order.

    Returns (dw_indices, pw_indices): each maps layer.name → integer index used
    by ``depthwise_conv_engine`` / ``pointwise_conv_engine`` to select the layer's ROM.
    """
    dw_idx: Dict[str, int] = {}
    pw_idx: Dict[str, int] = {}
    for layer in conv_layers:
        if layer.op_kind == "depthwise":
            dw_idx[layer.name] = len(dw_idx) + 1
        elif layer.op_kind == "pointwise":
            pw_idx[layer.name] = len(pw_idx) + 1
        else:
            raise ValueError(f"{layer.name}: unsupported op_kind {layer.op_kind!r}")
    return dw_idx, pw_idx


# =============================================================================
# Multiclass top module builder
#
# Emits ``multiclass_NN.sv`` — the wired top-level IP. Sections of the body:
#   1. Parameter declarations (per-conv-layer and per-FC-block scales + spatial dims)
#   2. AXI slave / master ports
#   3. Per-stage signal declarations
#   4. Conv chain instantiations (DW + PW engines, vector ReLU between blocks)
#   5. Average pool instantiation
#   6. Flatten unit instantiation
#   7. FC chain instantiations (fc_in_layer + fc_out_layer + relu_layer per pair)
#   8. Classifier head: softmax_layer (passthrough) or argmax_layer (single index)
#   9. Output FIFO + AXI master logic with tlast generation
# =============================================================================

def _sv_int_lit(v: int) -> str:
    """Format a SystemVerilog 32-bit integer parameter literal (signed-aware)."""
    return f"32'd{v}" if v >= 0 else f"-32'sd{-v}"


def _build_multiclass_nn_sv_content(
    conv_layers: List[ConvLayerInfo],
    fc_linear_layers: List[LayerInfo],
    fc_params: Dict[str, Any],
    *,
    final_op_kind: str,        # "softmax" or "argmax"
    num_classes: int,
    pool_kernel: int,
    pool_stride: int,
    pool_frame_rows: int,
    pool_channels: int,
    pool_out_rows: int,
    flatten_size: int,
) -> str:
    """Build the multiclass_NN.sv body (module header + signals + instantiations).

    All conv layer parameters (LAYER_SCALE, K_H/K_W, channels) are emitted as
    top-level parameters so an integrator can override them; defaults come from
    each ``ConvLayerInfo`` plus ``layer.rtl_quant``.

    The AXI slave port is ``s_axis_tdata_i`` (q_data_t) consumed one int8 per cycle —
    the upstream feeder must serialize the input frame in (row, col) order matching
    the line buffer's expectation.
    """
    if not conv_layers:
        raise ValueError("multiclass_NN requires at least one conv layer")
    if not fc_linear_layers:
        raise ValueError("multiclass_NN requires at least one FC layer")
    if final_op_kind not in ("softmax", "argmax"):
        raise ValueError(f"final_op_kind must be 'softmax' or 'argmax', got {final_op_kind!r}")

    dw_idx, pw_idx = assign_conv_layer_indices(conv_layers)
    blocks = group_conv_blocks(conv_layers)
    n_fc = len(fc_linear_layers)
    num_fc_blocks = (n_fc + 1) // 2

    # -------------------- Parameter declarations --------------------
    p_lines: List[str] = []
    # Per-conv-layer parameters: NUM_FILTERS, INPUTS_PER_CYCLE (or kernel size for DW), LAYER_SCALE
    for layer in conv_layers:
        lname_upper = layer.name.upper()
        ls = compute_conv_layer_scale(layer)
        if layer.op_kind == "depthwise":
            p_lines.append(f"  parameter int {lname_upper}_K_H        = {_sv_int_lit(layer.kernel_h)},")
            p_lines.append(f"  parameter int {lname_upper}_K_W        = {_sv_int_lit(layer.kernel_w * layer.in_channels)},")
            p_lines.append(f"  parameter int {lname_upper}_NUM_FIL    = {_sv_int_lit(layer.out_channels)},")
            p_lines.append(f"  parameter int {lname_upper}_INPUTS_PC  = {_sv_int_lit(layer.in_channels)},")
        else:  # pointwise
            p_lines.append(f"  parameter int {lname_upper}_NUM_FIL    = {_sv_int_lit(layer.out_channels)},")
            p_lines.append(f"  parameter int {lname_upper}_INPUTS_PC  = {_sv_int_lit(layer.in_channels)},")
        p_lines.append(f"  parameter int {lname_upper}_LAYER_SCALE = {_sv_int_lit(ls)},")
    # AvgPool params
    p_lines.append(f"  parameter int POOL_KERNEL    = {_sv_int_lit(pool_kernel)},")
    p_lines.append(f"  parameter int POOL_STRIDE    = {_sv_int_lit(pool_stride)},")
    p_lines.append(f"  parameter int POOL_FRAME_ROWS= {_sv_int_lit(pool_frame_rows)},")
    p_lines.append(f"  parameter int POOL_CHANNELS  = {_sv_int_lit(pool_channels)},")
    p_lines.append(f"  parameter int POOL_OUT_ROWS  = {_sv_int_lit(pool_out_rows)},")
    p_lines.append(f"  parameter int FLATTEN_SIZE   = {_sv_int_lit(flatten_size)},")
    # FC block parameters
    for b in range(1, num_fc_blocks + 1):
        p = f"FC_{b}"
        p_lines.append(f"  parameter int {p}_NEURONS         = {_sv_int_lit(fc_params.get(f'{p}_NEURONS', 1))},")
        p_lines.append(f"  parameter int {p}_INPUT_SIZE      = {_sv_int_lit(fc_params.get(f'{p}_INPUT_SIZE', 1))},")
        p_lines.append(f"  parameter int {p}_ROM_DEPTH       = {_sv_int_lit(fc_params.get(f'{p}_ROM_DEPTH', 1))},")
        p_lines.append(f"  parameter int {p}_IN_LAYER_SCALE  = {_sv_int_lit(fc_params.get(f'{p}_IN_LAYER_SCALE', 0))},")
        p_lines.append(f"  parameter int {p}_IN_BIAS_SCALE   = {_sv_int_lit(fc_params.get(f'{p}_IN_BIAS_SCALE', 0))},")
        p_lines.append(f"  parameter int {p}_OUT_LAYER_SCALE = {_sv_int_lit(fc_params.get(f'{p}_OUT_LAYER_SCALE', 0))},")
        p_lines.append(f"  parameter int {p}_OUT_BIAS_SCALE  = {_sv_int_lit(fc_params.get(f'{p}_OUT_BIAS_SCALE', 0))},")
    p_lines.append(f"  parameter int NUM_CLASSES        = {_sv_int_lit(num_classes)},")
    p_lines.append(f"  parameter int FIFO_DEPTH         = 32'd1024")
    param_block = "\n".join(p_lines)

    # -------------------- Conv chain wiring --------------------
    sig_lines: List[str] = ["  // ----- Conv chain signals -----"]
    inst_lines: List[str] = ["  // ----- Conv chain instantiations -----"]

    # First conv consumes AXI slave input directly (q_data_t scalar)
    prev_data_kind: str = "scalar"   # scalar (1 q_data_t) or array (q_data_t[N])
    prev_signal: str = "s_axis_tdata_i"
    prev_valid: str = "s_axis_tvalid_i"
    prev_n_ch: int = 1               # input has 1 channel (NCHW with C=1)

    def _emit_dw(layer: ConvLayerInfo, prev_data: str, prev_v: str, prev_kind: str, prev_nc: int) -> Tuple[str, str, str, int]:
        idx = dw_idx[layer.name]
        lname = layer.name
        sname = lname  # signal prefix
        nf = layer.out_channels
        ipc = layer.in_channels
        sig_lines.append(f"  logic signed [7:0] {sname}_out_s [0:{nf}-1];")
        sig_lines.append(f"  logic              {sname}_valid_s;")
        # Input wiring: must match line_buffers' INPUTS_PER_CYCLE * DATA_WIDTH-bit packed bus
        if prev_kind == "scalar":
            # 1 channel — connect directly
            data_in_expr = prev_data
        else:
            # array → flatten into packed bus (LSB-first by index)
            sig_lines.append(f"  logic signed [{ipc*8}-1:0] {sname}_data_in_s;")
            packs = " , ".join(f"{prev_data}[{i}]" for i in reversed(range(ipc)))
            inst_lines.append(f"  assign {sname}_data_in_s = {{{packs}}};")
            data_in_expr = f"{sname}_data_in_s"
        inst_lines.append(f"""  depthwise_conv_engine #(
    .DATA_WIDTH      (8),
    .K_H             ({lname.upper()}_K_H),
    .K_W             ({lname.upper()}_K_W),
    .NUM_FILTERS     ({lname.upper()}_NUM_FIL),
    .BIAS_WIDTH      (32),
    .LAYER_SCALE     ({lname.upper()}_LAYER_SCALE),
    .LAYER_INDEX     ({idx}),
    .INPUTS_PER_CYCLE({lname.upper()}_INPUTS_PC)
  ) u_{lname} (
    .clk_i   (clk_i),
    .rst_n_i (rst_n_i),
    .valid_in({prev_v}),
    .data_in ({data_in_expr}),
    .conv_out({sname}_out_s),
    .valid_out({sname}_valid_s)
  );""")
        return f"{sname}_out_s", f"{sname}_valid_s", "array", nf

    def _emit_pw(layer: ConvLayerInfo, prev_data: str, prev_v: str, prev_kind: str, prev_nc: int) -> Tuple[str, str, str, int]:
        idx = pw_idx[layer.name]
        lname = layer.name
        sname = lname
        nf = layer.out_channels
        ipc = layer.in_channels
        sig_lines.append(f"  logic signed [7:0] {sname}_out_s [0:{nf}-1];")
        sig_lines.append(f"  logic              {sname}_valid_s;")
        # Input wiring: pointwise wants ``data_in [0:INPUTS_PER_CYCLE-1]``, an array
        if prev_kind == "scalar":
            sig_lines.append(f"  logic signed [7:0] {sname}_data_in_s [0:{ipc}-1];")
            for k in range(ipc):
                inst_lines.append(f"  assign {sname}_data_in_s[{k}] = {prev_data};")
            data_in_expr = f"{sname}_data_in_s"
        else:
            data_in_expr = prev_data
        inst_lines.append(f"""  pointwise_conv_engine #(
    .DATA_WIDTH      (8),
    .NUM_FILTERS     ({lname.upper()}_NUM_FIL),
    .INPUTS_PER_CYCLE({lname.upper()}_INPUTS_PC),
    .BIAS_WIDTH      (32),
    .LAYER_INDEX     ({idx}),
    .LAYER_SCALE     ({lname.upper()}_LAYER_SCALE)
  ) u_{lname} (
    .clk_i   (clk_i),
    .rst_n_i (rst_n_i),
    .valid_in({prev_v}),
    .data_in ({data_in_expr}),
    .conv_out({sname}_out_s),
    .valid_out({sname}_valid_s)
  );""")
        return f"{sname}_out_s", f"{sname}_valid_s", "array", nf

    # Walk conv layers, inserting a per-block ReLU vector when activation=='relu'
    for b_i, blk in enumerate(blocks):
        for layer in blk:
            if layer.op_kind == "depthwise":
                prev_signal, prev_valid, prev_data_kind, prev_n_ch = _emit_dw(
                    layer, prev_signal, prev_valid, prev_data_kind, prev_n_ch
                )
            else:
                prev_signal, prev_valid, prev_data_kind, prev_n_ch = _emit_pw(
                    layer, prev_signal, prev_valid, prev_data_kind, prev_n_ch
                )
        # Vector ReLU after the block (if the last layer's activation is 'relu')
        last = blk[-1]
        if (last.activation or "").strip().lower() == "relu":
            relu_sig = f"relu{b_i+1}_out_s"
            relu_valid = f"relu{b_i+1}_valid_s"
            sig_lines.append(f"  logic signed [7:0] {relu_sig} [0:{prev_n_ch}-1];")
            sig_lines.append(f"  logic              {relu_valid};")
            inst_lines.append(f"""  // Vector ReLU after block {b_i+1} ({prev_n_ch} channels)
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      for (int c = 0; c < {prev_n_ch}; c++) {relu_sig}[c] <= '0;
      {relu_valid} <= 1'b0;
    end
    else begin
      {relu_valid} <= {prev_valid};
      for (int c = 0; c < {prev_n_ch}; c++)
        {relu_sig}[c] <= ({prev_signal}[c] < 0) ? 8'sd0 : {prev_signal}[c];
    end
  end""")
            prev_signal = relu_sig
            prev_valid = relu_valid
            prev_data_kind = "array"

    # -------------------- AvgPool --------------------
    sig_lines.append("  // ----- AvgPool signals -----")
    sig_lines.append(f"  logic [7:0] pool_out_s [POOL_CHANNELS];")
    sig_lines.append(f"  logic       pool_valid_s;")
    inst_lines.append(f"""  // Average pooling along row dimension (POOL_KERNEL x 1, stride POOL_STRIDE)
  avg_pool_kx1 #(
    .DATA_W    (8),
    .CHANNELS  (POOL_CHANNELS),
    .FRAME_ROWS(POOL_FRAME_ROWS),
    .KERNEL    (POOL_KERNEL),
    .STRIDE    (POOL_STRIDE)
  ) u_avg_pool (
    .clk      (clk_i),
    .rst_n    (rst_n_i),
    .data_in  ({prev_signal}),
    .valid_in ({prev_valid}),
    .data_out (pool_out_s),
    .valid_out(pool_valid_s)
  );""")

    # -------------------- Flatten unit (transpose [0,1,3,2] + reshape) --------------------
    sig_lines.append("  // ----- Flatten unit signals -----")
    sig_lines.append("  logic [7:0] flat_data_s;")
    sig_lines.append("  logic       flat_valid_s;")
    sig_lines.append("  logic       flat_tlast_s;")
    inst_lines.append(f"""  flatten_unit #(
    .DATA_W   (8),
    .CHANNELS (POOL_CHANNELS),
    .POOL_ROWS(POOL_OUT_ROWS)
  ) u_flatten (
    .clk      (clk_i),
    .rst_n    (rst_n_i),
    .data_in  (pool_out_s),
    .valid_in (pool_valid_s),
    .data_out (flat_data_s),
    .valid_out(flat_valid_s),
    .tlast_o  (flat_tlast_s)
  );""")

    # -------------------- FC chain (paired blocks: fc_in_layer + fc_out_layer + relu) --------------------
    sig_lines.append("  // ----- FC chain signals -----")
    fc_inst_lines: List[str] = ["  // ----- FC chain instantiations -----"]
    fc_prev_data = "flat_data_s"
    fc_prev_valid = "flat_valid_s"
    for b in range(1, num_fc_blocks + 1):
        p = f"FC_{b}"
        is_last = (b == num_fc_blocks)
        sig_lines.append(f"  q_data_t fc{b}_out_s [{p}_NEURONS];")
        sig_lines.append(f"  logic    fc{b}_valid_s;")
        if not is_last:
            sig_lines.append(f"  q_data_t fc{b}_pre_relu_s;")
            sig_lines.append(f"  q_data_t fc{b}_post_relu_s;")
            sig_lines.append(f"  logic    relu{b}_in_s;")
            sig_lines.append(f"  logic    relu{b}_out_s;")
        else:
            sig_lines.append(f"  q_data_t logits_s;")
            sig_lines.append(f"  logic    logits_valid_s;")
        fc_inst_lines.append(f"""  fc_in_layer #(
    .NUM_NEURONS({p}_NEURONS),
    .INPUT_SIZE ({p}_INPUT_SIZE),
    .BIAS_SCALE ({p}_IN_BIAS_SCALE),
    .LAYER_SCALE({p}_IN_LAYER_SCALE)
  ) u_fc{b}_in_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i({fc_prev_data if False else fc_prev_valid}),
    .data_i ({fc_prev_data}),
    .data_o (fc{b}_out_s),
    .valid_o(fc{b}_valid_s)
  );""")
        if not is_last:
            fc_inst_lines.append(f"""  fc_out_layer #(
    .NUM_NEURONS({p}_NEURONS),
    .ROM_DEPTH  ({p}_ROM_DEPTH),
    .BIAS_SCALE ({p}_OUT_BIAS_SCALE),
    .LAYER_SCALE({p}_OUT_LAYER_SCALE)
  ) u_fc{b}_out_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(fc{b}_valid_s),
    .data_i (fc{b}_out_s),
    .data_o (fc{b}_pre_relu_s),
    .valid_o(relu{b}_in_s)
  );
  relu_layer u_relu{b} (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(relu{b}_in_s),
    .data_i (fc{b}_pre_relu_s),
    .data_o (fc{b}_post_relu_s),
    .valid_o(relu{b}_out_s)
  );""")
            fc_prev_data = f"fc{b}_post_relu_s"
            fc_prev_valid = f"relu{b}_out_s"
        else:
            fc_inst_lines.append(f"""  fc_out_layer #(
    .NUM_NEURONS({p}_NEURONS),
    .ROM_DEPTH  ({p}_ROM_DEPTH),
    .BIAS_SCALE ({p}_OUT_BIAS_SCALE),
    .LAYER_SCALE({p}_OUT_LAYER_SCALE)
  ) u_fc{b}_out_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(fc{b}_valid_s),
    .data_i (fc{b}_out_s),
    .data_o (logits_s),
    .valid_o(logits_valid_s)
  );""")

    # -------------------- Classifier head (softmax / argmax) --------------------
    sig_lines.append("  // ----- Classifier head + AXI output signals -----")
    if final_op_kind == "softmax":
        sig_lines.append("  q_data_t head_data_s;")
        sig_lines.append("  logic    head_valid_s;")
        head_inst = f"""  softmax_layer #(
    .NUM_CLASSES(NUM_CLASSES)
  ) u_softmax_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(logits_valid_s),
    .data_i (logits_s),
    .data_o (head_data_s),
    .valid_o(head_valid_s)
  );"""
        head_signed_extend = "{{24{head_data_s[7]}}, head_data_s}"
    else:  # argmax
        sig_lines.append("  logic [$clog2(NUM_CLASSES)-1:0] head_data_s;")
        sig_lines.append("  logic                            head_valid_s;")
        head_inst = f"""  argmax_layer #(
    .NUM_CLASSES(NUM_CLASSES)
  ) u_argmax_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(logits_valid_s),
    .data_i (logits_s),
    .data_o (head_data_s),
    .valid_o(head_valid_s)
  );"""
        head_signed_extend = "{{(DATA_WIDTH-$clog2(NUM_CLASSES)){1'b0}}, head_data_s}"

    # -------------------- AXI master + FIFO --------------------
    sig_lines.append("  logic                  fifo_empty_s;")
    sig_lines.append("  logic                  fifo_full_s;")
    sig_lines.append("  logic                  fifo_write_en_s;")
    sig_lines.append("  logic                  fifo_read_en_s;")
    sig_lines.append("  logic                  fifo_read_en_q;")
    sig_lines.append("  logic [DATA_WIDTH-1:0] fifo_read_data_s;")
    sig_lines.append("  logic [DATA_WIDTH-1:0] fifo_write_data_s;")
    if final_op_kind == "softmax":
        sig_lines.append("  logic [$clog2(NUM_CLASSES + 1) - 1:0] out_count_q;")
        out_count_term = "NUM_CLASSES"
    else:
        sig_lines.append("  logic                                  out_count_q;")
        out_count_term = "32'd1"  # argmax emits one beat per inference
    sig_lines.append("  logic                  tvalid_q;")

    axi_logic = f"""  sync_fifo #(
    .DATA_WIDTH(DATA_WIDTH),
    .DEPTH     (FIFO_DEPTH)
  ) u_pred_fifo (
    .clk_i      (clk_i),
    .rst_n_i    (rst_n_i),
    .write_en_i (fifo_write_en_s),
    .write_data_i(fifo_write_data_s),
    .full_o     (fifo_full_s),
    .read_en_i  (fifo_read_en_s),
    .read_data_o(fifo_read_data_s),
    .empty_o    (fifo_empty_s)
  );

  always_ff @(posedge clk_i or negedge rst_n_i) begin : fifo_read_pipe
    if (!rst_n_i) fifo_read_en_q <= 1'b0;
    else          fifo_read_en_q <= fifo_read_en_s;
  end : fifo_read_pipe

  always_ff @(posedge clk_i or negedge rst_n_i) begin : axi_output_logic
    if (!rst_n_i) begin
      m_axis_prediction_tdata_o <= 32'h0;
      tvalid_q                  <= 1'b0;
    end
    else if (fifo_read_en_q) begin
      m_axis_prediction_tdata_o <= fifo_read_data_s;
      tvalid_q                  <= 1'b1;
    end
    else if (m_axis_prediction_tready_i) begin
      tvalid_q <= 1'b0;
    end
  end : axi_output_logic

  always_ff @(posedge clk_i or negedge rst_n_i) begin : output_tracking
    if (!rst_n_i) begin
      out_count_q <= '0;
    end
    else if (m_axis_prediction_tvalid_o && m_axis_prediction_tready_i) begin
      if (out_count_q == ({out_count_term} - 1))
        out_count_q <= '0;
      else
        out_count_q <= out_count_q + 1'b1;
    end
  end : output_tracking

  assign fifo_write_en_s   = head_valid_s && !fifo_full_s;
  assign fifo_read_en_s    = !fifo_empty_s && m_axis_prediction_tready_i;
  assign fifo_write_data_s = {head_signed_extend};

  assign m_axis_prediction_tvalid_o = tvalid_q;
  assign m_axis_prediction_tkeep_o  = 4'h1;
  assign m_axis_prediction_tlast_o  = (m_axis_prediction_tvalid_o && (out_count_q == ({out_count_term} - 1)));
"""

    sig_block = "\n".join(sig_lines)
    inst_block = "\n".join(inst_lines)
    fc_inst_block = "\n".join(fc_inst_lines)

    body = f"""module multiclass_NN
  import quant_pkg::*;
#(
{param_block}
)(
  input  logic clk_i,
  input  logic rst_n_i,

  // AXI4-Stream Slave Interface: Input data (one int8 per cycle, row-major)
  input  q_data_t s_axis_tdata_i,
  input  logic    s_axis_tvalid_i,
  input  logic    s_axis_tlast_i,

  // AXI4-Stream Master Interface: Prediction output
  output logic [DATA_WIDTH-1:0] m_axis_prediction_tdata_o,
  output logic [KEEP_WIDTH-1:0] m_axis_prediction_tkeep_o,
  output logic                  m_axis_prediction_tvalid_o,
  input  logic                  m_axis_prediction_tready_i,
  output logic                  m_axis_prediction_tlast_o
);

  timeunit 1ns;
  timeprecision 1ps;

  // ============================================================
  // Internal signals
  // ============================================================
{sig_block}

  // ============================================================
  // Conv chain
  // ============================================================
{inst_block}

  // ============================================================
  // FC chain
  // ============================================================
{fc_inst_block}

  // ============================================================
  // Classifier head ({final_op_kind})
  // ============================================================
{head_inst}

  // ============================================================
  // Output FIFO + AXI4-Stream master
  // ============================================================
{axi_logic}

endmodule : multiclass_NN
"""
    return body


def generate_multiclass_NN_top(
    out_dir: Path,
    conv_layers: List[ConvLayerInfo],
    fc_linear_layers: List[LayerInfo],
    *,
    final_op_kind: str,
    num_classes: int,
    pool_kernel: int,
    pool_stride: int,
    pool_frame_rows: int,
    pool_channels: int,
    pool_out_rows: int,
    flatten_size: int,
    weight_width: int = 8,
    scale: int = 256,
) -> Path:
    """Emit ``multiclass_NN.sv`` (top module) into ``out_dir``."""
    fc_params = _compute_fc_chain_params(fc_linear_layers, weight_width, scale)
    body = _build_multiclass_nn_sv_content(
        conv_layers,
        fc_linear_layers,
        fc_params,
        final_op_kind=final_op_kind,
        num_classes=num_classes,
        pool_kernel=pool_kernel,
        pool_stride=pool_stride,
        pool_frame_rows=pool_frame_rows,
        pool_channels=pool_channels,
        pool_out_rows=pool_out_rows,
        flatten_size=flatten_size,
    )
    out_path = out_dir / "multiclass_NN.sv"
    desc = f"Top-level Multiclass classifier (Conv + FC + {final_op_kind}) with AXI4-Stream interfaces."
    out_path.write_text(_pyramidtech_wrap(body, "multiclass_NN.sv", desc), encoding="utf-8")
    LOGGER.info("  Generated multiclass_NN.sv (head=%s, classes=%d)", final_op_kind, num_classes)
    return out_path


def generate_multiclass_NN_wrapper(
    out_dir: Path,
    fc_linear_layers: List[LayerInfo],
    *,
    weight_width: int = 8,
    scale: int = 256,
) -> Path:
    """Emit ``multiclass_NN_wrapper.sv`` — thin pass-through (kept short for simplicity).

    The wrapper currently mirrors the top's port list; integrators that want to
    expose only architectural parameters can edit this file.
    """
    body = """module multiclass_NN_wrapper
  import quant_pkg::*;
(
  input  logic clk_i,
  input  logic rst_n_i,

  input  q_data_t s_axis_tdata_i,
  input  logic    s_axis_tvalid_i,
  input  logic    s_axis_tlast_i,

  output logic [DATA_WIDTH-1:0] m_axis_prediction_tdata_o,
  output logic [KEEP_WIDTH-1:0] m_axis_prediction_tkeep_o,
  output logic                  m_axis_prediction_tvalid_o,
  input  logic                  m_axis_prediction_tready_i,
  output logic                  m_axis_prediction_tlast_o
);

  timeunit 1ns;
  timeprecision 1ps;

  multiclass_NN u_multiclass_nn (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .s_axis_tdata_i           (s_axis_tdata_i),
    .s_axis_tvalid_i          (s_axis_tvalid_i),
    .s_axis_tlast_i           (s_axis_tlast_i),
    .m_axis_prediction_tdata_o (m_axis_prediction_tdata_o),
    .m_axis_prediction_tkeep_o (m_axis_prediction_tkeep_o),
    .m_axis_prediction_tvalid_o(m_axis_prediction_tvalid_o),
    .m_axis_prediction_tready_i(m_axis_prediction_tready_i),
    .m_axis_prediction_tlast_o (m_axis_prediction_tlast_o)
  );

endmodule : multiclass_NN_wrapper
"""
    out_path = out_dir / "multiclass_NN_wrapper.sv"
    desc = "Top-level wrapper for multiclass classifier (AXI4-Stream pass-through)."
    out_path.write_text(_pyramidtech_wrap(body, "multiclass_NN_wrapper.sv", desc), encoding="utf-8")
    return out_path


# =============================================================================
# Master orchestrator: emit_multiclass_format
#
# Sequenced steps:
#   1. Write reusable building blocks (quant_pkg, mac, fc_in/out, fc_in/out_layer,
#      relu_layer, sync_fifo).
#   2. Write multiclass-specific shells (line_buffers, dw/pw engines, avg_pool,
#      flatten_unit, softmax/argmax).
#   3. Generate per-conv-layer ROM .sv files + .mem files.
#   4. Generate FC chain mem files via generate_proj_mem_files.
#   5. Generate FC ROM .sv via _generate_fc_rom + fc_in_layer / fc_out_layer.
#   6. Generate the top module (multiclass_NN.sv) and wrapper.
#   7. Write rtl_filelist.f in dependency order.
# =============================================================================

def emit_multiclass_format(
    out_dir: Path,
    conv_layers: List[ConvLayerInfo],
    fc_layers: List[LayerInfo],
    *,
    final_op_kind: str,
    num_classes: int,
    pool_kernel: int,
    pool_stride: int,
    pool_frame_rows: int,
    pool_channels: int,
    pool_out_rows: int,
    flatten_size: int,
    weight_width: int = 8,
    scale: int = 256,
    debug_mem: bool = False,
) -> None:
    """Master function: write the entire multiclass IP to ``out_dir``."""
    fc_linear_only = [l for l in fc_layers if l.layer_type == "linear"]
    if not conv_layers and not fc_linear_only:
        raise RuntimeError("multiclass format requires at least one Conv + one FC layer.")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mem_dir = out_dir / "mem_files"
    mem_dir.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: Shared FC infrastructure -----
    (out_dir / "quant_pkg.sv").write_text(_get_quant_pkg_content(weight_width), encoding="utf-8")
    _write_embedded_sv(out_dir, "mac.sv", MAC_SV)
    _write_embedded_sv(out_dir, "fc_in.sv", FC_IN_SV)
    _write_embedded_sv(out_dir, "fc_out.sv", FC_OUT_SV)
    _write_embedded_sv(out_dir, "relu_layer.sv", RELU_LAYER_SV)
    _write_embedded_sv(out_dir, "sync_fifo.sv", SYNC_FIFO_SV)
    LOGGER.info("  Wrote shared building blocks: quant_pkg, mac, fc_in, fc_out, relu_layer, sync_fifo")

    # ----- Step 2: Multiclass-specific SV shells -----
    _generate_line_buffers_sv(out_dir)
    dw_layer_names = [l.name for l in conv_layers if l.op_kind == "depthwise"]
    pw_layer_names = [l.name for l in conv_layers if l.op_kind == "pointwise"]
    if dw_layer_names:
        _generate_dw_conv_engine_sv(out_dir, dw_layer_names)
    if pw_layer_names:
        _generate_pw_conv_engine_sv(out_dir, pw_layer_names)
    _generate_avg_pool_sv(out_dir)
    _generate_flatten_unit_sv(out_dir)
    if final_op_kind == "softmax":
        _generate_softmax_sv(out_dir)
    else:
        _generate_argmax_sv(out_dir)
    LOGGER.info(
        "  Wrote conv stack shells: line_buffers, depthwise/pointwise engines (DW=%d, PW=%d), avg_pool, flatten, %s",
        len(dw_layer_names),
        len(pw_layer_names),
        final_op_kind,
    )

    # ----- Step 3: Conv .mem files + per-layer ROM .sv -----
    for layer in conv_layers:
        rq = layer.rtl_quant
        if rq is None:
            raise RuntimeError(f"{layer.name}: missing rtl_quant for conv mem export")
        W_int = np.asarray(rq.W_int, dtype=np.int32).reshape(
            layer.out_channels, layer.in_channels, layer.kernel_h, layer.kernel_w
        )
        B_int = np.asarray(rq.B_int, dtype=np.int32).ravel()
        # Always write .mem files first (so $readmemh paths are valid at sim time)
        if layer.op_kind == "depthwise":
            window_size = layer.kernel_h * layer.kernel_w * layer.in_channels
            packed_w_bits = layer.out_channels * window_size * weight_width
            generate_dw_conv_weight_mem(W_int, mem_dir / f"{layer.name}_weights.mem", weight_width)
        else:
            packed_w_bits = layer.out_channels * layer.in_channels * weight_width
            generate_pw_conv_weight_mem(W_int, mem_dir / f"{layer.name}_weights.mem", weight_width)
        # Bias .mem: NUM_FILTERS * 32-bit acc-domain sign-extended
        # Convert int8 bias to int32 sign-extended (engine reads BIAS_WIDTH=32 per filter).
        b_acc = np.asarray(B_int, dtype=np.int64)
        # If fb_rtl > 0 we need to express the bias in F_acc domain (= Fin + Fw = F_mac).
        # Bias was quantized with fb_rtl <= F_mac; align by left-shift to F_mac.
        f_mac = int(rq.fin_qdq) + int(rq.fw_frac)
        align_shift = f_mac - int(rq.fb_rtl)
        if align_shift > 0:
            b_acc = b_acc << align_shift
        elif align_shift < 0:
            # Should not happen because fb_rtl <= F_mac; defensive.
            b_acc = b_acc >> (-align_shift)
        generate_conv_bias_mem(b_acc, mem_dir / f"{layer.name}_bias.mem", bias_width=32)
        # ROM .sv files
        packed_b_bits = layer.out_channels * 32  # BIAS_WIDTH=32
        _generate_conv_rom_sv(
            out_dir,
            f"{layer.name}_weights_rom",
            f"{layer.name}_weights.mem",
            packed_w_bits,
            description=f"Weights ROM for conv layer {layer.name}",
        )
        _generate_conv_rom_sv(
            out_dir,
            f"{layer.name}_bias_rom",
            f"{layer.name}_bias.mem",
            packed_b_bits,
            description=f"Bias ROM for conv layer {layer.name} (BIAS_WIDTH=32 per filter)",
        )

    # ----- Step 4: FC chain .mem files -----
    flatten_layer = LayerInfo(name="flatten_1", layer_type="flatten", out_shape=(1, flatten_size))
    fc_chain: List[LayerInfo] = [flatten_layer]
    for i, lyr in enumerate(fc_linear_only):
        fc_chain.append(lyr)
        if i < len(fc_linear_only) - 1:
            fc_chain.append(LayerInfo(name=f"fc_relu_{i+1}", layer_type="relu"))
    generate_proj_mem_files(fc_chain, mem_dir, scale, weight_width, debug_mem=debug_mem)

    # ----- Step 5: FC ROM .sv + fc_in_layer / fc_out_layer -----
    n_fc = len(fc_linear_only)
    num_fc_blocks = (n_fc + 1) // 2
    fc_params = _compute_fc_chain_params(fc_linear_only, weight_width, scale)
    input_sizes = [fc_params[f"FC_{b}_INPUT_SIZE"] for b in range(1, num_fc_blocks + 1)]
    rom_depths = [fc_params[f"FC_{b}_ROM_DEPTH"] for b in range(1, num_fc_blocks + 1)]
    for b in range(num_fc_blocks):
        in_idx = 2 * b
        out_idx = 2 * b + 1 if 2 * b + 1 < n_fc else None
        in_layer = fc_linear_only[in_idx]
        out_layer = fc_linear_only[out_idx] if out_idx is not None else None
        in_f = in_layer.in_features or 0
        in_out = in_layer.out_features or 0
        rom_d = (out_layer.out_features or 1) if out_layer else 1
        proj_out_f = (out_layer.out_features or 1) if out_layer else 1
        prefix = _proj_prefix(b)
        _generate_fc_rom(out_dir, f"{prefix}_in_weights_rom", in_f, in_out, weight_width)
        _generate_fc_rom(out_dir, f"{prefix}_in_bias_rom", 1, in_out, weight_width)
        _generate_fc_rom(out_dir, f"{prefix}_out_weights_rom", rom_d, proj_out_f, weight_width)
        _generate_fc_rom(out_dir, f"{prefix}_out_bias_rom", rom_d, 1, weight_width)
    _generate_fc_in_layer_module(out_dir, input_sizes, weight_width)
    _generate_fc_out_layer_module(out_dir, rom_depths, weight_width)

    # ----- Step 6: Top module + wrapper -----
    generate_multiclass_NN_top(
        out_dir,
        conv_layers,
        fc_linear_only,
        final_op_kind=final_op_kind,
        num_classes=num_classes,
        pool_kernel=pool_kernel,
        pool_stride=pool_stride,
        pool_frame_rows=pool_frame_rows,
        pool_channels=pool_channels,
        pool_out_rows=pool_out_rows,
        flatten_size=flatten_size,
        weight_width=weight_width,
        scale=scale,
    )
    generate_multiclass_NN_wrapper(out_dir, fc_linear_only, weight_width=weight_width, scale=scale)


# =============================================================================
# rtl_filelist.f generation (compilation order)
# =============================================================================

def _filelist_incdir_line(out_root: Path) -> str:
    d = out_root.resolve().as_posix().rstrip("/") + "/"
    if any(ch in d for ch in (" ", "\t")):
        return f'+incdir+"{d}"\n'
    return f"+incdir+{d}\n"


def _filelist_src_line(out_root: Path, name: str) -> str:
    p = (out_root / name).resolve().as_posix()
    if any(ch in p for ch in (" ", "\t")):
        return f'"{p}"\n'
    return f"{p}\n"


def generate_multiclass_rtl_filelist(
    out_dir: Path,
    model_name: str,
    conv_layers: List[ConvLayerInfo],
    fc_linear_layers: List[LayerInfo],
    *,
    final_op_kind: str,
) -> Path:
    """Write rtl_filelist.f listing all generated .sv files in dependency order."""
    out_root = out_dir.resolve()
    out_path = out_root / "rtl_filelist.f"
    lines: List[str] = [
        "// Auto-generated by multiclass_onnx_to_rtl.py — paths match --out-dir\n",
        f"// model: {model_name} | head: {final_op_kind} | RTL root: {out_root.as_posix()}\n",
        _filelist_incdir_line(out_root),
    ]
    files: List[str] = ["quant_pkg.sv"]

    # Per-conv-layer ROM modules (must come before engines that instantiate them)
    for layer in conv_layers:
        files.append(f"{layer.name}_weights_rom.sv")
        files.append(f"{layer.name}_bias_rom.sv")
    # FC ROMs
    n_fc = len(fc_linear_layers)
    num_fc_blocks = (n_fc + 1) // 2
    for b in range(num_fc_blocks):
        prefix = _proj_prefix(b)
        files.extend(
            [
                f"{prefix}_in_weights_rom.sv",
                f"{prefix}_in_bias_rom.sv",
                f"{prefix}_out_weights_rom.sv",
                f"{prefix}_out_bias_rom.sv",
            ]
        )
    # Building blocks + conv stack
    files.extend(
        [
            "line_buffers.sv",
        ]
    )
    if any(l.op_kind == "depthwise" for l in conv_layers):
        files.append("depthwise_conv_engine.sv")
    if any(l.op_kind == "pointwise" for l in conv_layers):
        files.append("pointwise_conv_engine.sv")
    files.extend(
        [
            "avg_pool_kx1.sv",
            "flatten_unit.sv",
        ]
    )
    # FC building blocks
    files.extend(
        [
            "mac.sv",
            "fc_in.sv",
            "fc_in_layer.sv",
            "fc_out.sv",
            "fc_out_layer.sv",
            "relu_layer.sv",
            "sync_fifo.sv",
        ]
    )
    # Classifier head
    if final_op_kind == "softmax":
        files.append("softmax_layer.sv")
    else:
        files.append("argmax_layer.sv")
    # Top + wrapper
    files.append("multiclass_NN.sv")
    files.append("multiclass_NN_wrapper.sv")

    for name in files:
        lines.append(_filelist_src_line(out_root, name))
    out_path.write_text("".join(lines), encoding="utf-8")
    LOGGER.info("  Wrote rtl_filelist.f (%d source files)", len(files))
    return out_path


# =============================================================================
# Mapping report (text summary of generated IP)
# =============================================================================

def generate_multiclass_mapping_report(
    out_dir: Path,
    model_name: str,
    conv_layers: List[ConvLayerInfo],
    fc_linear_layers: List[LayerInfo],
    *,
    final_op_kind: str,
    num_classes: int,
    pool_kernel: int,
    pool_stride: int,
    pool_frame_rows: int,
    pool_out_rows: int,
    flatten_size: int,
    weight_width: int = 8,
) -> Path:
    """Write a human-readable mapping_report.txt summarising the generated multiclass IP."""
    lines: List[str] = []
    lines.append("Multiclass RTL Mapping Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Model:           {model_name}")
    lines.append(f"Weight width:    {weight_width} bits (int8)")
    lines.append(f"Classifier head: {final_op_kind}")
    lines.append(f"Num classes:     {num_classes}")
    lines.append("")
    lines.append("Convolution layers:")
    lines.append("-" * 80)
    for layer in conv_layers:
        rq = layer.rtl_quant
        ls = compute_conv_layer_scale(layer) if rq else 0
        bs = compute_conv_bias_scale(layer) if rq else 0
        lines.append(
            f"  {layer.name:30s} {layer.op_kind:10s} "
            f"in_ch={layer.in_channels:3d} out_ch={layer.out_channels:3d} "
            f"K={layer.kernel_h}x{layer.kernel_w}  stride={layer.stride_h}x{layer.stride_w}"
        )
        if rq:
            lines.append(
                f"     Fin={rq.fin_qdq}  Fw={rq.fw_frac}  Fb={rq.fb_rtl}  Fout={rq.fout_qdq}  "
                f"LAYER_SCALE={ls}  BIAS_SCALE={bs}"
            )
            lines.append(
                f"     w_sat(lo,hi)=({rq.weight_sat_lo_pct:.2f}%,{rq.weight_sat_hi_pct:.2f}%)  "
                f"b_sat(lo,hi)=({rq.bias_sat_lo_pct:.2f}%,{rq.bias_sat_hi_pct:.2f}%)"
            )
        if layer.activation:
            lines.append(f"     activation: {layer.activation}")
    lines.append("")
    lines.append(f"AvgPool: kernel={pool_kernel}x1  stride={pool_stride}  frame_rows={pool_frame_rows}  out_rows={pool_out_rows}")
    lines.append(f"Flatten size: {flatten_size}  (channels x pool_out_rows)")
    lines.append("")
    lines.append("Fully-connected layers (paired into blocks):")
    lines.append("-" * 80)
    for i, lyr in enumerate(fc_linear_layers):
        block = i // 2 + 1
        half = "in" if i % 2 == 0 else "out"
        lines.append(
            f"  fc{i+1}  block {block} {half:3s}  in={lyr.in_features:5d}  out={lyr.out_features:5d}  activation={lyr.activation or '-'}"
        )
    lines.append("")
    out_path = out_dir / "mapping_report.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("  Wrote mapping_report.txt")
    return out_path


def generate_multiclass_netlist_json(
    out_dir: Path,
    model_name: str,
    conv_layers: List[ConvLayerInfo],
    fc_linear_layers: List[LayerInfo],
    *,
    final_op_kind: str,
    num_classes: int,
) -> Path:
    """Write netlist.json — machine-readable summary of the generated IP."""
    convs = []
    for layer in conv_layers:
        rq = layer.rtl_quant
        convs.append(
            {
                "name": layer.name,
                "op_kind": layer.op_kind,
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel": [layer.kernel_h, layer.kernel_w],
                "stride": [layer.stride_h, layer.stride_w],
                "pad": [layer.pad_h, layer.pad_w],
                "in_h": layer.in_h,
                "in_w": layer.in_w,
                "out_h": layer.out_h,
                "out_w": layer.out_w,
                "activation": layer.activation,
                "fin_qdq": rq.fin_qdq if rq else None,
                "fw_frac": rq.fw_frac if rq else None,
                "fb_rtl": rq.fb_rtl if rq else None,
                "fout_qdq": rq.fout_qdq if rq else None,
                "layer_scale": compute_conv_layer_scale(layer) if rq else None,
            }
        )
    fcs = []
    for i, lyr in enumerate(fc_linear_layers):
        rq = lyr.rtl_quant
        fcs.append(
            {
                "name": lyr.name,
                "in_features": lyr.in_features,
                "out_features": lyr.out_features,
                "activation": lyr.activation,
                "fin_qdq": rq.fin_qdq if rq else None,
                "fw_frac": rq.fw_frac if rq else None,
                "fb_rtl": rq.fb_rtl if rq else None,
                "fout_qdq": rq.fout_qdq if rq else None,
            }
        )
    netlist = {
        "model": model_name,
        "head": final_op_kind,
        "num_classes": num_classes,
        "conv_layers": convs,
        "fc_layers": fcs,
    }
    out_path = out_dir / "netlist.json"
    out_path.write_text(json.dumps(netlist, indent=2), encoding="utf-8")
    LOGGER.info("  Wrote netlist.json")
    return out_path
