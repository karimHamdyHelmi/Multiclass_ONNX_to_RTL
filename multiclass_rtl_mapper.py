#!/usr/bin/env python3
"""
RTL generation engine for ``multiclass_onnx_to_rtl.py``.
========================================================

Takes per-layer Conv + FC weight/bias arrays (with quantization metadata) and emits
synthesizable SystemVerilog modules plus ``.mem`` weight/bias ROM files for a complete
multiclass classifier IP.

**Pipeline overview (matches binary script convention but extended for Conv layers):**

1. **Fixed-point quantization** (``build_rtl_conv_quant_descriptors`` / FC reuse) — convert float32
   weights/biases to power-of-two int8 (W_int, B_int) with exponents Fw, Fb.

2. **Scale derivation** — derive per-layer ``LAYER_SCALE = (Fin + Fw) - Fout`` and
   ``BIAS_SCALE = (Fin + Fw) - Fb``, expressed as signed integer right-/left-shifts.

3. **Memory file generation** — emit packed ``.mem`` files in the layout required by:
     - depthwise_conv_engine.sv    : single ROM row, ``NUM_FILTERS * (K_H * K_W * K_C) * Q_WIDTH`` bits
     - pointwise_conv_engine.sv    : single ROM row, ``NUM_FILTERS * INPUTS_PER_CYCLE * Q_WIDTH`` bits
     - conv bias ROMs              : single ROM row, ``NUM_FILTERS * BIAS_WIDTH`` bits (sign-extended int32)
     - fc_in / fc_out ROMs         : reuse binary script packing (see binary rtl_mapper.py)

4. **SystemVerilog emission** (``emit_multiclass_format``) — write the entire IP:
     - Reused from binary script: quant_pkg.sv, mac.sv, fc_in.sv, fc_out.sv,
       fc_in_layer.sv, fc_out_layer.sv, relu_layer.sv, sync_fifo.sv
     - New (multiclass-specific): line_buffers.sv, depthwise_conv_engine.sv,
       pointwise_conv_engine.sv, avg_pool_kx1.sv, flatten_unit.sv,
       softmax_layer.sv / argmax_layer.sv (chosen from ONNX final op),
       multiclass_NN.sv (top module), multiclass_NN_wrapper.sv

The script imports shared infrastructure from the binary script's ``rtl_mapper.py``
(found in the sibling ``Binary_ONNX_to_RTL`` directory).  This avoids duplicating ~3000
lines of proven FC chain code while letting the multiclass mapper add only conv-specific
emitters and the new top-module template.
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

# -----------------------------------------------------------------------------
# Locate and import shared infrastructure from the binary script's rtl_mapper.
# This is the sibling project at <pyramidstech>/Binary_ONNX_to_RTL/rtl_mapper.py
# from which we reuse: LayerInfo, RtlLayerQuantDescriptor, header utilities,
# weight/bias quantization, FC mem packers, scale alignment helpers, and the
# proven SV templates for quant_pkg / mac / fc_in / fc_out / relu / sync_fifo.
# -----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_BINARY_DIR_CANDIDATES = [
    _THIS_DIR.parent / "pyramidstech" / "Binary_ONNX_to_RTL",
    _THIS_DIR.parent.parent / "pyramidstech" / "Binary_ONNX_to_RTL",
    Path(r"C:/Users/Kimo_/OneDrive - Alexandria University/Desktop/pyramidstech/Binary_ONNX_to_RTL"),
]
_BINARY_DIR: Optional[Path] = None
for _cand in _BINARY_DIR_CANDIDATES:
    if (_cand / "rtl_mapper.py").is_file():
        _BINARY_DIR = _cand.resolve()
        break
if _BINARY_DIR is None:
    raise RuntimeError(
        "Cannot locate sibling Binary_ONNX_to_RTL directory containing rtl_mapper.py. "
        "Searched: " + ", ".join(str(c) for c in _BINARY_DIR_CANDIDATES)
    )
if str(_BINARY_DIR) not in sys.path:
    sys.path.insert(0, str(_BINARY_DIR))

import rtl_mapper as _bin_rtl  # noqa: E402

# Re-export the names we use directly in this module (and in multiclass_onnx_to_rtl.py).
LayerInfo = _bin_rtl.LayerInfo
RtlLayerQuantDescriptor = _bin_rtl.RtlLayerQuantDescriptor
quantize_weight_for_rtl = _bin_rtl.quantize_weight_for_rtl
quantize_bias_for_rtl = _bin_rtl.quantize_bias_for_rtl
build_rtl_layer_quant_descriptors = _bin_rtl.build_rtl_layer_quant_descriptors
compute_layer_scales_from_rtl_descriptors = _bin_rtl.compute_layer_scales_from_rtl_descriptors
_qdq_pair_bias_alignment = _bin_rtl._qdq_pair_bias_alignment
_qdq_pair_layer_alignment = _bin_rtl._qdq_pair_layer_alignment
_rtl_signed_bias_scale = _bin_rtl._rtl_signed_bias_scale
_rtl_signed_layer_scale = _bin_rtl._rtl_signed_layer_scale
_pyramidtech_wrap = _bin_rtl._pyramidtech_wrap
_pyramidtech_header = _bin_rtl._pyramidtech_header
_get_quant_pkg_content = _bin_rtl._get_quant_pkg_content
_write_embedded_sv = _bin_rtl._write_embedded_sv
generate_proj_mem_files = _bin_rtl.generate_proj_mem_files
_proj_prefix = _bin_rtl._proj_prefix
_generate_binaryclass_nn_rom = _bin_rtl._generate_binaryclass_nn_rom
_generate_fc_in_layer_binaryclass = _bin_rtl._generate_fc_in_layer_binaryclass
_generate_fc_out_layer_binaryclass = _bin_rtl._generate_fc_out_layer_binaryclass
_compute_binaryclass_nn_params = _bin_rtl._compute_binaryclass_nn_params

BINARYCLASS_MAC_SV = _bin_rtl.BINARYCLASS_MAC_SV
BINARYCLASS_FC_IN_SV = _bin_rtl.BINARYCLASS_FC_IN_SV
BINARYCLASS_FC_OUT_SV = _bin_rtl.BINARYCLASS_FC_OUT_SV
BINARYCLASS_RELU_LAYER_SV = _bin_rtl.BINARYCLASS_RELU_LAYER_SV
BINARYCLASS_SYNC_FIFO_SV = _bin_rtl.BINARYCLASS_SYNC_FIFO_SV

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


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
# used for FC weights and matches the binary script.
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
    methodology as ``build_rtl_layer_quant_descriptors`` (binary FC path).

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
    # FC block parameters (mirror binary script's _compute_binaryclass_nn_params)
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
    fc_params = _compute_binaryclass_nn_params(fc_linear_layers, weight_width, scale)
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
# Sequenced steps (mirrors emit_binaryclass_nn_format):
#   1. Write reusable building blocks (quant_pkg, mac, fc_in/out, fc_in/out_layer,
#      relu_layer, sync_fifo).
#   2. Write multiclass-specific shells (line_buffers, dw/pw engines, avg_pool,
#      flatten_unit, softmax/argmax).
#   3. Generate per-conv-layer ROM .sv files + .mem files.
#   4. Generate FC chain mem files via the binary script's generate_proj_mem_files.
#   5. Generate FC ROM .sv via _generate_binaryclass_nn_rom + fc_in_layer / fc_out_layer.
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

    # ----- Step 1: Reusable building blocks (from binary script) -----
    (out_dir / "quant_pkg.sv").write_text(_get_quant_pkg_content(weight_width), encoding="utf-8")
    _write_embedded_sv(out_dir, "mac.sv", BINARYCLASS_MAC_SV)
    _write_embedded_sv(out_dir, "fc_in.sv", BINARYCLASS_FC_IN_SV)
    _write_embedded_sv(out_dir, "fc_out.sv", BINARYCLASS_FC_OUT_SV)
    _write_embedded_sv(out_dir, "relu_layer.sv", BINARYCLASS_RELU_LAYER_SV)
    _write_embedded_sv(out_dir, "sync_fifo.sv", BINARYCLASS_SYNC_FIFO_SV)
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

    # ----- Step 4: FC chain .mem files (reuse binary script) -----
    flatten_layer = LayerInfo(name="flatten_1", layer_type="flatten", out_shape=(1, flatten_size))
    fc_chain: List[LayerInfo] = [flatten_layer]
    for i, lyr in enumerate(fc_linear_only):
        fc_chain.append(lyr)
        if i < len(fc_linear_only) - 1:
            fc_chain.append(LayerInfo(name=f"fc_relu_{i+1}", layer_type="relu"))
    generate_proj_mem_files(fc_chain, mem_dir, scale, weight_width, debug_mem=debug_mem)

    # ----- Step 5: FC ROM .sv + fc_in_layer / fc_out_layer (reuse binary script) -----
    n_fc = len(fc_linear_only)
    num_fc_blocks = (n_fc + 1) // 2
    fc_params = _compute_binaryclass_nn_params(fc_linear_only, weight_width, scale)
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
        _generate_binaryclass_nn_rom(out_dir, f"{prefix}_in_weights_rom", in_f, in_out, weight_width)
        _generate_binaryclass_nn_rom(out_dir, f"{prefix}_in_bias_rom", 1, in_out, weight_width)
        _generate_binaryclass_nn_rom(out_dir, f"{prefix}_out_weights_rom", rom_d, proj_out_f, weight_width)
        _generate_binaryclass_nn_rom(out_dir, f"{prefix}_out_bias_rom", rom_d, 1, weight_width)
    _generate_fc_in_layer_binaryclass(out_dir, input_sizes, weight_width)
    _generate_fc_out_layer_binaryclass(out_dir, rom_depths, weight_width)

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
