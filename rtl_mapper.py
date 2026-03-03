#!/usr/bin/env python3
"""
Minimal rtl_mapper for multiclass_onnx_to_rtl.py.
Contains only the code needed for multiclass ONNX-to-RTL conversion (FC layers, softmax).
No PyTorch model loading, MNIST, or binary-specific logic.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def float_to_int(val: np.ndarray, scale: int, bit_width: int) -> np.ndarray:
    """Quantize float array to signed integer with specified bit width."""
    if bit_width == 4:
        min_val, max_val = -8, 7
        dtype = np.int8  # Use int8 to store, but values are in [-8, 7]
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


# Model Introspection

@dataclass
class LayerInfo:
    """Information about a layer in the model."""
    name: str
    layer_type: str  # 'flatten', 'linear', 'relu', 'softmax'
    module_qualname: Optional[str] = None
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    weight: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    in_shape: Optional[Tuple[int, ...]] = None
    out_shape: Optional[Tuple[int, ...]] = None


# Memory File Generation (quant_pkg style only)

def generate_quant_pkg_style_weight_mem(
    weight_matrix: np.ndarray,
    out_path: Path,
    layer_name: str,
    in_features: int,
    out_features: int,
    bit_width: int
) -> None:
    """
    Generate weight .mem in quant_pkg format (same as Desktop/rtl).
    - fc1: DEPTH=in_features, each line = all num_neurons weights for that input (neuron 0 in LSBs).
    - fc2/fc3: DEPTH=out_features, each line = all in_features weights for that neuron (input 0 in LSBs).
    """
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
                wf.write(f"{packed:0{hex_width}x}\n")
        else:
            num_inputs = in_features
            total_bits = num_inputs * bit_width
            hex_width = (total_bits + 3) // 4
            for neuron_idx in range(out_features):
                packed = 0
                for inp_idx in range(num_inputs):
                    val = int(weight_matrix[neuron_idx, inp_idx]) & mask
                    packed |= val << (inp_idx * bit_width)
                wf.write(f"{packed:0{hex_width}x}\n")


def generate_quant_pkg_style_bias_mem(
    bias_vector: np.ndarray,
    out_path: Path,
    num_neurons: int,
    bit_width: int
) -> None:
    """
    Generate bias .mem in quant_pkg format: single line, all biases packed (neuron 0 in LSBs).
    """
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
        bf.write(f"{packed:0{hex_width}x}\n")


def generate_quant_pkg_style_mems(
    layer_name: str,
    weight_matrix: np.ndarray,
    bias_vector: np.ndarray,
    bit_width: int,
    rtl_root: Path,
) -> None:
    """
    Generate mem files compatible with the existing quant_pkg / rtl/ flow.

    Expected format (for Q_WIDTH = 4/8/16):
    - Weights:
      * fc1:  fc1_weights_<bits>.mem
            DEPTH = in_features (e.g. 784)
            each line packs all NUM_NEURONS weights for that input index
              (neuron 0 in least-significant bits).
      * fc2/3: fc{N}_weights_<bits>.mem
            DEPTH = out_features (neurons)
            each line packs all NUM_INPUTS weights for that neuron
              (input 0 in least-significant bits).
    - Biases:
      * fc{N}_biases_<bits>.mem
            single line; packs all neuron biases, neuron 0 in LSBs.
    """
    if bit_width not in (4, 8, 16):
        raise ValueError(f"Unsupported bit width for quant_pkg-style mems: {bit_width}. Supported: 4, 8, 16")

    mem_dir = rtl_root / "mem"
    mem_dir.mkdir(parents=True, exist_ok=True)

    weight_matrix = np.asarray(weight_matrix, dtype=np.int32)
    bias_vector = np.asarray(bias_vector, dtype=np.int32)

    out_features, in_features = weight_matrix.shape
    mask = (1 << bit_width) - 1  # 0xF for 4-bit two's complement

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    weights_path = mem_dir / f"{layer_name}_weights_{bit_width}.mem"
    with weights_path.open("w", encoding="utf-8") as wf:
        if layer_name == "fc1":
            # fc1: address = input index, word packs all neuron weights
            num_neurons = out_features
            total_bits = num_neurons * bit_width
            hex_width = total_bits // 4

            for j in range(in_features):
                packed = 0
                # neuron index increases with bit position
                for neuron_idx in range(num_neurons):
                    val = int(weight_matrix[neuron_idx, j]) & mask
                    packed |= val << (neuron_idx * bit_width)
                wf.write(f"{packed:0{hex_width}x}\n")
        else:
            # fc2 / fc3 style:
            # address = neuron index, word packs all input weights
            num_inputs = in_features
            total_bits = num_inputs * bit_width
            hex_width = total_bits // 4

            for neuron_idx in range(out_features):
                packed = 0
                for inp_idx in range(num_inputs):
                    val = int(weight_matrix[neuron_idx, inp_idx]) & mask
                    packed |= val << (inp_idx * bit_width)
                wf.write(f"{packed:0{hex_width}x}\n")

    # ------------------------------------------------------------------
    # Biases
    # ------------------------------------------------------------------
    biases_path = mem_dir / f"{layer_name}_biases_{bit_width}.mem"
    num_neurons = bias_vector.shape[0]
    total_bits = num_neurons * bit_width
    hex_width = (total_bits + 3) // 4  # should be num_neurons for 4-bit

    packed_biases = 0
    for neuron_idx in range(num_neurons):
        val = int(bias_vector[neuron_idx]) & mask
        packed_biases |= val << (neuron_idx * bit_width)

    with biases_path.open("w", encoding="utf-8") as bf:
        bf.write(f"{packed_biases:0{hex_width}x}\n")


def emit_legacy_rtl_outputs(
    legacy_rtl_dir: Path,
    layers: List[LayerInfo],
    scale: int,
    bits_list: Sequence[int] = (4, 8, 16),
    write_sv: bool = True,
) -> None:
    """
    Emit the legacy `rtl/` flow outputs:
    - `legacy_rtl_dir/*.sv` (optionally)
    - `legacy_rtl_dir/mem/fc*_weights_{4,8,16}.mem` and `fc*_biases_{4,8,16}.mem`

    Quantization scale uses the provided `scale` for all bit widths to match the
    existing project convention (QuantizedMNISTNet uses SCALE_FACTOR=256).
    """
    legacy_rtl_dir = legacy_rtl_dir.resolve()
    # write_sv: legacy MNIST RTL removed; multiclass uses embedded templates

    for layer in layers:
        if layer.layer_type != "linear":
            continue

        if layer.weight is None:
            raise RuntimeError(f"Linear layer {layer.name} missing weights")

        weight_np = layer.weight.detach().cpu().numpy().astype(np.float32)
        bias_np = (
            layer.bias.detach().cpu().numpy().astype(np.float32)
            if layer.bias is not None
            else np.zeros((weight_np.shape[0],), dtype=np.float32)
        )

        for bits in bits_list:
            wq = float_to_int(weight_np, scale, bits)
            bq = float_to_int(bias_np, scale, bits)
            generate_quant_pkg_style_mems(layer.name, wq, bq, bits, legacy_rtl_dir)


# ============================================================================
# Embedded RTL Templates (from rtl folder - quant_pkg, mac, fc_in, relu_layer)
# ============================================================================

def _get_quant_pkg_content(weight_width: int) -> str:
    """Generate quant_pkg.sv with Q_WIDTH based on weight_width. Bakes in Q_INT4/8/16."""
    if weight_width not in (4, 8, 16):
        weight_width = 16
    define_line = f"`define Q_INT{weight_width}\n"
    default_def = "`ifndef Q_INT4\n`ifndef Q_INT8\n`ifndef Q_INT16\n`define Q_INT16\n`endif\n`endif\n`endif\n"
    return define_line + default_def + '''package quant_pkg;

   `ifdef Q_INT4
      localparam int Q_WIDTH = 4;
   `elsif Q_INT8
      localparam int Q_WIDTH = 8;
   `elsif Q_INT16
      localparam int Q_WIDTH = 16;
   `endif
   
   `ifdef Q_INT4
      localparam int FRAC_WIDTH = 2;
   `elsif Q_INT8
      localparam int FRAC_WIDTH = 4;
   `elsif Q_INT16
      localparam int FRAC_WIDTH = 8;
   `endif
   
   `ifdef Q_INT4
      localparam int SATURATION_H = 32'sd7;
   `elsif Q_INT8
      localparam int SATURATION_H = 32'sd127;
   `elsif Q_INT16
      localparam int SATURATION_H = 32'sd32767;
   `endif
   
   `ifdef Q_INT4
      localparam int SATURATION_L = -32'sd8;
   `elsif Q_INT8
      localparam int SATURATION_L = -32'sd128;
   `elsif Q_INT16
      localparam int SATURATION_L = -32'sd32768;
   `endif
   
   `ifdef Q_INT4
      localparam int LIMIT_H = 4'sd7;
   `elsif Q_INT8
      localparam int LIMIT_H = 8'sd127;
   `elsif Q_INT16
      localparam int LIMIT_H = 16'sd32767;
   `endif
   
   `ifdef Q_INT4
      localparam int LIMIT_L = -4'sd8;
   `elsif Q_INT8
      localparam int LIMIT_L = -8'sd128;
   `elsif Q_INT16
      localparam int LIMIT_L = -16'sd32768;
   `endif
   
   typedef logic signed [Q_WIDTH-1:0]  q_data_t;
   typedef logic        [Q_WIDTH-1:0]  pred_t;
   typedef logic signed [31:0]         acc_t;
   // Softmax output: 16-bit unsigned, 0.16 fixed-point (1.0 = 65535)
   typedef logic        [15:0]         prob_t;

endpackage
'''


EMBEDDED_MAC_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module mac (
   input  logic clk,
   input  logic rst_n,
   input  logic enable,
   input  logic clear,

   input  q_data_t a,
   input  q_data_t b,

   output logic out_valid,
   output acc_t acc
);

   (* use_dsp = "yes" *) logic signed [63:0] mult_reg;
   logic signed [63:0] acc_reg;
   logic               enable_q;
   logic               enable_qq;

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
         mult_reg <= '0;
      else if (enable)
         mult_reg <= $signed(a) * $signed(b);
   end

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
         acc_reg <= '0;
      else if (clear)
         acc_reg <= '0;
      else if (enable)
         acc_reg <= acc_reg + mult_reg;
   end

   always_ff @(posedge clk) begin
      if (acc_reg > $signed(32'h7FFFFFFF))
         acc <= 32'h7FFFFFFF;
      else if (acc_reg < $signed(32'h80000000))
         acc <= 32'h80000000;
      else
         acc <= acc_reg[31:0];
   end
   
   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)begin
         enable_q  <= 1'b0;
         enable_qq <= 1'b0;
         out_valid <= 1'b0;
      end
      else if (clear)begin
         enable_q  <= 1'b0;
         enable_qq <= 1'b0;
         out_valid <= 1'b0;
      end
      else if (enable)begin
         enable_q  <= enable;
         enable_qq <= enable_q;
         out_valid <= enable_qq;
      end
   end

endmodule
'''


EMBEDDED_FC_IN_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module fc_in #(
   parameter int NUM_NEURONS = 8,
   parameter int INPUT_SIZE  = 16
)(
   input  logic                 clk,
   input  logic                 rst_n,

   input  logic                 in_valid,
   input  q_data_t              data_in,

   input  q_data_t              weights [NUM_NEURONS],
   input  q_data_t              bias    [NUM_NEURONS],

   output q_data_t              fc_out  [NUM_NEURONS],
   output logic                 out_valid
);

   acc_t mac_acc     [NUM_NEURONS];
   acc_t mac_acc_tmp [NUM_NEURONS];
   acc_t acc_tmp [NUM_NEURONS];
   acc_t bias_32 [NUM_NEURONS];

   logic [$clog2(INPUT_SIZE+1)-1:0] out_count;
   logic mac_enable;
   logic mac_clear;
   logic [NUM_NEURONS-1:0]mac_out_valid;
   logic all_mac_valid;

   genvar i;
   generate
      for (i = 0; i < NUM_NEURONS; i++) begin : FC_MACS
         mac mac_inst (
            .clk       (clk),
            .rst_n     (rst_n),
            .enable    (mac_enable),
            .clear     (mac_clear),
            .a         (data_in),
            .b         (weights[i]),
            .out_valid (mac_out_valid[i]),
            .acc       (mac_acc[i])
         );
      end
   endgenerate

   assign all_mac_valid = &mac_out_valid;
   
   always_ff @(posedge clk) begin
      if (!rst_n) begin
         out_count  <= '0;
         out_valid <= 1'b0;
      end else begin
         out_valid <= 1'b0;
         if (all_mac_valid) begin
            if (out_count == INPUT_SIZE-1) begin
               out_count  <= '0;
               out_valid <= 1'b1;
            end else begin
               out_count <= out_count + 1'b1;
            end
         end
      end
   end

   assign mac_enable = in_valid;
   assign mac_clear  = out_valid;

   generate
      for (i = 0; i < NUM_NEURONS; i++) begin : FC_BIAS_ADD
         assign mac_acc_tmp[i] = (mac_acc[i] >>> FRAC_WIDTH);
         assign bias_32[i] = {{32-Q_WIDTH{bias[i][Q_WIDTH-1]}}, bias[i]};
         assign acc_tmp[i] = mac_acc_tmp[i] + bias_32[i];
         always_ff @(posedge clk) begin
            if (!rst_n)
               fc_out[i] <= '0;
            else if (out_count == INPUT_SIZE-1)
               if (acc_tmp[i] > SATURATION_H)
                  fc_out[i] <= LIMIT_H;
               else if (acc_tmp[i] < SATURATION_L)
                  fc_out[i] <= $signed(LIMIT_L);
               else
                  fc_out[i] <= acc_tmp[i][Q_WIDTH:0];
         end
      end
   endgenerate

endmodule
'''


EMBEDDED_RELU_LAYER_SV = '''`timescale 1ns / 1ps
import quant_pkg::*;

module relu_layer #(
   parameter int NUM_NEURONS = 8
)(
    input  logic         clk,
    input  logic         rst_n,
    input  logic         valid_in,
    input  q_data_t      data_in  [NUM_NEURONS],   

    output q_data_t      data_out [NUM_NEURONS],  
    output logic         valid_out
);

    logic valid_pipeline;
    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipeline <= 1'b0;
        else
            valid_pipeline <= valid_in;
    end

    assign valid_out = valid_pipeline;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_NEURONS; i++)
                data_out[i] <= '0;
        end
        else if (valid_in) begin
            for (i = 0; i < NUM_NEURONS; i++) begin
                if (data_in[i] < 0)
                    data_out[i] <= '0;
                else
                    data_out[i] <= data_in[i];
            end
        end
    end

endmodule
'''


def _get_softmax_exp_lut_initial() -> str:
    """Generate 256-entry exp LUT initial block for softmax. exp(-i/32) in 0.16 format."""
    import math
    lines = []
    for i in range(256):
        val = math.exp(-i / 32.0)
        scaled = int(round(val * 65535))
        scaled = min(scaled, 65535)
        lines.append(f"        exp_lut[{i}] = 16'h{scaled:04X};")
    return "\n".join(lines)


def _get_softmax_layer_sv() -> str:
    """Generate softmax_layer.sv content with embedded exp LUT."""
    lut_init = _get_softmax_exp_lut_initial()
    return f'''`timescale 1ns/1ps
import quant_pkg::*;

//------------------------------------------------------------------------------
// Module: softmax_layer
// Description: Softmax activation. Converts logits to probabilities (0.16 format).
//   Algorithm: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
//   Uses LUT for exp approximation. Output: prob_t (16-bit, 1.0 = 65535)
//------------------------------------------------------------------------------
module softmax_layer #(
   parameter int NUM_CLASSES = 10
)(
    input  logic         clk,
    input  logic         rst_n,
    input  logic         valid_in,
    input  q_data_t      logits_in [NUM_CLASSES],

    output prob_t        probs_out [NUM_CLASSES],
    output logic         valid_out
);

    // exp LUT: exp(-i/32) in 0.16 format, index 0 = exp(0)=1.0
    logic [15:0] exp_lut [0:255];

    initial begin
{lut_init}
    end

    // Stage 1: find max, register inputs
    q_data_t logits_reg [NUM_CLASSES];
    q_data_t max_reg;
    logic valid_s1;

    always_comb begin
        max_reg = logits_reg[0];
        for (int i = 1; i < NUM_CLASSES; i++)
            if (logits_reg[i] > max_reg) max_reg = logits_reg[i];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1 <= 1'b0;
            for (int i = 0; i < NUM_CLASSES; i++) logits_reg[i] <= '0;
        end else begin
            valid_s1 <= valid_in;
            for (int i = 0; i < NUM_CLASSES; i++) logits_reg[i] <= logits_in[i];
        end
    end

    // Stage 2: subtract max, lookup exp (delta in [-255,0], use -delta as idx)
    logic [15:0] exp_vals [NUM_CLASSES];
    logic [15:0] exp_vals_reg [NUM_CLASSES];
    logic [31:0] sum_exp;
    logic valid_s2;

    always_comb begin
        for (int i = 0; i < NUM_CLASSES; i++) begin
            automatic logic signed [31:0] delta = $signed(logits_reg[i]) - $signed(max_reg);
            automatic int idx = (delta < 0) ? (-delta) : 0;
            if (idx > 255) idx = 255;
            exp_vals[i] = exp_lut[idx];
        end
    end

    always_comb begin
        sum_exp = 0;
        for (int i = 0; i < NUM_CLASSES; i++)
            sum_exp = sum_exp + exp_vals[i];
        if (sum_exp == 0) sum_exp = 1;  // avoid div by zero
    end

    logic [31:0] sum_exp_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s2 <= 1'b0;
            sum_exp_reg <= 1;
            for (int i = 0; i < NUM_CLASSES; i++) exp_vals_reg[i] <= '0;
        end else begin
            valid_s2 <= valid_s1;
            sum_exp_reg <= sum_exp;
            for (int i = 0; i < NUM_CLASSES; i++) exp_vals_reg[i] <= exp_vals[i];
        end
    end

    // Stage 3: divide (prob = exp * 65535 / sum)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (int i = 0; i < NUM_CLASSES; i++) probs_out[i] <= '0;
        end else begin
            valid_out <= valid_s2;
            if (valid_s2)
                for (int i = 0; i < NUM_CLASSES; i++)
                    probs_out[i] <= (exp_vals_reg[i] * 65535) / sum_exp_reg;
        end
    end

endmodule
'''


EMBEDDED_FC_OUT_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module fc_out #(
    parameter int NUM_MULT = 2
)(
    input  logic clk,
    input  logic rst_n,
    input  logic en,

    input  q_data_t    fc_in   [NUM_MULT],
    input  q_data_t    weights [NUM_MULT],
    input  q_data_t    bias,

    output logic       fc_out_valid,
    output q_data_t    fc_out
);

    (* use_dsp = "yes" *) logic signed [63:0] mult_res [NUM_MULT];
    logic signed [63:0] mult_res_tmp [NUM_MULT];
    logic signed [63:0] sum_comb;

    genvar i;
    generate
        for (i = 0; i < NUM_MULT; i++) begin : GEN_MULT
            assign mult_res[i]     = $signed(fc_in[i]) * $signed(weights[i]);
            assign mult_res_tmp[i] = (mult_res[i] >>> FRAC_WIDTH);
        end
    endgenerate

    integer k;
    always_comb begin
        sum_comb = {{64-Q_WIDTH{bias[Q_WIDTH-1]}}, bias};
        for (k = 0; k < NUM_MULT; k++)
            sum_comb = sum_comb + mult_res_tmp[k];
    end

    always_ff @(posedge clk or negedge rst_n) begin
       if (!rst_n) begin
           fc_out       <= '0;
           fc_out_valid <= 1'b0;
       end
       else if (en) begin
          fc_out_valid <= 1'b1;
          if (sum_comb > SATURATION_H)
             fc_out <= LIMIT_H;
          else if (sum_comb < SATURATION_L)
             fc_out <= $signed(LIMIT_L);
          else
             fc_out <= sum_comb[Q_WIDTH:0];
       end
       else
          fc_out_valid <= 1'b0;
    end

endmodule
'''


def generate_fc_out_layer(
    layer_name: str,
    num_neurons: int,
    num_inputs: int,
    out_dir: Path
) -> Path:
    """Generate fc_out_layer_<layer>.sv for fc2/fc3 (quant_pkg format, by-neuron ROM)."""
    content = f"""`timescale 1ns/1ps
import quant_pkg::*;

module fc_out_layer_{layer_name} #(
    parameter int NUM_NEURONS = {num_neurons},
    parameter int NUM_INPUTS  = {num_inputs}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,

    input  q_data_t fc_in [NUM_INPUTS],

    output q_data_t fc_out [NUM_NEURONS],
    output logic    valid_out
);

    logic [$clog2(NUM_NEURONS)-1:0] addr_cnt;
    logic valid_in_reg;
    q_data_t fc_in_reg [NUM_INPUTS];

    logic [NUM_INPUTS*Q_WIDTH-1:0] weights_rom_data_raw;
    q_data_t weights_rom_data [NUM_INPUTS];
    q_data_t bias_rom_data [NUM_NEURONS];
    logic [NUM_NEURONS*Q_WIDTH-1:0] bias_rom_data_raw;
    q_data_t bias_rom_data_reg;

    logic fc_out_valid;
    q_data_t fc_out_tmp;
    logic [$clog2(NUM_NEURONS+1)-1:0] fc_out_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            addr_cnt <= '0;
        else if (addr_cnt == NUM_NEURONS-1)
            addr_cnt <= '0;
        else if (valid_in || addr_cnt > 0)
            addr_cnt <= addr_cnt + 1'b1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_in_reg <= 1'b0;
            for (int j = 0; j < NUM_INPUTS; j++)
                fc_in_reg[j] <= '0;
        end
        else begin
            if (addr_cnt == '0) begin
                valid_in_reg <= valid_in;
                fc_in_reg    <= fc_in;
            end
            else if (addr_cnt == NUM_NEURONS) begin
                valid_in_reg <= 1'b0;
                for (int j = 0; j < NUM_INPUTS; j++)
                    fc_in_reg[j] <= '0;
            end
        end
    end

    weight_rom_{layer_name} u_weights_rom (
        .addr(addr_cnt),
        .data(weights_rom_data_raw)
    );

    bias_rom_{layer_name} u_bias_rom (
        .data(bias_rom_data_raw)
    );

    genvar i;
    generate
        for (i = 0; i < NUM_INPUTS; i++) begin : INPUT_SPLIT
            assign weights_rom_data[i] = weights_rom_data_raw[i*Q_WIDTH +: Q_WIDTH];
        end
    endgenerate

    genvar k;
    generate
        for (k = 0; k < NUM_NEURONS; k++) begin : SPLIT_BIAS
            assign bias_rom_data[k] = bias_rom_data_raw[k*Q_WIDTH +: Q_WIDTH];
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            bias_rom_data_reg <= '0;
        else
            bias_rom_data_reg <= bias_rom_data[addr_cnt];
    end

    fc_out #(.NUM_MULT(NUM_INPUTS)) u_fc_out (
        .clk         (clk),
        .rst_n       (rst_n),
        .en          (valid_in_reg),
        .fc_in       (fc_in_reg),
        .weights     (weights_rom_data),
        .bias        (bias_rom_data_reg),
        .fc_out_valid(fc_out_valid),
        .fc_out      (fc_out_tmp)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int n = 0; n < NUM_NEURONS; n++)
                fc_out[n] <= '0;
            fc_out_cnt <= '0;
            valid_out  <= 1'b0;
        end
        else begin
            valid_out <= 1'b0;
            if (fc_out_valid) begin
                for (int n = 0; n < NUM_NEURONS-1; n++)
                    fc_out[n] <= fc_out[n+1];
                fc_out[NUM_NEURONS-1] <= fc_out_tmp;
                if (fc_out_cnt == NUM_NEURONS-1) begin
                    valid_out <= 1'b1;
                    fc_out_cnt <= '0;
                end
                else if (fc_out_cnt < NUM_NEURONS)
                    fc_out_cnt <= fc_out_cnt + 1'b1;
            end
        end
    end

endmodule
"""
    out_path = out_dir / f"fc_out_layer_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def write_embedded_rtl_templates(sv_dir: Path, weight_width: int, *, write_submodules: bool = True, has_softmax: bool = False) -> None:
    """Write embedded quant_pkg and optionally mac, fc_in, fc_out, relu_layer, softmax_layer to sv_dir."""
    sv_dir.mkdir(parents=True, exist_ok=True)
    
    (sv_dir / "quant_pkg.sv").write_text(_get_quant_pkg_content(weight_width), encoding="utf-8")
    if write_submodules:
        (sv_dir / "mac.sv").write_text(EMBEDDED_MAC_SV, encoding="utf-8")
        (sv_dir / "fc_in.sv").write_text(EMBEDDED_FC_IN_SV, encoding="utf-8")
        (sv_dir / "fc_out.sv").write_text(EMBEDDED_FC_OUT_SV, encoding="utf-8")
        (sv_dir / "relu_layer.sv").write_text(EMBEDDED_RELU_LAYER_SV, encoding="utf-8")
    if has_softmax:
        (sv_dir / "softmax_layer.sv").write_text(_get_softmax_layer_sv(), encoding="utf-8")
    if write_submodules:
        if has_softmax:
            LOGGER.info("Wrote embedded RTL templates (quant_pkg, mac, fc_in, fc_out, relu_layer, softmax_layer)")
        else:
            LOGGER.info("Wrote embedded RTL templates (quant_pkg, mac, fc_in, fc_out, relu_layer)")
    elif has_softmax:
        LOGGER.info("Wrote embedded RTL templates (quant_pkg, softmax_layer)")
    else:
        LOGGER.info("Wrote quant_pkg.sv only (flattened mode)")


# ============================================================================
# SystemVerilog Generation
# ============================================================================

def generate_weight_rom(
    layer_name: str,
    in_features: int,
    num_neurons: int,
    weight_width: int,
    out_dir: Path
) -> Path:
    """
    Generate weight_rom_<layer>.sv module.
    Uses quant_pkg format (same as Desktop/rtl):
    - fc1: DEPTH=in_features, each line = packed num_neurons weights (by input)
    - fc2/fc3: DEPTH=num_neurons, each line = packed in_features weights (by neuron)
    """
    mem_file = f"src/rtl/systemverilog/mem/{layer_name}_weights_packed.mem"
    if layer_name == "fc1":
        depth = in_features
        packed_width = num_neurons * weight_width
    else:
        depth = num_neurons
        packed_width = in_features * weight_width

    content = f"""`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: weight_rom_{layer_name}
// Description: Weight ROM for {layer_name} (quant_pkg format, packed per line)
//   - Depth: {depth}, each line = one packed word
//------------------------------------------------------------------------------
module weight_rom_{layer_name} #(
    parameter int ADDR_WIDTH = $clog2({depth}),
    parameter int DATA_WIDTH = {packed_width}
)(
    input  logic [ADDR_WIDTH-1:0] addr,
    output logic [DATA_WIDTH-1:0] data
);

    logic [DATA_WIDTH-1:0] mem [0:{depth}-1];

    initial begin
        $readmemh("{mem_file}", mem);
    end

    assign data = mem[addr];

endmodule
"""
    # Here, out_dir is expected to be the SystemVerilog RTL directory
    out_path = out_dir / f"weight_rom_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_bias_rom(
    layer_name: str,
    num_neurons: int,
    weight_width: int,
    out_dir: Path
) -> Path:
    """
    Generate bias_rom_<layer>.sv module.
    Uses quant_pkg format: single packed line (neuron 0 in LSBs), same as Desktop/rtl.
    """
    mem_file = f"src/rtl/systemverilog/mem/{layer_name}_biases.mem"
    packed_width = num_neurons * weight_width
    
    content = f"""`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: bias_rom_{layer_name}
// Description: Bias ROM (quant_pkg format - single packed line)
//------------------------------------------------------------------------------
module bias_rom_{layer_name} #(
    parameter int DATA_WIDTH = {packed_width}
)(
    output logic [DATA_WIDTH-1:0] data
);

    logic [DATA_WIDTH-1:0] mem [0:0];

    initial begin
        $readmemh("{mem_file}", mem);
    end

    assign data = mem[0];

endmodule
"""
    # Here, out_dir is expected to be the SystemVerilog RTL directory
    out_path = out_dir / f"bias_rom_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_fc_layer_wrapper(
    layer_name: str,
    in_features: int,
    num_neurons: int,
    data_width: int,
    weight_width: int,
    out_dir: Path
) -> Path:
    """Generate fc_layer_<layer>.sv wrapper module (uses quant_pkg fc_in, outputs q_data_t)."""
    content = f"""`timescale 1ns/1ps
import quant_pkg::*;

//------------------------------------------------------------------------------
// Module: fc_layer_{layer_name}
// Description: Wrapper for {layer_name} FC layer with ROMs (rtl-style, q_data_t)
//------------------------------------------------------------------------------
module fc_layer_{layer_name} #(
    parameter int NUM_NEURONS   = {num_neurons},
    parameter int INPUT_SIZE    = {in_features}
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid_in,
    input  q_data_t     input_data,
    output q_data_t     fc_out [NUM_NEURONS],
    output logic        valid_out
);

    logic [$clog2(INPUT_SIZE)-1:0] weight_addr;
    logic [NUM_NEURONS*Q_WIDTH-1:0] weight_row;
    logic [NUM_NEURONS*Q_WIDTH-1:0] bias_row;

    q_data_t weights [NUM_NEURONS];
    q_data_t bias    [NUM_NEURONS];

    genvar i;

    weight_rom_{layer_name} u_weight_rom (
        .addr(weight_addr),
        .data(weight_row)
    );

    bias_rom_{layer_name} u_bias_rom (
        .data(bias_row)
    );

    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : WEIGHT_SLICE
            assign weights[i] = weight_row[i*Q_WIDTH +: Q_WIDTH];
        end
    endgenerate

    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : BIAS_SLICE
            assign bias[i] = bias_row[i*Q_WIDTH +: Q_WIDTH];
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n)
            weight_addr <= '0;
        else if (valid_in) begin
            if (weight_addr == INPUT_SIZE-1)
                weight_addr <= '0;
            else
                weight_addr <= weight_addr + 1'b1;
        end
    end

    fc_in #(
        .NUM_NEURONS  (NUM_NEURONS),
        .INPUT_SIZE   (INPUT_SIZE)
    ) u_fc_in (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (valid_in),
        .data_in  (input_data),
        .weights  (weights),
        .bias     (bias),
        .fc_out   (fc_out),
        .out_valid(valid_out)
    );

endmodule
"""
    # Here, out_dir is expected to be the SystemVerilog RTL directory
    out_path = out_dir / f"fc_layer_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_top_module(
    model_name: str,
    layers: List[LayerInfo],
    input_size: int,
    data_width: int,
    weight_width: int,
    out_dir: Path,
    *,
    has_softmax: bool = False,
) -> Path:
    """Generate top module with streaming control logic (rtl-style, quant_pkg).
    When has_softmax=True, inserts softmax_layer after last FC and outputs prob_t."""
    
    active_layers = [l for l in layers if l.layer_type != "flatten"]
    
    port_decls = [
        "    // Input interface",
        "    input  logic        clk,",
        "    input  logic        rst_n,",
        "    input  logic        in_valid,",
        "    input  q_data_t     in_data,",
    ]
    
    signal_decls = []
    instantiations = []
    connections = []
    
    stream_valid = "in_valid"
    stream_data = "in_data"
    stream_vector = None  # For fc2/fc3 (fc_out_layer) - full vector
    stream_vector_valid = None
    prev_fc_out = None
    prev_fc_valid = None
    
    for i, layer in enumerate(active_layers):
        if layer.layer_type == "linear":
            fc_name = layer.name
            num_neurons = layer.out_features
            in_features = layer.in_features
            
            fc_valid_out = f"{fc_name}_valid_out"
            fc_out = f"{fc_name}_out"
            
            signal_decls.extend([
                f"    // {fc_name} layer signals",
                f"    logic      {fc_valid_out};",
                f"    q_data_t   {fc_out} [{num_neurons}];",
                ""
            ])
            
            if fc_name == "fc1":
                # fc1: fc_layer (fc_in style) - serial input
                fc_valid_in = f"{fc_name}_valid_in"
                fc_data_in = f"{fc_name}_data_in"
                signal_decls.extend([
                    f"    logic      {fc_valid_in};",
                    f"    q_data_t   {fc_data_in};",
                    ""
                ])
                connections.append(f"    assign {fc_valid_in} = {stream_valid};")
                connections.append(f"    assign {fc_data_in} = {stream_data};")
                connections.append("")
                instantiations.extend([
                    f"    // {fc_name} layer (fc_in style, quant_pkg format)",
                    f"    fc_layer_{fc_name} #(",
                    f"        .NUM_NEURONS  ({num_neurons}),",
                    f"        .INPUT_SIZE   ({in_features})",
                    f"    ) u_{fc_name} (",
                    f"        .clk       (clk),",
                    f"        .rst_n     (rst_n),",
                    f"        .valid_in  ({fc_valid_in}),",
                    f"        .input_data({fc_data_in}),",
                    f"        .fc_out    ({fc_out}),",
                    f"        .valid_out ({fc_valid_out})",
                    f"    );",
                    ""
                ])
            else:
                # fc2/fc3: fc_out_layer - full vector input, quant_pkg format
                connections.append(f"    // fc_out_layer_{fc_name} uses {stream_vector} and {stream_vector_valid}")
                connections.append("")
                instantiations.extend([
                    f"    // {fc_name} layer (fc_out style, quant_pkg format)",
                    f"    fc_out_layer_{fc_name} #(",
                    f"        .NUM_NEURONS ({num_neurons}),",
                    f"        .NUM_INPUTS  ({in_features})",
                    f"    ) u_{fc_name} (",
                    f"        .clk      (clk),",
                    f"        .rst_n    (rst_n),",
                    f"        .valid_in ({stream_vector_valid}),",
                    f"        .fc_in    ({stream_vector}),",
                    f"        .fc_out   ({fc_out}),",
                    f"        .valid_out({fc_valid_out})",
                    f"    );",
                    ""
                ])
            
            has_relu = (i + 1 < len(active_layers) and 
                       active_layers[i + 1].layer_type == "relu")
            
            if has_relu:
                prev_fc_out = fc_out
                prev_fc_valid = fc_valid_out
            else:
                prev_fc_out = fc_out
                prev_fc_valid = fc_valid_out
                break
                
        elif layer.layer_type == "relu":
            if prev_fc_out is None or prev_fc_valid is None:
                raise RuntimeError(f"ReLU {layer.name} has no preceding FC layer output")
            
            prev_fc = None
            for j in range(i-1, -1, -1):
                if active_layers[j].layer_type == "linear":
                    prev_fc = active_layers[j]
                    break
            
            if prev_fc is None:
                raise RuntimeError(f"ReLU {layer.name} has no preceding FC layer")
            
            num_neurons = prev_fc.out_features
            relu_name = layer.name
            relu_in_vector = prev_fc_out
            relu_in_valid = prev_fc_valid
            relu_out_vector = f"{relu_name}_out"
            relu_out_valid = f"{relu_name}_out_valid"
            
            signal_decls.extend([
                f"    // {relu_name} layer signals",
                f"    q_data_t   {relu_out_vector} [{num_neurons}];",
                f"    logic      {relu_out_valid};",
                ""
            ])
            
            instantiations.extend([
                f"    // {relu_name} - relu_layer with array ports",
                f"    relu_layer #(",
                f"        .NUM_NEURONS({num_neurons})",
                f"    ) u_{relu_name} (",
                f"        .clk       (clk),",
                f"        .rst_n     (rst_n),",
                f"        .valid_in  ({relu_in_valid}),",
                f"        .data_in   ({relu_in_vector}),",
                f"        .data_out  ({relu_out_vector}),",
                f"        .valid_out ({relu_out_valid})",
                f"    );",
                ""
            ])
            
            # Next FC is fc2 or fc3 (fc_out_layer) - pass full vector, no serialization
            stream_vector = relu_out_vector
            stream_vector_valid = relu_out_valid
    
    # Output port
    last_fc = None
    for layer in reversed(active_layers):
        if layer.layer_type == "linear":
            last_fc = layer
            break
    
    if last_fc is None:
        raise RuntimeError("No output FC layer found")
    
    output_size = last_fc.out_features
    last_fc_out = f"{last_fc.name}_out"
    last_fc_valid = f"{last_fc.name}_valid_out"

    if has_softmax:
        port_decls.extend([
            "    // Output interface (probabilities, 0.16 format)",
            "    output logic    out_valid,",
            f"    output prob_t   out_data [{output_size}]"
        ])
        signal_decls.extend([
            "    // softmax layer signals",
            f"    prob_t   softmax_probs [{output_size}];",
            "    logic    softmax_valid;",
            ""
        ])
        instantiations.extend([
            "    // softmax_layer: logits -> probabilities",
            f"    softmax_layer #(.NUM_CLASSES({output_size})) u_softmax (",
            "        .clk       (clk),",
            "        .rst_n     (rst_n),",
            f"        .valid_in  ({last_fc_valid}),",
            f"        .logits_in ({last_fc_out}),",
            "        .probs_out (softmax_probs),",
            "        .valid_out (softmax_valid)",
            "    );",
            ""
        ])
        connections.extend([
            "    // Output assignment (from softmax)",
            "    assign out_valid = softmax_valid;",
            "    assign out_data = softmax_probs;"
        ])
    else:
        port_decls.extend([
            "    // Output interface",
            "    output logic    out_valid,",
            f"    output q_data_t out_data [{output_size}]"
        ])
        connections.extend([
            "    // Output assignment",
            f"    assign out_valid = {last_fc_valid};",
            f"    assign out_data = {last_fc_out};"
        ])
    
    # Build complete module
    content = f"""`timescale 1ns/1ps
import quant_pkg::*;

//------------------------------------------------------------------------------
// Module: {model_name}_top
// Description: Top-level module for {model_name} neural network (rtl-style)
//   Uses quant_pkg, fc_in, relu_layer from embedded templates
//------------------------------------------------------------------------------
module {model_name}_top (
{chr(10).join(port_decls)}
);

{chr(10).join(signal_decls)}

{chr(10).join(connections)}

{chr(10).join(instantiations)}

endmodule
"""
    
    # Here, out_dir is expected to be the SystemVerilog RTL directory
    out_path = out_dir / f"{model_name}_top.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_flattened_top_module(
    model_name: str,
    layers: List[LayerInfo],
    input_size: int,
    data_width: int,
    weight_width: int,
    out_dir: Path,
    *,
    has_softmax: bool = False,
) -> Path:
    """
    Generate a single flattened top module with all logic inlined (no submodule instances).
    ROMs become inline memory arrays with $readmemh. Supports 3+ FC layers (fc1, fc2, ..., fcN).
    """
    active_layers = [l for l in layers if l.layer_type != "flatten"]
    linear_layers = [l for l in active_layers if l.layer_type == "linear"]
    if len(linear_layers) < 3:
        raise RuntimeError(
            f"Flattened generation requires at least 3 FC layers; found {len(linear_layers)}"
        )
    for i, fc in enumerate(linear_layers):
        if fc.name != f"fc{i + 1}":
            raise RuntimeError(f"Flattened expects fc1..fcN; got {fc.name}")

    num_layers = len(linear_layers)
    n_list = [fc.out_features or 0 for fc in linear_layers]
    in_list = [linear_layers[0].in_features or input_size] + [linear_layers[i].out_features or 0 for i in range(num_layers - 1)]

    mem_prefix = "src/rtl/systemverilog/mem"
    output_size = n_list[-1]

    parts: List[str] = []
    localparam_lines = []
    for i in range(num_layers):
        localparam_lines.append(f"    localparam FC{i+1}_NEURONS    = {n_list[i]};")
    for i in range(num_layers):
        localparam_lines.append(f"    localparam FC{i+1}_INPUT_SIZE = {in_list[i]};")

    out_type = "prob_t" if has_softmax else "q_data_t"
    parts.append(f'''`timescale 1ns/1ps
import quant_pkg::*;

//------------------------------------------------------------------------------
// Module: {{model_name}}_top
// Description: Flattened top-level (all logic inlined, no submodules)
//------------------------------------------------------------------------------
module {{model_name}}_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  q_data_t     in_data,
    output logic        out_valid,
    output {out_type}   out_data [{output_size}]
);

{chr(10).join(localparam_lines)}
''')

    parts.append(f'''
    // -------------------------------------------------------------------------
    // FC1 (fc_in style)
    // -------------------------------------------------------------------------
    logic [$clog2(FC1_INPUT_SIZE)-1:0] fc1_weight_addr;
    logic [FC1_NEURONS*Q_WIDTH-1:0] fc1_weight_row;
    logic [FC1_NEURONS*Q_WIDTH-1:0] fc1_bias_row;
    q_data_t fc1_weights [FC1_NEURONS];
    q_data_t fc1_bias    [FC1_NEURONS];
    acc_t fc1_mac_acc [FC1_NEURONS];
    acc_t fc1_mac_acc_tmp [FC1_NEURONS];
    acc_t fc1_acc_tmp [FC1_NEURONS];
    acc_t fc1_bias_32 [FC1_NEURONS];
    logic [$clog2(FC1_INPUT_SIZE+1)-1:0] fc1_out_count;
    logic [FC1_NEURONS-1:0] fc1_mac_out_valid;
    logic fc1_all_mac_valid;
    logic signed [63:0] fc1_mult_reg [FC1_NEURONS];
    logic signed [63:0] fc1_acc_reg [FC1_NEURONS];
    logic fc1_enable_q, fc1_enable_qq;
    q_data_t fc1_out [FC1_NEURONS];
    logic fc1_valid_out;

    (* rom_style = "block" *) logic [FC1_NEURONS*Q_WIDTH-1:0] fc1_mem_weight [0:FC1_INPUT_SIZE-1];
    (* rom_style = "block" *) logic [FC1_NEURONS*Q_WIDTH-1:0] fc1_mem_bias [0:0];

    initial begin
        $readmemh("{mem_prefix}/fc1_weights_packed.mem", fc1_mem_weight);
        $readmemh("{mem_prefix}/fc1_biases.mem", fc1_mem_bias);
    end

    genvar gi;
    generate
        for (gi = 0; gi < FC1_NEURONS; gi++) begin : FC1_WEIGHT_SLICE
            assign fc1_weights[gi] = fc1_weight_row[gi*Q_WIDTH +: Q_WIDTH];
            assign fc1_bias[gi] = fc1_bias_row[gi*Q_WIDTH +: Q_WIDTH];
        end
    endgenerate
    assign fc1_weight_row = fc1_mem_weight[fc1_weight_addr];
    assign fc1_bias_row = fc1_mem_bias[0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fc1_weight_addr <= '0;
        else if (in_valid)
            fc1_weight_addr <= (fc1_weight_addr == FC1_INPUT_SIZE-1) ? '0 : fc1_weight_addr + 1'b1;
    end

    generate
        for (gi = 0; gi < FC1_NEURONS; gi++) begin : FC1_MAC
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) fc1_mult_reg[gi] <= '0;
                else if (in_valid) fc1_mult_reg[gi] <= $signed(in_data) * $signed(fc1_weights[gi]);
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) fc1_acc_reg[gi] <= '0;
                else if (fc1_valid_out) fc1_acc_reg[gi] <= '0;
                else if (in_valid) fc1_acc_reg[gi] <= fc1_acc_reg[gi] + fc1_mult_reg[gi];
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin fc1_enable_q <= 1'b0; fc1_enable_qq <= 1'b0; end
        else if (fc1_valid_out) begin fc1_enable_q <= 1'b0; fc1_enable_qq <= 1'b0; end
        else begin fc1_enable_q <= in_valid; fc1_enable_qq <= fc1_enable_q; end
    end

    generate for (gi = 0; gi < FC1_NEURONS; gi++) begin : FC1_MAC_VALID
        assign fc1_mac_out_valid[gi] = fc1_enable_qq;
    end endgenerate
    assign fc1_all_mac_valid = &fc1_mac_out_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin fc1_out_count <= '0; fc1_valid_out <= 1'b0; end
        else begin
            fc1_valid_out <= 1'b0;
            if (fc1_all_mac_valid) begin
                if (fc1_out_count == FC1_INPUT_SIZE-1) begin
                    fc1_out_count <= '0;
                    fc1_valid_out <= 1'b1;
                end else fc1_out_count <= fc1_out_count + 1'b1;
            end
        end
    end

    generate
        for (gi = 0; gi < FC1_NEURONS; gi++) begin : FC1_BIAS_ADD
            assign fc1_mac_acc_tmp[gi] = ((fc1_acc_reg[gi] > $signed(32'h7FFFFFFF)) ? 32'sh7FFFFFFF :
                ((fc1_acc_reg[gi] < $signed(32'h80000000)) ? 32'sh80000000 : fc1_acc_reg[gi][31:0])) >>> FRAC_WIDTH;
            assign fc1_bias_32[gi] = {{ {{64-Q_WIDTH{{fc1_bias[gi][Q_WIDTH-1]}}}}, fc1_bias[gi] }};
            assign fc1_acc_tmp[gi] = fc1_mac_acc_tmp[gi] + fc1_bias_32[gi];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) fc1_out[gi] <= '0;
                else if (fc1_out_count == FC1_INPUT_SIZE-1) begin
                    if (fc1_acc_tmp[gi] > SATURATION_H) fc1_out[gi] <= LIMIT_H;
                    else if (fc1_acc_tmp[gi] < SATURATION_L) fc1_out[gi] <= $signed(LIMIT_L);
                    else fc1_out[gi] <= fc1_acc_tmp[gi][Q_WIDTH:0];
                end
            end
        end
    endgenerate
''')

    parts.append(f'''
    // -------------------------------------------------------------------------
    // ReLU 1
    // -------------------------------------------------------------------------
    q_data_t relu1_out [FC1_NEURONS];
    logic relu1_valid_out;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) relu1_valid_out <= 1'b0;
        else relu1_valid_out <= fc1_valid_out;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) for (int i=0; i<FC1_NEURONS; i++) relu1_out[i] <= '0;
        else if (fc1_valid_out)
            for (int i=0; i<FC1_NEURONS; i++)
                relu1_out[i] <= (fc1_out[i] < 0) ? '0 : fc1_out[i];
    end
''')

    # FC2..FCN (fc_out style) + ReLU blocks
    for idx in range(1, num_layers):
        i = idx + 1  # 2..N
        n_neurons = n_list[idx]
        n_inputs = in_list[idx]
        prev_relu = f"relu{idx}_out" if idx > 1 else "relu1_out"
        prev_valid = f"relu{idx}_valid_out" if idx > 1 else "relu1_valid_out"
        is_last = idx == num_layers - 1

        fc_block = f'''
    // -------------------------------------------------------------------------
    // FC{i} (fc_out style)
    // -------------------------------------------------------------------------
    logic [$clog2(FC{i}_NEURONS)-1:0] fc{i}_addr_cnt;
    logic fc{i}_valid_in_reg;
    q_data_t fc{i}_fc_in_reg [FC{i}_INPUT_SIZE];
    logic [FC{i}_INPUT_SIZE*Q_WIDTH-1:0] fc{i}_weights_raw;
    q_data_t fc{i}_weights [FC{i}_INPUT_SIZE];
    logic [FC{i}_NEURONS*Q_WIDTH-1:0] fc{i}_bias_raw;
    q_data_t fc{i}_bias [FC{i}_NEURONS];
    q_data_t fc{i}_bias_reg;
    logic signed [63:0] fc{i}_mult_res [FC{i}_INPUT_SIZE];
    logic signed [63:0] fc{i}_mult_tmp [FC{i}_INPUT_SIZE];
    logic signed [63:0] fc{i}_sum_comb;
    logic fc{i}_out_valid;
    q_data_t fc{i}_out_tmp;
    q_data_t fc{i}_out [FC{i}_NEURONS];
    logic [$clog2(FC{i}_NEURONS+1)-1:0] fc{i}_out_cnt;
    logic fc{i}_valid_out;

    (* rom_style = "block" *) logic [FC{i}_INPUT_SIZE*Q_WIDTH-1:0] fc{i}_mem_weight [0:FC{i}_NEURONS-1];
    (* rom_style = "block" *) logic [FC{i}_NEURONS*Q_WIDTH-1:0] fc{i}_mem_bias [0:0];

    initial begin
        $readmemh("{mem_prefix}/fc{i}_weights_packed.mem", fc{i}_mem_weight);
        $readmemh("{mem_prefix}/fc{i}_biases.mem", fc{i}_mem_bias);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fc{i}_addr_cnt <= '0;
        else if (fc{i}_addr_cnt == FC{i}_NEURONS-1) fc{i}_addr_cnt <= '0;
        else if ({prev_valid} || fc{i}_addr_cnt > 0) fc{i}_addr_cnt <= fc{i}_addr_cnt + 1'b1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc{i}_valid_in_reg <= 1'b0;
            for (int j=0; j<FC{i}_INPUT_SIZE; j++) fc{i}_fc_in_reg[j] <= '0;
        end else begin
            if (fc{i}_addr_cnt == '0) begin
                fc{i}_valid_in_reg <= {prev_valid};
                fc{i}_fc_in_reg <= {prev_relu};
            end else if (fc{i}_addr_cnt == FC{i}_NEURONS) begin
                fc{i}_valid_in_reg <= 1'b0;
                for (int j=0; j<FC{i}_INPUT_SIZE; j++) fc{i}_fc_in_reg[j] <= '0;
            end
        end
    end

    assign fc{i}_weights_raw = fc{i}_mem_weight[fc{i}_addr_cnt];
    assign fc{i}_bias_raw = fc{i}_mem_bias[0];
    generate for (gi=0; gi<FC{i}_INPUT_SIZE; gi++) assign fc{i}_weights[gi] = fc{i}_weights_raw[gi*Q_WIDTH +: Q_WIDTH]; endgenerate
    generate for (gi=0; gi<FC{i}_NEURONS; gi++) assign fc{i}_bias[gi] = fc{i}_bias_raw[gi*Q_WIDTH +: Q_WIDTH]; endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fc{i}_bias_reg <= '0;
        else fc{i}_bias_reg <= fc{i}_bias[fc{i}_addr_cnt];
    end

    generate for (gi=0; gi<FC{i}_INPUT_SIZE; gi++) begin : FC{i}_MULT
        assign fc{i}_mult_res[gi] = $signed(fc{i}_fc_in_reg[gi]) * $signed(fc{i}_weights[gi]);
        assign fc{i}_mult_tmp[gi] = (fc{i}_mult_res[gi] >>> FRAC_WIDTH);
    end endgenerate

    integer fc{i}_k;
    always_comb begin
        fc{i}_sum_comb = {{ {{64-Q_WIDTH{{fc{i}_bias_reg[Q_WIDTH-1]}}}}, fc{i}_bias_reg }};
        for (fc{i}_k=0; fc{i}_k<FC{i}_INPUT_SIZE; fc{i}_k++) fc{i}_sum_comb = fc{i}_sum_comb + fc{i}_mult_tmp[fc{i}_k];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin fc{i}_out_tmp <= '0; fc{i}_out_valid <= 1'b0; end
        else if (fc{i}_valid_in_reg) begin
            fc{i}_out_valid <= 1'b1;
            if (fc{i}_sum_comb > SATURATION_H) fc{i}_out_tmp <= LIMIT_H;
            else if (fc{i}_sum_comb < SATURATION_L) fc{i}_out_tmp <= $signed(LIMIT_L);
            else fc{i}_out_tmp <= fc{i}_sum_comb[Q_WIDTH:0];
        end else fc{i}_out_valid <= 1'b0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int n=0; n<FC{i}_NEURONS; n++) fc{i}_out[n] <= '0;
            fc{i}_out_cnt <= '0; fc{i}_valid_out <= 1'b0;
        end else begin
            fc{i}_valid_out <= 1'b0;
            if (fc{i}_out_valid) begin
                for (int n=0; n<FC{i}_NEURONS-1; n++) fc{i}_out[n] <= fc{i}_out[n+1];
                fc{i}_out[FC{i}_NEURONS-1] <= fc{i}_out_tmp;
                if (fc{i}_out_cnt == FC{i}_NEURONS-1) begin fc{i}_valid_out <= 1'b1; fc{i}_out_cnt <= '0; end
                else if (fc{i}_out_cnt < FC{i}_NEURONS) fc{i}_out_cnt <= fc{i}_out_cnt + 1'b1;
            end
        end
    end
'''
        parts.append(fc_block)

        if is_last:
            if has_softmax:
                parts.append(f'''
    // -------------------------------------------------------------------------
    // Softmax (logits -> probabilities, 0.16 format)
    // -------------------------------------------------------------------------
    prob_t softmax_probs [FC{i}_NEURONS];
    logic softmax_valid_out;
    softmax_layer #(.NUM_CLASSES(FC{i}_NEURONS)) u_softmax (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (fc{i}_valid_out),
        .logits_in (fc{i}_out),
        .probs_out (softmax_probs),
        .valid_out (softmax_valid_out)
    );
    assign out_valid = softmax_valid_out;
    assign out_data = softmax_probs;
''')
            else:
                parts.append(f'''
    assign out_valid = fc{i}_valid_out;
    assign out_data = fc{i}_out;
''')
        else:
            parts.append(f'''
    // -------------------------------------------------------------------------
    // ReLU {i}
    // -------------------------------------------------------------------------
    q_data_t relu{i}_out [FC{i}_NEURONS];
    logic relu{i}_valid_out;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) relu{i}_valid_out <= 1'b0;
        else relu{i}_valid_out <= fc{i}_valid_out;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) for (int j=0; j<FC{i}_NEURONS; j++) relu{i}_out[j] <= '0;
        else if (fc{i}_valid_out)
            for (int j=0; j<FC{i}_NEURONS; j++)
                relu{i}_out[j] <= (fc{i}_out[j] < 0) ? '0 : fc{i}_out[j];
    end
''')

    parts.append("endmodule\n")
    content = "\n".join(parts).replace("{model_name}", model_name)
    out_path = out_dir / f"{model_name}_top.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_wrapper_module(
    model_name: str,
    layers: List[LayerInfo],
    weight_width: int,
    out_dir: Path,
    *,
    has_softmax: bool = False,
) -> Path:
    """
    Generate a thin wrapper module that instantiates the top module (like binaryclass_NN_wrapper).
    When has_softmax=True, output type is prob_t (probabilities).
    """
    last_fc = next((l for l in reversed(layers) if l.layer_type == "linear"), None)
    if not last_fc:
        raise RuntimeError("No linear layer for wrapper")
    output_size = last_fc.out_features or 0
    out_type = "prob_t" if has_softmax else "q_data_t"

    content = f'''`timescale 1ns/1ps
import quant_pkg::*;

//------------------------------------------------------------------------------
// Module: {{model_name}}_wrapper
// Description: Wrapper that instantiates {{model_name}}_top
//------------------------------------------------------------------------------
module {{model_name}}_wrapper (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  q_data_t     in_data,
    output logic        out_valid,
    output {out_type}   out_data [{output_size}]
);

    {{model_name}}_top u_top (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (in_valid),
        .in_data  (in_data),
        .out_valid(out_valid),
        .out_data (out_data)
    );

endmodule
'''
    content = content.replace("{model_name}", model_name)
    out_path = out_dir / f"{model_name}_wrapper.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_testbench(
    model_name: str,
    input_size: int,
    output_size: int,
    weight_width: int,
    out_dir: Path
) -> Path:
    """Generate simple testbench (uses quant_pkg q_data_t)."""
    content = f"""`timescale 1ns/1ps
`define Q_INT{weight_width}
import quant_pkg::*;

//------------------------------------------------------------------------------
// Testbench: tb_{model_name}_top
// Description: Simple testbench for {model_name}_top
//------------------------------------------------------------------------------
module tb_{model_name}_top;

    parameter int INPUT_SIZE = {input_size};
    parameter int OUTPUT_SIZE = {output_size};

    logic clk;
    logic rst_n;
    logic in_valid;
    q_data_t in_data;
    logic out_valid;
    q_data_t out_data [OUTPUT_SIZE];

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Reset generation
    initial begin
        rst_n = 0;
        #100;
        rst_n = 1;
    end

    // DUT instantiation
    {model_name}_top u_dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (in_valid),
        .in_data  (in_data),
        .out_valid(out_valid),
        .out_data (out_data)
    );

    // Test stimulus
    initial begin
        in_valid = 0;
        in_data = 0;
        
        wait(rst_n);
        #20;
        
        // Feed dummy input vector
        $display("Starting input stream...");
        for (int i = 0; i < INPUT_SIZE; i++) begin
            @(posedge clk);
            in_valid = 1;
            in_data = $signed(i);  // Simple test pattern
        end
        
        @(posedge clk);
        in_valid = 0;
        
        // Wait for output
        wait(out_valid);
        @(posedge clk);
        
        $display("Output received:");
        for (int i = 0; i < OUTPUT_SIZE; i++) begin
            $display("  out_data[%0d] = %0d", i, out_data[i]);
        end
        
        #100;
        $finish;
    end

    // Monitor
    always @(posedge clk) begin
        if (out_valid) begin
            $display("[%0t] out_valid asserted", $time);
        end
    end

endmodule
"""
    # Here, out_dir is expected to be the testbench directory (e.g. tb/sim)
    out_path = out_dir / f"tb_{model_name}_top.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


# ============================================================================
# Report Generation
# ============================================================================

def generate_mapping_report(
    out_dir: Path,
    model_name: str,
    layers: List[LayerInfo],
    scale: int,
    data_width: int,
    weight_width: int,
    acc_width: int,
    frac_bits: int
) -> Path:
    """Generate mapping_report.txt."""
    lines = []
    lines.append("RTL Mapping Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Model: {model_name}")
    lines.append(f"Scale Factor: {scale}")
    lines.append(f"Data Width: {data_width}")
    lines.append(f"Weight Width: {weight_width}")
    lines.append(f"Accumulator Width: {acc_width}")
    lines.append(f"Fractional Bits: {frac_bits}")
    lines.append("")
    lines.append("Layer Mapping:")
    lines.append("-" * 80)
    
    for i, layer in enumerate(layers):
        line = f"{i+1:2d}. {layer.name:20s} {layer.layer_type:10s}"
        if layer.layer_type == "linear":
            line += f" in_features={layer.in_features:4d} out_features={layer.out_features:4d}"
            lines.append(line)
            lines.append(f"     Weights: SIM/{layer.name}_weights_packed.mem")
            lines.append(f"     Biases:  SIM/{layer.name}_biases.mem")
        elif layer.layer_type == "relu":
            line += " (element-wise)"
            lines.append(line)
        elif layer.layer_type == "flatten":
            line += " (pass-through in streaming)"
            lines.append(line)
        else:
            lines.append(line)
        lines.append("")
    
    lines.append("Generated Files (my_ip-style layout):")
    lines.append("-" * 80)
    lines.append("RTL Modules (under src/rtl/systemverilog):")
    lines.append("  - src/rtl/systemverilog/quant_pkg.sv")
    lines.append("  - src/rtl/systemverilog/mac.sv")
    lines.append("  - src/rtl/systemverilog/fc_in.sv")
    lines.append("  - src/rtl/systemverilog/fc_out.sv")
    lines.append("  - src/rtl/systemverilog/relu_layer.sv")
    
    for layer in layers:
        if layer.layer_type == "linear":
            if layer.name == "fc1":
                lines.append(f"  - src/rtl/systemverilog/fc_layer_{layer.name}.sv")
            else:
                lines.append(f"  - src/rtl/systemverilog/fc_out_layer_{layer.name}.sv")
            lines.append(f"  - src/rtl/systemverilog/weight_rom_{layer.name}.sv")
            lines.append(f"  - src/rtl/systemverilog/bias_rom_{layer.name}.sv")
    
    lines.append(f"  - src/rtl/systemverilog/{model_name}_top.sv")
    lines.append("")
    lines.append("Memory Files (quant_pkg format, same as Desktop/rtl):")
    for layer in layers:
        if layer.layer_type == "linear":
            lines.append(f"  - src/rtl/systemverilog/mem/{layer.name}_weights_packed.mem")
            lines.append(f"  - src/rtl/systemverilog/mem/{layer.name}_biases.mem")
    lines.append("")
    out_path = out_dir / "mapping_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_netlist_json(
    out_dir: Path,
    model_name: str,
    layers: List[LayerInfo]
) -> Path:
    """Generate netlist.json."""
    netlist = {
        "model_name": model_name,
        "layers": []
    }
    
    for layer in layers:
        layer_dict = {
            "name": layer.name,
            "type": layer.layer_type
        }
        if layer.layer_type == "linear":
            layer_dict["in_features"] = layer.in_features
            layer_dict["out_features"] = layer.out_features
            layer_dict["weight_mem"] = f"src/rtl/systemverilog/mem/{layer.name}_weights_packed.mem"
            layer_dict["bias_mem"] = f"src/rtl/systemverilog/mem/{layer.name}_biases.mem"
        netlist["layers"].append(layer_dict)
    
    out_path = out_dir / "netlist.json"
    out_path.write_text(json.dumps(netlist, indent=2), encoding="utf-8")
    return out_path
