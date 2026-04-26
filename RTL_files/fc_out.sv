/****************************************************************************************
"PYRAMIDTECH CONFIDENTIAL

Copyright (c) 2026 PyramidTech LLC. All rights reserved.

This file contains proprietary and confidential information of PyramidTech LLC.
The information contained herein is unpublished and subject to trade secret
protection. No part of this file may be reproduced, modified, distributed,
transmitted, disclosed, or used in any form or by any means without the
prior written permission of PyramidTech LLC.

This material must be returned immediately upon request by PyramidTech LLC"
/****************************************************************************************
File name:      fc_out.sv
  
Description:    Fully-connected (FC) output layer module with a 3-stage pipeline: Stage 1 (Multiplication), Stage 2 (Addition/Scaling), and Stage 3 (Saturation/Output).
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

module fc_out 
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

`end_keywords
