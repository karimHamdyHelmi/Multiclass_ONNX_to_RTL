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
File name:      mac.sv
  
Description:    Pipelined Multiply-Accumulate (MAC) unit with output saturation
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

module mac
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

`end_keywords
