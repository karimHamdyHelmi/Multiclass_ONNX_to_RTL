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
File name:      fc_in_layer.sv
  
Description:    Fully-connected (FC) input layer with sequential ROM access.
                Reads weights and biases from ROM and feeds the compute block.
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

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
    if (INPUT_SIZE == 1432) begin : gen_rom_1432
      fc_proj_in_weights_rom #(
        .DEPTH(INPUT_SIZE),
        .WIDTH(NUM_NEURONS * Q_WIDTH)
      ) u_weights_rom (
        .clk_i  (clk_i),
        .addr_i (weight_addr_q),
        .data_o (weight_rom_row_s)
      );

      fc_proj_in_bias_rom #(
        .DEPTH(1),
        .WIDTH(NUM_NEURONS * Q_WIDTH * 4)
      ) u_bias_rom (
        .clk_i  (clk_i),
        .addr_i (1'b0),
        .data_o (bias_rom_row_s)
      );
    end 
    else if (INPUT_SIZE == 50) begin : gen_rom_50
      fc_2_proj_in_weights_rom #(
        .DEPTH(INPUT_SIZE),
        .WIDTH(NUM_NEURONS * Q_WIDTH)
      ) u_weights_rom (
        .clk_i  (clk_i),
        .addr_i (weight_addr_q),
        .data_o (weight_rom_row_s)
      );

      fc_2_proj_in_bias_rom #(
        .DEPTH(1),
        .WIDTH(NUM_NEURONS * Q_WIDTH * 4)
      ) u_bias_rom (
        .clk_i  (clk_i),
        .addr_i (1'b0),
        .data_o (bias_rom_row_s)
      );
    end 
    else if (INPUT_SIZE == 50) begin : gen_rom_50
      fc_3_proj_in_weights_rom #(
        .DEPTH(INPUT_SIZE),
        .WIDTH(NUM_NEURONS * Q_WIDTH)
      ) u_weights_rom (
        .clk_i  (clk_i),
        .addr_i (weight_addr_q),
        .data_o (weight_rom_row_s)
      );

      fc_3_proj_in_bias_rom #(
        .DEPTH(1),
        .WIDTH(NUM_NEURONS * Q_WIDTH * 4)
      ) u_bias_rom (
        .clk_i  (clk_i),
        .addr_i (1'b0),
        .data_o (bias_rom_row_s)
      );
    end 
    else if (INPUT_SIZE == 25) begin : gen_rom_25
      fc_4_proj_in_weights_rom #(
        .DEPTH(INPUT_SIZE),
        .WIDTH(NUM_NEURONS * Q_WIDTH)
      ) u_weights_rom (
        .clk_i  (clk_i),
        .addr_i (weight_addr_q),
        .data_o (weight_rom_row_s)
      );

      fc_4_proj_in_bias_rom #(
        .DEPTH(1),
        .WIDTH(NUM_NEURONS * Q_WIDTH * 4)
      ) u_bias_rom (
        .clk_i  (clk_i),
        .addr_i (1'b0),
        .data_o (bias_rom_row_s)
      );
    end
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

`end_keywords