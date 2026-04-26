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
File name:      fc_out_layer.sv
  
Description:    Fully-connected output layer with sequential ROM access.
                Reads weights and biases from ROM and feeds the fc_out compute block.
  
Author:         AA
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

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
    if (ROM_DEPTH == 50) begin : gen_rom_50
      fc_proj_out_weights_rom #(
        .DEPTH(ROM_DEPTH),
        .WIDTH(NUM_NEURONS * Q_WIDTH)
      ) u_weights_rom (
        .clk_i  (clk_i),
        .addr_i (weights_rom_addr_s),
        .data_o (weights_rom_data_raw_s)
      );

      fc_proj_out_bias_rom #(
        .DEPTH(ROM_DEPTH),
        .WIDTH(Q_WIDTH*4)
      ) u_bias_rom (
        .clk_i  (clk_i),
        .addr_i (bias_rom_addr_s),
        .data_o (bias_rom_data_s)
      );
    end 
    else if (ROM_DEPTH == 50) begin : gen_rom_50
      fc_2_proj_out_weights_rom #(
        .DEPTH(ROM_DEPTH),
        .WIDTH(NUM_NEURONS * Q_WIDTH)
      ) u_weights_rom (
        .clk_i  (clk_i),
        .addr_i (weights_rom_addr_s),
        .data_o (weights_rom_data_raw_s)
      );

      fc_2_proj_out_bias_rom #(
        .DEPTH(ROM_DEPTH),
        .WIDTH(Q_WIDTH*4)
      ) u_bias_rom (
        .clk_i  (clk_i),
        .addr_i (bias_rom_addr_s),
        .data_o (bias_rom_data_s)
      );
    end 
    else if (ROM_DEPTH == 25) begin : gen_rom_25
      fc_3_proj_out_weights_rom #(
        .DEPTH(ROM_DEPTH),
        .WIDTH(NUM_NEURONS * Q_WIDTH)
      ) u_weights_rom (
        .clk_i  (clk_i),
        .addr_i (weights_rom_addr_s),
        .data_o (weights_rom_data_raw_s)
      );

      fc_3_proj_out_bias_rom #(
        .DEPTH(ROM_DEPTH),
        .WIDTH(Q_WIDTH*4)
      ) u_bias_rom (
        .clk_i  (clk_i),
        .addr_i (bias_rom_addr_s),
        .data_o (bias_rom_data_s)
      );
    end 
    else if (ROM_DEPTH == 10) begin : gen_rom_10
      fc_4_proj_out_weights_rom #(
        .DEPTH(ROM_DEPTH),
        .WIDTH(NUM_NEURONS * Q_WIDTH)
      ) u_weights_rom (
        .clk_i  (clk_i),
        .addr_i (weights_rom_addr_s),
        .data_o (weights_rom_data_raw_s)
      );

      fc_4_proj_out_bias_rom #(
        .DEPTH(ROM_DEPTH),
        .WIDTH(Q_WIDTH*4)
      ) u_bias_rom (
        .clk_i  (clk_i),
        .addr_i (bias_rom_addr_s),
        .data_o (bias_rom_data_s)
      );
    end
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

`end_keywords