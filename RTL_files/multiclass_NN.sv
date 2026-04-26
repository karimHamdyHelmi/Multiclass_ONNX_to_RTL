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
File name:      multiclass_NN.sv
  
Description:    Top-level Multiclass classifier (Conv + FC + softmax) with AXI4-Stream interfaces.
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

module multiclass_NN
  import quant_pkg::*;
#(
  parameter int CONV2D_PROJ_CONV_K_H        = 32'd3,
  parameter int CONV2D_PROJ_CONV_K_W        = 32'd3,
  parameter int CONV2D_PROJ_CONV_NUM_FIL    = 32'd2,
  parameter int CONV2D_PROJ_CONV_INPUTS_PC  = 32'd1,
  parameter int CONV2D_PROJ_CONV_LAYER_SCALE = 32'd8,
  parameter int CONV2D_PROJ_OUT_NUM_FIL    = 32'd32,
  parameter int CONV2D_PROJ_OUT_INPUTS_PC  = 32'd2,
  parameter int CONV2D_PROJ_OUT_LAYER_SCALE = 32'd5,
  parameter int CONV2D_1_PROJ_IN_NUM_FIL    = 32'd2,
  parameter int CONV2D_1_PROJ_IN_INPUTS_PC  = 32'd32,
  parameter int CONV2D_1_PROJ_IN_LAYER_SCALE = 32'd10,
  parameter int CONV2D_1_PROJ_CONV_K_H        = 32'd3,
  parameter int CONV2D_1_PROJ_CONV_K_W        = 32'd2,
  parameter int CONV2D_1_PROJ_CONV_NUM_FIL    = 32'd2,
  parameter int CONV2D_1_PROJ_CONV_INPUTS_PC  = 32'd2,
  parameter int CONV2D_1_PROJ_CONV_LAYER_SCALE = 32'd9,
  parameter int CONV2D_1_PROJ_OUT_NUM_FIL    = 32'd16,
  parameter int CONV2D_1_PROJ_OUT_INPUTS_PC  = 32'd2,
  parameter int CONV2D_1_PROJ_OUT_LAYER_SCALE = 32'd6,
  parameter int CONV2D_2_PROJ_IN_NUM_FIL    = 32'd2,
  parameter int CONV2D_2_PROJ_IN_INPUTS_PC  = 32'd16,
  parameter int CONV2D_2_PROJ_IN_LAYER_SCALE = 32'd8,
  parameter int CONV2D_2_PROJ_CONV_K_H        = 32'd3,
  parameter int CONV2D_2_PROJ_CONV_K_W        = 32'd2,
  parameter int CONV2D_2_PROJ_CONV_NUM_FIL    = 32'd2,
  parameter int CONV2D_2_PROJ_CONV_INPUTS_PC  = 32'd2,
  parameter int CONV2D_2_PROJ_CONV_LAYER_SCALE = 32'd9,
  parameter int CONV2D_2_PROJ_OUT_NUM_FIL    = 32'd8,
  parameter int CONV2D_2_PROJ_OUT_INPUTS_PC  = 32'd2,
  parameter int CONV2D_2_PROJ_OUT_LAYER_SCALE = 32'd6,
  parameter int POOL_KERNEL    = 32'd2,
  parameter int POOL_STRIDE    = 32'd4,
  parameter int POOL_FRAME_ROWS= 32'd714,
  parameter int POOL_CHANNELS  = 32'd8,
  parameter int POOL_OUT_ROWS  = 32'd179,
  parameter int FLATTEN_SIZE   = 32'd1432,
  parameter int FC_1_NEURONS         = 32'd4,
  parameter int FC_1_INPUT_SIZE      = 32'd1432,
  parameter int FC_1_ROM_DEPTH       = 32'd50,
  parameter int FC_1_IN_LAYER_SCALE  = 32'd8,
  parameter int FC_1_IN_BIAS_SCALE   = -32'sd1,
  parameter int FC_1_OUT_LAYER_SCALE = 32'd7,
  parameter int FC_1_OUT_BIAS_SCALE  = -32'sd1,
  parameter int FC_2_NEURONS         = 32'd4,
  parameter int FC_2_INPUT_SIZE      = 32'd50,
  parameter int FC_2_ROM_DEPTH       = 32'd50,
  parameter int FC_2_IN_LAYER_SCALE  = 32'd7,
  parameter int FC_2_IN_BIAS_SCALE   = 32'd0,
  parameter int FC_2_OUT_LAYER_SCALE = 32'd7,
  parameter int FC_2_OUT_BIAS_SCALE  = -32'sd1,
  parameter int FC_3_NEURONS         = 32'd3,
  parameter int FC_3_INPUT_SIZE      = 32'd50,
  parameter int FC_3_ROM_DEPTH       = 32'd25,
  parameter int FC_3_IN_LAYER_SCALE  = 32'd7,
  parameter int FC_3_IN_BIAS_SCALE   = -32'sd1,
  parameter int FC_3_OUT_LAYER_SCALE = 32'd7,
  parameter int FC_3_OUT_BIAS_SCALE  = -32'sd1,
  parameter int FC_4_NEURONS         = 32'd3,
  parameter int FC_4_INPUT_SIZE      = 32'd25,
  parameter int FC_4_ROM_DEPTH       = 32'd10,
  parameter int FC_4_IN_LAYER_SCALE  = 32'd7,
  parameter int FC_4_IN_BIAS_SCALE   = 32'd0,
  parameter int FC_4_OUT_LAYER_SCALE = 32'd7,
  parameter int FC_4_OUT_BIAS_SCALE  = 32'd0,
  parameter int NUM_CLASSES        = 32'd10,
  parameter int FIFO_DEPTH         = 32'd1024
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
  // ----- Conv chain signals -----
  logic signed [7:0] conv2d_proj_conv_out_s [0:2-1];
  logic              conv2d_proj_conv_valid_s;
  logic signed [7:0] conv2d_proj_out_out_s [0:32-1];
  logic              conv2d_proj_out_valid_s;
  logic signed [7:0] conv2d_1_proj_in_out_s [0:2-1];
  logic              conv2d_1_proj_in_valid_s;
  logic signed [7:0] conv2d_1_proj_conv_out_s [0:2-1];
  logic              conv2d_1_proj_conv_valid_s;
  logic signed [16-1:0] conv2d_1_proj_conv_data_in_s;
  logic signed [7:0] conv2d_1_proj_out_out_s [0:16-1];
  logic              conv2d_1_proj_out_valid_s;
  logic signed [7:0] conv2d_2_proj_in_out_s [0:2-1];
  logic              conv2d_2_proj_in_valid_s;
  logic signed [7:0] conv2d_2_proj_conv_out_s [0:2-1];
  logic              conv2d_2_proj_conv_valid_s;
  logic signed [16-1:0] conv2d_2_proj_conv_data_in_s;
  logic signed [7:0] conv2d_2_proj_out_out_s [0:8-1];
  logic              conv2d_2_proj_out_valid_s;
  // ----- AvgPool signals -----
  logic [7:0] pool_out_s [POOL_CHANNELS];
  logic       pool_valid_s;
  // ----- Flatten unit signals -----
  logic [7:0] flat_data_s;
  logic       flat_valid_s;
  logic       flat_tlast_s;
  // ----- FC chain signals -----
  q_data_t fc1_out_s [FC_1_NEURONS];
  logic    fc1_valid_s;
  q_data_t fc1_pre_relu_s;
  q_data_t fc1_post_relu_s;
  logic    relu1_in_s;
  logic    relu1_out_s;
  q_data_t fc2_out_s [FC_2_NEURONS];
  logic    fc2_valid_s;
  q_data_t fc2_pre_relu_s;
  q_data_t fc2_post_relu_s;
  logic    relu2_in_s;
  logic    relu2_out_s;
  q_data_t fc3_out_s [FC_3_NEURONS];
  logic    fc3_valid_s;
  q_data_t fc3_pre_relu_s;
  q_data_t fc3_post_relu_s;
  logic    relu3_in_s;
  logic    relu3_out_s;
  q_data_t fc4_out_s [FC_4_NEURONS];
  logic    fc4_valid_s;
  q_data_t logits_s;
  logic    logits_valid_s;
  // ----- Classifier head + AXI output signals -----
  q_data_t head_data_s;
  logic    head_valid_s;
  logic                  fifo_empty_s;
  logic                  fifo_full_s;
  logic                  fifo_write_en_s;
  logic                  fifo_read_en_s;
  logic                  fifo_read_en_q;
  logic [DATA_WIDTH-1:0] fifo_read_data_s;
  logic [DATA_WIDTH-1:0] fifo_write_data_s;
  logic [$clog2(NUM_CLASSES + 1) - 1:0] out_count_q;
  logic                  tvalid_q;

  // ============================================================
  // Conv chain
  // ============================================================
  // ----- Conv chain instantiations -----
  depthwise_conv_engine #(
    .DATA_WIDTH      (8),
    .K_H             (CONV2D_PROJ_CONV_K_H),
    .K_W             (CONV2D_PROJ_CONV_K_W),
    .NUM_FILTERS     (CONV2D_PROJ_CONV_NUM_FIL),
    .BIAS_WIDTH      (32),
    .LAYER_SCALE     (CONV2D_PROJ_CONV_LAYER_SCALE),
    .LAYER_INDEX     (1),
    .INPUTS_PER_CYCLE(CONV2D_PROJ_CONV_INPUTS_PC)
  ) u_conv2d_proj_conv (
    .clk_i   (clk_i),
    .rst_n_i (rst_n_i),
    .valid_in(s_axis_tvalid_i),
    .data_in (s_axis_tdata_i),
    .conv_out(conv2d_proj_conv_out_s),
    .valid_out(conv2d_proj_conv_valid_s)
  );
  pointwise_conv_engine #(
    .DATA_WIDTH      (8),
    .NUM_FILTERS     (CONV2D_PROJ_OUT_NUM_FIL),
    .INPUTS_PER_CYCLE(CONV2D_PROJ_OUT_INPUTS_PC),
    .BIAS_WIDTH      (32),
    .LAYER_INDEX     (1),
    .LAYER_SCALE     (CONV2D_PROJ_OUT_LAYER_SCALE)
  ) u_conv2d_proj_out (
    .clk_i   (clk_i),
    .rst_n_i (rst_n_i),
    .valid_in(conv2d_proj_conv_valid_s),
    .data_in (conv2d_proj_conv_out_s),
    .conv_out(conv2d_proj_out_out_s),
    .valid_out(conv2d_proj_out_valid_s)
  );
  pointwise_conv_engine #(
    .DATA_WIDTH      (8),
    .NUM_FILTERS     (CONV2D_1_PROJ_IN_NUM_FIL),
    .INPUTS_PER_CYCLE(CONV2D_1_PROJ_IN_INPUTS_PC),
    .BIAS_WIDTH      (32),
    .LAYER_INDEX     (2),
    .LAYER_SCALE     (CONV2D_1_PROJ_IN_LAYER_SCALE)
  ) u_conv2d_1_proj_in (
    .clk_i   (clk_i),
    .rst_n_i (rst_n_i),
    .valid_in(conv2d_proj_out_valid_s),
    .data_in (conv2d_proj_out_out_s),
    .conv_out(conv2d_1_proj_in_out_s),
    .valid_out(conv2d_1_proj_in_valid_s)
  );
  assign conv2d_1_proj_conv_data_in_s = {conv2d_1_proj_in_out_s[1] , conv2d_1_proj_in_out_s[0]};
  depthwise_conv_engine #(
    .DATA_WIDTH      (8),
    .K_H             (CONV2D_1_PROJ_CONV_K_H),
    .K_W             (CONV2D_1_PROJ_CONV_K_W),
    .NUM_FILTERS     (CONV2D_1_PROJ_CONV_NUM_FIL),
    .BIAS_WIDTH      (32),
    .LAYER_SCALE     (CONV2D_1_PROJ_CONV_LAYER_SCALE),
    .LAYER_INDEX     (2),
    .INPUTS_PER_CYCLE(CONV2D_1_PROJ_CONV_INPUTS_PC)
  ) u_conv2d_1_proj_conv (
    .clk_i   (clk_i),
    .rst_n_i (rst_n_i),
    .valid_in(conv2d_1_proj_in_valid_s),
    .data_in (conv2d_1_proj_conv_data_in_s),
    .conv_out(conv2d_1_proj_conv_out_s),
    .valid_out(conv2d_1_proj_conv_valid_s)
  );
  pointwise_conv_engine #(
    .DATA_WIDTH      (8),
    .NUM_FILTERS     (CONV2D_1_PROJ_OUT_NUM_FIL),
    .INPUTS_PER_CYCLE(CONV2D_1_PROJ_OUT_INPUTS_PC),
    .BIAS_WIDTH      (32),
    .LAYER_INDEX     (3),
    .LAYER_SCALE     (CONV2D_1_PROJ_OUT_LAYER_SCALE)
  ) u_conv2d_1_proj_out (
    .clk_i   (clk_i),
    .rst_n_i (rst_n_i),
    .valid_in(conv2d_1_proj_conv_valid_s),
    .data_in (conv2d_1_proj_conv_out_s),
    .conv_out(conv2d_1_proj_out_out_s),
    .valid_out(conv2d_1_proj_out_valid_s)
  );
  pointwise_conv_engine #(
    .DATA_WIDTH      (8),
    .NUM_FILTERS     (CONV2D_2_PROJ_IN_NUM_FIL),
    .INPUTS_PER_CYCLE(CONV2D_2_PROJ_IN_INPUTS_PC),
    .BIAS_WIDTH      (32),
    .LAYER_INDEX     (4),
    .LAYER_SCALE     (CONV2D_2_PROJ_IN_LAYER_SCALE)
  ) u_conv2d_2_proj_in (
    .clk_i   (clk_i),
    .rst_n_i (rst_n_i),
    .valid_in(conv2d_1_proj_out_valid_s),
    .data_in (conv2d_1_proj_out_out_s),
    .conv_out(conv2d_2_proj_in_out_s),
    .valid_out(conv2d_2_proj_in_valid_s)
  );
  assign conv2d_2_proj_conv_data_in_s = {conv2d_2_proj_in_out_s[1] , conv2d_2_proj_in_out_s[0]};
  depthwise_conv_engine #(
    .DATA_WIDTH      (8),
    .K_H             (CONV2D_2_PROJ_CONV_K_H),
    .K_W             (CONV2D_2_PROJ_CONV_K_W),
    .NUM_FILTERS     (CONV2D_2_PROJ_CONV_NUM_FIL),
    .BIAS_WIDTH      (32),
    .LAYER_SCALE     (CONV2D_2_PROJ_CONV_LAYER_SCALE),
    .LAYER_INDEX     (3),
    .INPUTS_PER_CYCLE(CONV2D_2_PROJ_CONV_INPUTS_PC)
  ) u_conv2d_2_proj_conv (
    .clk_i   (clk_i),
    .rst_n_i (rst_n_i),
    .valid_in(conv2d_2_proj_in_valid_s),
    .data_in (conv2d_2_proj_conv_data_in_s),
    .conv_out(conv2d_2_proj_conv_out_s),
    .valid_out(conv2d_2_proj_conv_valid_s)
  );
  pointwise_conv_engine #(
    .DATA_WIDTH      (8),
    .NUM_FILTERS     (CONV2D_2_PROJ_OUT_NUM_FIL),
    .INPUTS_PER_CYCLE(CONV2D_2_PROJ_OUT_INPUTS_PC),
    .BIAS_WIDTH      (32),
    .LAYER_INDEX     (5),
    .LAYER_SCALE     (CONV2D_2_PROJ_OUT_LAYER_SCALE)
  ) u_conv2d_2_proj_out (
    .clk_i   (clk_i),
    .rst_n_i (rst_n_i),
    .valid_in(conv2d_2_proj_conv_valid_s),
    .data_in (conv2d_2_proj_conv_out_s),
    .conv_out(conv2d_2_proj_out_out_s),
    .valid_out(conv2d_2_proj_out_valid_s)
  );
  // Average pooling along row dimension (POOL_KERNEL x 1, stride POOL_STRIDE)
  avg_pool_kx1 #(
    .DATA_W    (8),
    .CHANNELS  (POOL_CHANNELS),
    .FRAME_ROWS(POOL_FRAME_ROWS),
    .KERNEL    (POOL_KERNEL),
    .STRIDE    (POOL_STRIDE)
  ) u_avg_pool (
    .clk      (clk_i),
    .rst_n    (rst_n_i),
    .data_in  (conv2d_2_proj_out_out_s),
    .valid_in (conv2d_2_proj_out_valid_s),
    .data_out (pool_out_s),
    .valid_out(pool_valid_s)
  );
  flatten_unit #(
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
  );

  // ============================================================
  // FC chain
  // ============================================================
  // ----- FC chain instantiations -----
  fc_in_layer #(
    .NUM_NEURONS(FC_1_NEURONS),
    .INPUT_SIZE (FC_1_INPUT_SIZE),
    .BIAS_SCALE (FC_1_IN_BIAS_SCALE),
    .LAYER_SCALE(FC_1_IN_LAYER_SCALE)
  ) u_fc1_in_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(flat_valid_s),
    .data_i (flat_data_s),
    .data_o (fc1_out_s),
    .valid_o(fc1_valid_s)
  );
  fc_out_layer #(
    .NUM_NEURONS(FC_1_NEURONS),
    .ROM_DEPTH  (FC_1_ROM_DEPTH),
    .BIAS_SCALE (FC_1_OUT_BIAS_SCALE),
    .LAYER_SCALE(FC_1_OUT_LAYER_SCALE)
  ) u_fc1_out_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(fc1_valid_s),
    .data_i (fc1_out_s),
    .data_o (fc1_pre_relu_s),
    .valid_o(relu1_in_s)
  );
  relu_layer u_relu1 (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(relu1_in_s),
    .data_i (fc1_pre_relu_s),
    .data_o (fc1_post_relu_s),
    .valid_o(relu1_out_s)
  );
  fc_in_layer #(
    .NUM_NEURONS(FC_2_NEURONS),
    .INPUT_SIZE (FC_2_INPUT_SIZE),
    .BIAS_SCALE (FC_2_IN_BIAS_SCALE),
    .LAYER_SCALE(FC_2_IN_LAYER_SCALE)
  ) u_fc2_in_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(relu1_out_s),
    .data_i (fc1_post_relu_s),
    .data_o (fc2_out_s),
    .valid_o(fc2_valid_s)
  );
  fc_out_layer #(
    .NUM_NEURONS(FC_2_NEURONS),
    .ROM_DEPTH  (FC_2_ROM_DEPTH),
    .BIAS_SCALE (FC_2_OUT_BIAS_SCALE),
    .LAYER_SCALE(FC_2_OUT_LAYER_SCALE)
  ) u_fc2_out_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(fc2_valid_s),
    .data_i (fc2_out_s),
    .data_o (fc2_pre_relu_s),
    .valid_o(relu2_in_s)
  );
  relu_layer u_relu2 (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(relu2_in_s),
    .data_i (fc2_pre_relu_s),
    .data_o (fc2_post_relu_s),
    .valid_o(relu2_out_s)
  );
  fc_in_layer #(
    .NUM_NEURONS(FC_3_NEURONS),
    .INPUT_SIZE (FC_3_INPUT_SIZE),
    .BIAS_SCALE (FC_3_IN_BIAS_SCALE),
    .LAYER_SCALE(FC_3_IN_LAYER_SCALE)
  ) u_fc3_in_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(relu2_out_s),
    .data_i (fc2_post_relu_s),
    .data_o (fc3_out_s),
    .valid_o(fc3_valid_s)
  );
  fc_out_layer #(
    .NUM_NEURONS(FC_3_NEURONS),
    .ROM_DEPTH  (FC_3_ROM_DEPTH),
    .BIAS_SCALE (FC_3_OUT_BIAS_SCALE),
    .LAYER_SCALE(FC_3_OUT_LAYER_SCALE)
  ) u_fc3_out_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(fc3_valid_s),
    .data_i (fc3_out_s),
    .data_o (fc3_pre_relu_s),
    .valid_o(relu3_in_s)
  );
  relu_layer u_relu3 (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(relu3_in_s),
    .data_i (fc3_pre_relu_s),
    .data_o (fc3_post_relu_s),
    .valid_o(relu3_out_s)
  );
  fc_in_layer #(
    .NUM_NEURONS(FC_4_NEURONS),
    .INPUT_SIZE (FC_4_INPUT_SIZE),
    .BIAS_SCALE (FC_4_IN_BIAS_SCALE),
    .LAYER_SCALE(FC_4_IN_LAYER_SCALE)
  ) u_fc4_in_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(relu3_out_s),
    .data_i (fc3_post_relu_s),
    .data_o (fc4_out_s),
    .valid_o(fc4_valid_s)
  );
  fc_out_layer #(
    .NUM_NEURONS(FC_4_NEURONS),
    .ROM_DEPTH  (FC_4_ROM_DEPTH),
    .BIAS_SCALE (FC_4_OUT_BIAS_SCALE),
    .LAYER_SCALE(FC_4_OUT_LAYER_SCALE)
  ) u_fc4_out_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(fc4_valid_s),
    .data_i (fc4_out_s),
    .data_o (logits_s),
    .valid_o(logits_valid_s)
  );

  // ============================================================
  // Classifier head (softmax)
  // ============================================================
  softmax_layer #(
    .NUM_CLASSES(NUM_CLASSES)
  ) u_softmax_layer (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .valid_i(logits_valid_s),
    .data_i (logits_s),
    .data_o (head_data_s),
    .valid_o(head_valid_s)
  );

  // ============================================================
  // Output FIFO + AXI4-Stream master
  // ============================================================
  sync_fifo #(
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
      if (out_count_q == (NUM_CLASSES - 1))
        out_count_q <= '0;
      else
        out_count_q <= out_count_q + 1'b1;
    end
  end : output_tracking

  assign fifo_write_en_s   = head_valid_s && !fifo_full_s;
  assign fifo_read_en_s    = !fifo_empty_s && m_axis_prediction_tready_i;
  assign fifo_write_data_s = {{24{head_data_s[7]}}, head_data_s};

  assign m_axis_prediction_tvalid_o = tvalid_q;
  assign m_axis_prediction_tkeep_o  = 4'h1;
  assign m_axis_prediction_tlast_o  = (m_axis_prediction_tvalid_o && (out_count_q == (NUM_CLASSES - 1)));


endmodule : multiclass_NN

`end_keywords