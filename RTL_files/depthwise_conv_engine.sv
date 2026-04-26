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
File name:      depthwise_conv_engine.sv
  
Description:    Depthwise / kH x kW projection convolution engine with line buffer, parameterized layer-ROM selection.
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

module depthwise_conv_engine #(
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
    if (LAYER_INDEX == 1) begin : gen_conv2d_proj_conv
      conv2d_proj_conv_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
      ) the_conv2d_proj_conv_weights_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
      );

      conv2d_proj_conv_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
      ) the_conv2d_proj_conv_bias_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
      );
    end
    else if (LAYER_INDEX == 2) begin : gen_conv2d_1_proj_conv
      conv2d_1_proj_conv_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
      ) the_conv2d_1_proj_conv_weights_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
      );

      conv2d_1_proj_conv_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
      ) the_conv2d_1_proj_conv_bias_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
      );
    end
    else if (LAYER_INDEX == 3) begin : gen_conv2d_2_proj_conv
      conv2d_2_proj_conv_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
      ) the_conv2d_2_proj_conv_weights_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
      );

      conv2d_2_proj_conv_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
      ) the_conv2d_2_proj_conv_bias_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
      );
    end
    endgenerate

    // -------------------- Pipeline (multiply -> accumulate -> bias + scale) --------------------
    localparam int PIPELINE_DEPTH = 3;
    logic [PIPELINE_DEPTH-1:0] valid_pipe;
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            valid_pipe <= '0;
        else
            valid_pipe <= {valid_pipe[PIPELINE_DEPTH-2:0], lb_valid};
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

`end_keywords