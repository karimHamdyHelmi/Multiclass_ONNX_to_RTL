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
File name:      pointwise_conv_engine.sv
  
Description:    1x1 pointwise convolution engine with parallel input channels and parameterized layer-ROM selection.
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

module pointwise_conv_engine #(
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
    if (LAYER_INDEX == 1) begin : gen_conv2d_proj_out
      conv2d_proj_out_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
      ) the_conv2d_proj_out_weights_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
      );

      conv2d_proj_out_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
      ) the_conv2d_proj_out_bias_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
      );
    end
    else if (LAYER_INDEX == 2) begin : gen_conv2d_1_proj_in
      conv2d_1_proj_in_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
      ) the_conv2d_1_proj_in_weights_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
      );

      conv2d_1_proj_in_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
      ) the_conv2d_1_proj_in_bias_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
      );
    end
    else if (LAYER_INDEX == 3) begin : gen_conv2d_1_proj_out
      conv2d_1_proj_out_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
      ) the_conv2d_1_proj_out_weights_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
      );

      conv2d_1_proj_out_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
      ) the_conv2d_1_proj_out_bias_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
      );
    end
    else if (LAYER_INDEX == 4) begin : gen_conv2d_2_proj_in
      conv2d_2_proj_in_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
      ) the_conv2d_2_proj_in_weights_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
      );

      conv2d_2_proj_in_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
      ) the_conv2d_2_proj_in_bias_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
      );
    end
    else if (LAYER_INDEX == 5) begin : gen_conv2d_2_proj_out
      conv2d_2_proj_out_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
      ) the_conv2d_2_proj_out_weights_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
      );

      conv2d_2_proj_out_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
      ) the_conv2d_2_proj_out_bias_rom (
        .clk_i (clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
      );
    end
    endgenerate

    localparam int PIPELINE_DEPTH = 3;
    logic [PIPELINE_DEPTH-1:0] valid_pipe;
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) valid_pipe <= '0;
        else          valid_pipe <= {valid_pipe[PIPELINE_DEPTH-2:0], valid_in};
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

`end_keywords