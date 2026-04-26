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
File name:      line_buffers.sv
  
Description:    Sliding K_H x K_W window line buffer for streaming 2D convolution.
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

// ----------------------------------------------------------------------------
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

`end_keywords