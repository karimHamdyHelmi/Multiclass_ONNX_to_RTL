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
File name:      flatten_unit.sv
  
Description:    Transpose [0,1,3,2] + Reshape buffer: rebuilds POOL_ROWS x CHANNELS into a flat channel-major stream.
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

// ----------------------------------------------------------------------------
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

`end_keywords