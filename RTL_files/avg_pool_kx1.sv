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
File name:      avg_pool_kx1.sv
  
Description:    Streaming average pooling along the row dimension (kernel KxKERNEL=Kx1, stride STRIDExSTRIDE).
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

// ----------------------------------------------------------------------------
// avg_pool_kx1 — average pooling along the row dimension only (W=1)
// ----------------------------------------------------------------------------
// Implements AvgPool2d(kernel_size=(KERNEL,1), stride=(STRIDE,STRIDE)) for a
// streaming H x 1 x CHANNELS frame. Output rows = floor((FRAME_ROWS-KERNEL)/STRIDE)+1.
//
// FRAME_ROWS is parameterized at instantiation time so the same module serves
// any input height. The row counter wraps at FRAME_ROWS-1 to detect frame end
// without an external "last" pin. STRIDE is assumed power-of-two so cnt naturally
// cycles through 0..STRIDE-1 by bit-select on row_cnt.
// ----------------------------------------------------------------------------
module avg_pool_kx1 #(
    parameter int DATA_W     = 8,
    parameter int CHANNELS   = 8,
    parameter int FRAME_ROWS = 714,
    parameter int KERNEL     = 2,
    parameter int STRIDE     = 4
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [DATA_W-1:0] data_in  [CHANNELS],
    input  logic              valid_in,

    output logic [DATA_W-1:0] data_out [CHANNELS],
    output logic              valid_out
);

    timeunit 1ns;
    timeprecision 1ps;

    localparam int ROW_BITS = $clog2(FRAME_ROWS);
    localparam int CNT_BITS = $clog2(STRIDE);
    localparam int EMIT_AT  = KERNEL - 1;

    logic [ROW_BITS-1:0] row_cnt;
    logic                last_row;
    logic [CNT_BITS-1:0] cnt;
    logic [DATA_W-1:0]   acc [CHANNELS];
    logic [DATA_W:0]     sum [CHANNELS];
    logic [DATA_W-1:0]   avg [CHANNELS];

    assign last_row = (row_cnt == ROW_BITS'(FRAME_ROWS - 1));
    assign cnt      = row_cnt[CNT_BITS-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)        row_cnt <= '0;
        else if (valid_in) row_cnt <= last_row ? '0 : row_cnt + 1'b1;
    end

    always_ff @(posedge clk) begin
        if (valid_in && cnt == '0) acc <= data_in;
    end

    always_comb
        for (int c = 0; c < CHANNELS; c++) begin
            sum[c] = {1'b0, acc[c]} + {1'b0, data_in[c]};
            avg[c] = sum[c][DATA_W:1];
        end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
        end
        else begin
            valid_out <= valid_in && (cnt == CNT_BITS'(EMIT_AT));
            if (valid_in && cnt == CNT_BITS'(EMIT_AT))
                data_out <= avg;
        end
    end

endmodule

`end_keywords