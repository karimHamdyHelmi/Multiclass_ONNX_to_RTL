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
File name:      sync_fifo.sv
  
Description:    Synchronous FIFO module for data buffering. Uses a standard pointer-based implementation with full/empty flags
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

module sync_fifo #(
  parameter int DATA_WIDTH = 32,
  parameter int DEPTH      = 1024
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  logic                  write_en_i,
  input  logic [DATA_WIDTH-1:0] write_data_i,
  output logic                  full_o,

  input  logic                  read_en_i,
  output logic [DATA_WIDTH-1:0] read_data_o,
  output logic                  empty_o
);

  timeunit 1ns;
  timeprecision 1ps;

  localparam int ADDR_WIDTH = $clog2(DEPTH);

  logic [DATA_WIDTH-1:0] fifo_mem_q [0:DEPTH-1];
  logic [ADDR_WIDTH:0] write_ptr_q;
  logic [ADDR_WIDTH:0] read_ptr_q;

  always_ff @(posedge clk_i or negedge rst_n_i) begin : write_ptr_logic
    if (!rst_n_i) begin
      write_ptr_q <= '0;
    end
    else if (write_en_i && !full_o) begin
      write_ptr_q <= write_ptr_q + 1'b1;
    end
  end : write_ptr_logic

  always_ff @(posedge clk_i) begin : write_mem_logic
    if (write_en_i && !full_o) begin
      fifo_mem_q[write_ptr_q[ADDR_WIDTH-1:0]] <= write_data_i;
    end
  end : write_mem_logic

  always_ff @(posedge clk_i or negedge rst_n_i) begin : read_logic
    if (!rst_n_i) begin
      read_ptr_q <= '0;
      read_data_o <= '0;
    end
    else if (read_en_i && !empty_o) begin
      read_data_o <= fifo_mem_q[read_ptr_q[ADDR_WIDTH-1:0]];
      read_ptr_q <= read_ptr_q + 1'b1;
    end
  end : read_logic

  assign empty_o = (write_ptr_q == read_ptr_q);
  assign full_o  = (write_ptr_q[ADDR_WIDTH] != read_ptr_q[ADDR_WIDTH]) &&
                   (write_ptr_q[ADDR_WIDTH-1:0] == read_ptr_q[ADDR_WIDTH-1:0]);

endmodule : sync_fifo

`end_keywords
