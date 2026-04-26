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
File name:      conv2d_proj_conv_weights_rom.sv
  
Description:    Weights ROM for conv layer conv2d_proj_conv
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

module conv2d_proj_conv_weights_rom #(
  parameter int DEPTH  = 1,
  parameter int WIDTH  = 144,
  parameter int ADDR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
  input  logic              clk_i,
  input  logic [ADDR_W-1:0] addr_i,
  output logic [WIDTH-1:0]  data_o
);

  timeunit 1ns;
  timeprecision 1ps;

  (* rom_style = "block" *)
  logic [WIDTH-1:0] mem [0:DEPTH-1];

  initial begin
    $readmemh({`__FILE__, "/../mem_files/conv2d_proj_conv_weights.mem"}, mem);
  end

  always_ff @(posedge clk_i) begin : read_port
    data_o <= mem[addr_i];
  end : read_port

endmodule : conv2d_proj_conv_weights_rom

`end_keywords