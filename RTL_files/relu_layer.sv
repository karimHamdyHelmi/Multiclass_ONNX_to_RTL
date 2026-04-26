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
File name:      relu_layer.sv
  
Description:    ReLU activation function applied element-wise to a single input. Passes input directly if positive; outputs zero if negative.
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

module relu_layer 
  import quant_pkg::*;
(
  input  logic    clk_i,
  input  logic    rst_n_i,

  input  logic    valid_i,               // Input data valid
  input  q_data_t data_i,                // Input quantized data

  output q_data_t data_o,                // ReLU activated output
  output logic    valid_o                // Output valid signal
);

  timeunit 1ns;
  timeprecision 1ps;

  // ============================================================
  // ReLU logic with pipelined valid signal
  // ============================================================
  always_ff @(posedge clk_i or negedge rst_n_i) begin : relu_pipeline
    if (!rst_n_i) begin
      data_o  <= '0;
      valid_o <= 1'b0;
    end 
    else begin
      valid_o <= valid_i;
      if (valid_i) begin
        // Rectified Linear Unit logic: output = max(0, input)
        data_o <= (data_i < 0) ? 0 : data_i;
      end
    end
  end : relu_pipeline

endmodule : relu_layer

`end_keywords
