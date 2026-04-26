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
File name:      softmax_layer.sv
  
Description:    Softmax classifier head: passes raw logits through unchanged (downstream applies softmax in software).
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

// ----------------------------------------------------------------------------
// softmax_layer — INT8 streaming passthrough (raw logits)
// ----------------------------------------------------------------------------
// In an INT8 fixed-point pipeline a true softmax requires an exp() LUT and a
// division stage; both add significant area and are unnecessary when the
// downstream consumer only needs the predicted class (argmax of softmax ==
// argmax of logits). This module therefore passes the streamed logits through
// untouched, with one logit per valid pulse for NUM_CLASSES cycles per frame.
//
// If the integrating system needs probability values, apply softmax in software
// on the streamed logits. To enable hardware softmax later, replace this body
// with an exp-LUT + sum + divider pipeline.
// ----------------------------------------------------------------------------
module softmax_layer
  import quant_pkg::*;
#(
  parameter int NUM_CLASSES = 10
)(
  input  logic    clk_i,
  input  logic    rst_n_i,

  input  logic    valid_i,
  input  q_data_t data_i,

  output q_data_t data_o,
  output logic    valid_o
);

  timeunit 1ns;
  timeprecision 1ps;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      data_o  <= '0;
      valid_o <= 1'b0;
    end
    else begin
      valid_o <= valid_i;
      if (valid_i) data_o <= data_i;
    end
  end

endmodule : softmax_layer

`end_keywords