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
File name:      quant_pkg.sv
  
Description:    Package defining quantization widths, fixed-point data types,
                and saturation limits for accumulation and activation functions.
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
***************************************************************************************/
`begin_keywords "1800-2012"

package quant_pkg;

  timeunit 1ns;
  timeprecision 1ps;

  // =============================================================
  // Quantization mode (select ONE at compile time)
  // =============================================================
  `ifdef Q_INT4
    localparam int Q_WIDTH = 32'd4;
  `elsif Q_INT8
    localparam int Q_WIDTH = 32'd8;
  `elsif Q_INT16
    localparam int Q_WIDTH = 32'd16;
  `else
    // Default (safety)
    localparam int Q_WIDTH = 32'd8;
  `endif

  // =============================================================
  // Common fixed-point types
  // =============================================================
  typedef logic signed [Q_WIDTH-1:0]       q_data_t;   // Quantized data
  typedef logic signed [2*Q_WIDTH-1:0]     q_mult_t;   // Multiply result
  typedef logic signed [4*Q_WIDTH-1:0]     acc_t;      // Accumulator

  //Widths of prediction axi4 stream
  localparam int DATA_WIDTH = 32'd32;
  localparam int KEEP_WIDTH = 32'd4;
  

  // =============================================================
  // Method A: Narrow limits (Q-width), then cast to acc_t
  // Use when accumulator must saturate to Q range
  // =============================================================
  localparam q_data_t Q_MAX = {1'b0, {Q_WIDTH-1{1'b1}}};
  localparam q_data_t Q_MIN = {1'b1, {Q_WIDTH-1{1'b0}}};

  localparam acc_t ACC_Q_MAX = acc_t'(Q_MAX);
  localparam acc_t ACC_Q_MIN = acc_t'(Q_MIN);

  // =============================================================
  // Method B: Native accumulator limits (full acc_t width)
  // Use when accumulator keeps full dynamic range
  // =============================================================
  localparam acc_t ACC_FULL_MAX = {1'b0, {4*Q_WIDTH-1{1'b1}}};
  localparam acc_t ACC_FULL_MIN = {1'b1, {4*Q_WIDTH-1{1'b0}}};

  // =============================================================
  // Activation Function Limits
  // =============================================================
  localparam q_data_t SIGMOID_MAX = 1 << (Q_WIDTH - 2); 
  localparam q_data_t SIGMOID_MIN = 8'h0;

endpackage: quant_pkg

`end_keywords