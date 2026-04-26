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
File name:      multiclass_NN_wrapper.sv
  
Description:    Top-level wrapper for multiclass classifier (AXI4-Stream pass-through).
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

module multiclass_NN_wrapper
  import quant_pkg::*;
(
  input  logic clk_i,
  input  logic rst_n_i,

  input  q_data_t s_axis_tdata_i,
  input  logic    s_axis_tvalid_i,
  input  logic    s_axis_tlast_i,

  output logic [DATA_WIDTH-1:0] m_axis_prediction_tdata_o,
  output logic [KEEP_WIDTH-1:0] m_axis_prediction_tkeep_o,
  output logic                  m_axis_prediction_tvalid_o,
  input  logic                  m_axis_prediction_tready_i,
  output logic                  m_axis_prediction_tlast_o
);

  timeunit 1ns;
  timeprecision 1ps;

  multiclass_NN u_multiclass_nn (
    .clk_i  (clk_i),
    .rst_n_i(rst_n_i),
    .s_axis_tdata_i           (s_axis_tdata_i),
    .s_axis_tvalid_i          (s_axis_tvalid_i),
    .s_axis_tlast_i           (s_axis_tlast_i),
    .m_axis_prediction_tdata_o (m_axis_prediction_tdata_o),
    .m_axis_prediction_tkeep_o (m_axis_prediction_tkeep_o),
    .m_axis_prediction_tvalid_o(m_axis_prediction_tvalid_o),
    .m_axis_prediction_tready_i(m_axis_prediction_tready_i),
    .m_axis_prediction_tlast_o (m_axis_prediction_tlast_o)
  );

endmodule : multiclass_NN_wrapper

`end_keywords