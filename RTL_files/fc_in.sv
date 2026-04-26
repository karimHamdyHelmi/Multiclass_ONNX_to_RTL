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
File name:      fc_in.sv
  
Description:    Fully-connected (FC) input layer module. Instantiates MAC units, adds biases, and applies layer scaling with output saturation.
  
Author:         Ahmed Abou-Auf
  
Change History:
02-25-2026     AA  Initial Release
  
****************************************************************************************/
`begin_keywords "1800-2012"

// MAC (F_mac) + bias: bias_aligned = biases_i <<< BIAS_SCALE (see quantize_bias_for_rtl).
module fc_in
  import quant_pkg::*;
#(
  parameter int    NUM_NEURONS = 8,
  parameter int    INPUT_SIZE  = 16,
  parameter signed BIAS_SCALE  = 0,
  parameter signed LAYER_SCALE = 12
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  logic valid_i,                  // Input data valid
  input  q_data_t data_i,                // Input data
  input  q_data_t weights_i[NUM_NEURONS], // Weight vector per neuron
  input  acc_t    biases_i[NUM_NEURONS],  // Biases vector per neuron

  output q_data_t data_o[NUM_NEURONS],    // FC layer output
  output logic    valid_o                // Output valid pulse
);

  timeunit 1ns;
  timeprecision 1ps;

  // ------------------------------------------------------------
  // Internal signals
  // ------------------------------------------------------------
  acc_t mac_acc_q[NUM_NEURONS];          // Registered MAC outputs
  acc_t acc_tmp_s[NUM_NEURONS];          // Combinational bias addition
  acc_t bias_aligned_s[NUM_NEURONS];     // Combinational bias alignment
  acc_t data_out_temp_s[NUM_NEURONS];    // Combinational layer scaling

  logic [$clog2(INPUT_SIZE + 1) - 1:0] count_q;

  logic mac_enable_s;
  logic [NUM_NEURONS- 1:0] mac_valid_q;
  logic all_mac_valid_s;

  // ------------------------------------------------------------
  // Instantiate MAC units
  // ------------------------------------------------------------
  genvar i;
  generate
    for (i = 0; i < NUM_NEURONS; i++) begin : gen_mac_units
      mac #(
          .INPUT_SIZE(INPUT_SIZE)
      ) u_mac (
        .clk_i     (clk_i),
        .rst_n_i     (rst_n_i),
        .enable_i  (mac_enable_s),
        .a_i       (data_i),
        .b_i       (weights_i[i]),
        .valid_o   (mac_valid_q[i]),
        .acc_o     (mac_acc_q[i])
      );
    end
  endgenerate

  assign all_mac_valid_s = &mac_valid_q;
  assign mac_enable_s    = valid_i;

  // ------------------------------------------------------------
  // Count valid inputs and generate valid_o for last input
  // ------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin : count_logic
    if (!rst_n_i) begin
      count_q <= '0;
      valid_o <= 1'b0;
    end
    else begin
      valid_o <= 1'b0;
      if (all_mac_valid_s) begin
        if (count_q == (INPUT_SIZE - 1)) begin
          count_q <= '0;
          valid_o <= 1'b1;
        end
        else begin
          count_q <= count_q + 1'b1;
        end
      end
    end
  end : count_logic

  // ------------------------------------------------------------
  // Bias addition, layer scaling, and output saturation
  // ------------------------------------------------------------
  generate
    for (i = 0; i < NUM_NEURONS; i++) begin : gen_output_path
      // Align bias to accumulator width
      assign bias_aligned_s[i] = (BIAS_SCALE >= 0) ?
                                 (biases_i[i] <<< BIAS_SCALE) :
                                 (biases_i[i] >>> -BIAS_SCALE);

      // Add bias to MAC output
      assign acc_tmp_s[i] = mac_acc_q[i] + bias_aligned_s[i];

      // Apply layer scaling
      assign data_out_temp_s[i] = (LAYER_SCALE >= 0) ?
                                  (acc_tmp_s[i] >>> LAYER_SCALE) :
                                  (acc_tmp_s[i] <<< -LAYER_SCALE);

      // Register output with saturation logic
      always_ff @(posedge clk_i or negedge rst_n_i) begin : saturation_reg
        if (!rst_n_i) begin
          data_o[i] <= '0;
        end
        else if (count_q == (INPUT_SIZE-1)) begin
          if (data_out_temp_s[i] > ACC_Q_MAX) begin
            data_o[i] <= Q_MAX;
          end
          else if (data_out_temp_s[i] < ACC_Q_MIN) begin
            data_o[i] <= Q_MIN;
          end
          else begin
            // Cast to narrow quantized data type
            data_o[i] <= data_out_temp_s[i][Q_WIDTH-1:0];
          end
        end
      end : saturation_reg
    end
  endgenerate

endmodule : fc_in

`end_keywords
