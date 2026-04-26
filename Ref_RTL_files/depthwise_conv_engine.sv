module depthwise_conv_engine #(
    parameter DATA_WIDTH       = 8,        // input data width
    parameter K_H              = 3,        // kernel height
    parameter K_W              = 3,        // kernel width
    parameter NUM_FILTERS      = 2,        // number of convolution filters
    parameter BIAS_WIDTH       = 32,       // bias width
    parameter LAYER_SCALE      = 7,        // right shift for layer scaling
    parameter DW_CONV_INDEX    = 1,        // right shift for layer scaling
    parameter INPUTS_PER_CYCLE = 1         // number of pixels per cycle
)(
    input  logic                                     clk_i,
    input  logic                                     rst_n_i,
    input  logic                                     valid_in,
    input  logic signed [INPUTS_PER_CYCLE*DATA_WIDTH-1:0]  data_in,
    output logic signed [DATA_WIDTH-1:0]            conv_out [0:NUM_FILTERS-1],
    output logic                                     valid_out
);

    // ------------------------- Line Buffers ----------------------------
    logic signed [DATA_WIDTH-1:0] lb_out [0:K_H-1][0:K_W-1];
    logic                  lb_valid;

    line_buffers #(
        .DATA_WIDTH(DATA_WIDTH),
        .K_H(K_H),
        .K_W(K_W),
        .INPUTS_PER_CYCLE(INPUTS_PER_CYCLE)
    ) lb_inst (
        .clk(clk_i),
        .rst_n(rst_n_i),
        .valid_in(valid_in),
        .data_in(data_in),
        .data_out(lb_out),
        .valid_out(lb_valid)
    );

    // ------------------------- Flatten Window --------------------------
    localparam WINDOW_SIZE = K_H * K_W;
    logic signed [DATA_WIDTH-1:0] window_flat [0:WINDOW_SIZE-1];

    genvar i, j;
    generate
        for (i = 0; i < K_H; i = i+1) begin : GEN_FLATTEN_ROW
            for (j = 0; j < K_W; j = j+1) begin : GEN_FLATTEN_COL
                assign window_flat[i*K_W + j] = lb_out[i][j];
            end
        end
    endgenerate

    // ------------------------- Weights & Bias ROM ----------------------
    localparam WEIGHTS_ROW_WIDTH = NUM_FILTERS * WINDOW_SIZE * DATA_WIDTH;
    logic signed [WEIGHTS_ROW_WIDTH-1:0]      weights_rom_row;
    logic signed [BIAS_WIDTH*NUM_FILTERS-1:0] bias_rom_row;
    
	
	generate
	if(DW_CONV_INDEX==1)begin
    conv2d_proj_conv_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
    ) the_conv2d_proj_conv_weights_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
    );

    conv2d_proj_conv_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
    ) the_conv2d_proj_conv_bias_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
    );
	
	end
	
	else if (DW_CONV_INDEX==2) begin
	     conv2d_1_proj_conv_weights_rom #(
             .DEPTH(1),
             .WIDTH(WEIGHTS_ROW_WIDTH),
             .ADDR_W(1)
         ) the_conv2d_1_proj_conv_weights_rom (
             .clk_i(clk_i),
             .addr_i(1'b0),
             .data_o(weights_rom_row)
         );
	     
         conv2d_1_proj_conv_bias_rom #(
             .DEPTH(1),
             .WIDTH(BIAS_WIDTH * NUM_FILTERS),
             .ADDR_W(1)
         ) the_conv2d_1_proj_conv_bias_rom (
             .clk_i(clk_i),
             .addr_i(1'b0),
             .data_o(bias_rom_row)
         );
	end
	
	else begin
	     conv2d_2_proj_conv_weights_rom #(
             .DEPTH(1),
             .WIDTH(WEIGHTS_ROW_WIDTH),
             .ADDR_W(1)
         ) the_conv2d_2_proj_conv_weights_rom (
             .clk_i(clk_i),
             .addr_i(1'b0),
             .data_o(weights_rom_row)
         );
	     
         conv2d_2_proj_conv_bias_rom #(
             .DEPTH(1),
             .WIDTH(BIAS_WIDTH * NUM_FILTERS),
             .ADDR_W(1)
         ) the_conv2d_2_proj_conv_bias_rom (
             .clk_i(clk_i),
             .addr_i(1'b0),
             .data_o(bias_rom_row)
         );
	end
	endgenerate

    // ------------------------- Valid Pipeline --------------------------
    localparam PIPELINE_DEPTH = 3;
    logic [PIPELINE_DEPTH-1:0] valid_pipe;

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            valid_pipe <= '0;
        else
            valid_pipe <= {valid_pipe[PIPELINE_DEPTH-2:0], lb_valid};
    end
    assign valid_out = valid_pipe[PIPELINE_DEPTH-1];

    // ------------------------- Multiply Stage --------------------------
    logic signed [DATA_WIDTH*2-1:0] mult_pipe [0:NUM_FILTERS-1][0:WINDOW_SIZE-1];

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            for (int f = 0; f < NUM_FILTERS; f++)
                for (int k = 0; k < WINDOW_SIZE; k++)
                    mult_pipe[f][k] <= '0;
        end else if (lb_valid) begin
            for (int f = 0; f < NUM_FILTERS; f++)
                for (int k = 0; k < WINDOW_SIZE; k++)
                    mult_pipe[f][k] <= $signed(window_flat[k]) * 
                        $signed(weights_rom_row[DATA_WIDTH*(f*WINDOW_SIZE + k + 1)-1 -: DATA_WIDTH]);
        end
    end

    // ------------------------- Accumulate Stage ------------------------
    logic signed [31:0] sum_pipe [0:NUM_FILTERS-1];

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            for (int f = 0; f < NUM_FILTERS; f++)
                sum_pipe[f] <= '0;
        else if (valid_pipe[0]) begin
            for (int f = 0; f < NUM_FILTERS; f++) begin
                automatic logic signed [31:0] sum_tmp;
                sum_tmp = 0;
                for (int k = 0; k < WINDOW_SIZE; k++)
                    sum_tmp += mult_pipe[f][k];
                sum_pipe[f] <= sum_tmp;
            end
        end
    end

    // ------------------------- Bias + Scale + Saturation ----------------
    logic signed [31:0] conv_out_pipe [0:NUM_FILTERS-1];

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            for (int f = 0; f < NUM_FILTERS; f++)
                conv_out_pipe[f] <= '0;
        end else if (valid_pipe[1]) begin
            for (int f = 0; f < NUM_FILTERS; f++) begin
                automatic logic signed [31:0] pre_shift;
                automatic logic signed [31:0] shifted;
                automatic logic signed [31:0] rounding_bit;

                // Bias addition
                pre_shift = sum_pipe[f] + $signed(bias_rom_row[BIAS_WIDTH*(f+1)-1 -: BIAS_WIDTH]);

                /* // Signed rounding
                rounding_bit = pre_shift[LAYER_SCALE-1] ? (pre_shift >= 0 ? 1 : -1) : 0;

                // Arithmetic shift + rounding
                shifted = (pre_shift >>> LAYER_SCALE) + rounding_bit; */
                shifted = (pre_shift >>> LAYER_SCALE);

                // Saturate to int8
                if (shifted > 127)
                    conv_out_pipe[f] <= 127;
                else if (shifted < -128)
                    conv_out_pipe[f] <= -128;
                else
                    conv_out_pipe[f] <= shifted;
            end
        end
    end

    // ------------------------- Output Assignment ----------------------
    generate
        for (i = 0; i < NUM_FILTERS; i = i+1)
            assign conv_out[i] = conv_out_pipe[i][DATA_WIDTH-1:0];
    endgenerate

endmodule