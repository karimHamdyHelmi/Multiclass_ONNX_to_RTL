module pointwise_conv_engine #(
    parameter DATA_WIDTH       = 8,    // input/weight data width
    parameter NUM_FILTERS      = 2,   // number of pointwise filters
    parameter INPUTS_PER_CYCLE = 2,    // number of input channels per cycle
    parameter BIAS_WIDTH       = 32,   // bias word width
    parameter PW_CONV_INDEX     = 1,   // bias word width
    parameter LAYER_SCALE      = 5     // right shift for quantization scaling
)(
    input  logic                                      clk_i,
    input  logic                                      rst_n_i,
    input  logic                                      valid_in,
    input  logic signed [DATA_WIDTH-1:0]             data_in [0:INPUTS_PER_CYCLE-1], 
    output logic signed [DATA_WIDTH-1:0]             conv_out [0:NUM_FILTERS-1],
    output logic                                      valid_out
);

    // ------------------------- Input Assignment ------------------------
    logic signed [DATA_WIDTH-1:0] ch_in [0:INPUTS_PER_CYCLE-1];
    genvar k;
    generate
        for (k = 0; k < INPUTS_PER_CYCLE; k = k+1) begin : GEN_CH_IN
            assign ch_in[k] = data_in[k];
        end
    endgenerate

    // ------------------------- Weights & Bias ROMs ----------------------
    localparam WEIGHTS_ROW_WIDTH = NUM_FILTERS * INPUTS_PER_CYCLE * DATA_WIDTH;
    logic signed [WEIGHTS_ROW_WIDTH-1:0]       weights_rom_row;
    logic signed [BIAS_WIDTH*NUM_FILTERS-1:0]  bias_rom_row;
    
	
	generate 
	if(PW_CONV_INDEX==1)begin
    conv2d_proj_out_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
    ) the_conv2d_proj_out_weights_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
    );

    conv2d_proj_out_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
    ) the_conv2d_proj_out_bias_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
    );
	end
	else if(PW_CONV_INDEX==2) begin
	    conv2d_1_proj_in_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
    ) the_conv2d_1_proj_in_weights_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
    );

    conv2d_1_proj_in_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
    ) the_conv2d_1_proj_in_bias_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
    );
	
	end
	
	else if(PW_CONV_INDEX==3) begin
	    conv2d_1_proj_out_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
    ) the_conv2d_1_proj_out_weights_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
    );

    conv2d_1_proj_out_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
    ) the_conv2d_1_proj_out_bias_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
    );
	
	end
	
	else if(PW_CONV_INDEX==4) begin
	    conv2d_2_proj_in_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
    ) the_conv2d_2_proj_in_weights_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
    );

    conv2d_2_proj_in_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
    ) the_conv2d_2_proj_in_bias_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(bias_rom_row)
    );
	
	end
	else if(PW_CONV_INDEX==5) begin
	    conv2d_2_proj_out_weights_rom #(
        .DEPTH(1),
        .WIDTH(WEIGHTS_ROW_WIDTH),
        .ADDR_W(1)
    ) the_conv2d_2_proj_out_weights_rom (
        .clk_i(clk_i),
        .addr_i(1'b0),
        .data_o(weights_rom_row)
    );

    conv2d_2_proj_out_bias_rom #(
        .DEPTH(1),
        .WIDTH(BIAS_WIDTH * NUM_FILTERS),
        .ADDR_W(1)
    ) the_conv2d_2_proj_out_bias_rom (
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
            valid_pipe <= {valid_pipe[PIPELINE_DEPTH-2:0], valid_in};
    end
    assign valid_out = valid_pipe[PIPELINE_DEPTH-1];

    // ------------------------- Stage 1: Multiply -----------------------
    logic signed [DATA_WIDTH*2-1:0] mult_pipe [0:NUM_FILTERS-1][0:INPUTS_PER_CYCLE-1];
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            for (int f = 0; f < NUM_FILTERS; f++)
                for (int i = 0; i < INPUTS_PER_CYCLE; i++)
                    mult_pipe[f][i] <= '0;
        end else if (valid_in) begin
            for (int f = 0; f < NUM_FILTERS; f++)
                for (int i = 0; i < INPUTS_PER_CYCLE; i++)
                    mult_pipe[f][i] <= $signed(ch_in[i]) * 
                        $signed(weights_rom_row[DATA_WIDTH*(f*INPUTS_PER_CYCLE + i + 1)-1 -: DATA_WIDTH]);
        end
    end

    // ------------------------- Stage 2: Accumulate --------------------
    logic signed [31:0] sum_pipe [0:NUM_FILTERS-1];
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            for (int f = 0; f < NUM_FILTERS; f++)
                sum_pipe[f] <= '0;
        else if (valid_pipe[0])
            for (int f = 0; f < NUM_FILTERS; f++) begin
                automatic logic signed [31:0] sum_tmp = 0;
                for (int i = 0; i < INPUTS_PER_CYCLE; i++)
                    sum_tmp += mult_pipe[f][i];
                sum_pipe[f] <= sum_tmp;
            end
    end

    // ------------------------- Stage 3: Bias + Scale + Saturation -----
    logic signed [31:0] conv_out_pipe [0:NUM_FILTERS-1];
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            for (int f = 0; f < NUM_FILTERS; f++)
                conv_out_pipe[f] <= '0;
        else if (valid_pipe[1])
            for (int f = 0; f < NUM_FILTERS; f++) begin
                automatic logic signed [31:0] pre_shift;
                automatic logic signed [31:0] shifted;

                // Add bias
                pre_shift = sum_pipe[f] + $signed(bias_rom_row[BIAS_WIDTH*(f+1)-1 -: BIAS_WIDTH]);

                // Arithmetic right shift
                shifted = pre_shift >>> LAYER_SCALE;

               /*  // Signed rounding
                if (pre_shift >= 0)
                    shifted = shifted + pre_shift[LAYER_SCALE-1];
                else
                    shifted = shifted - pre_shift[LAYER_SCALE-1]; */

                // Saturate to int8
                if (shifted > 127)
                    conv_out_pipe[f] <= 127;
                else if (shifted < -128)
                    conv_out_pipe[f] <= -128;
                else
                    conv_out_pipe[f] <= shifted;
            end
    end

    // ------------------------- Output Assignment ----------------------
    genvar i;
    generate
        for (i = 0; i < NUM_FILTERS; i = i+1)
            assign conv_out[i] = conv_out_pipe[i][DATA_WIDTH-1:0];
    endgenerate

endmodule