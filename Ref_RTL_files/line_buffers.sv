// -----------------------------------------------------------------------------
// line_buffers_multi_input_fixed.sv
// Builds rows of width K_W from streaming input, stores last K_H rows,
// outputs a vertical sliding window of size K_H x K_W.
// Supports multi-input per cycle (INPUTS_PER_CYCLE >= 1)
// valid_out pulses for 1 cycle whenever a new full window is ready.
// -----------------------------------------------------------------------------
module line_buffers #(
    parameter DATA_WIDTH       = 8,
    parameter K_W              = 3,
    parameter K_H              = 3,
    parameter INPUTS_PER_CYCLE = 1
)(
    input  logic                                   clk,
    input  logic                                   rst_n,
    input  logic                                   valid_in,
    input  logic signed [INPUTS_PER_CYCLE*DATA_WIDTH-1:0] data_in,

    output logic signed [DATA_WIDTH-1:0] data_out [0:K_H-1][0:K_W-1],
    output logic                                   valid_out
);

    // ------------------------------------------------------------------------
    // Internal storage
    // ------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] row_buf     [0:K_H-1][0:K_W-1]; // stacked rows
    logic signed [DATA_WIDTH-1:0] current_row [0:K_W-1];          // building current row

    logic [$clog2(K_W+1)-1:0] col_counter;
    logic [$clog2(K_H+1)-1:0] valid_row_count;
    logic row_complete;

    // unpack input bus
    logic [DATA_WIDTH-1:0] input_vals [0:INPUTS_PER_CYCLE-1];

    // procedural variables declared at top
    logic [$clog2(K_W+1)-1:0] next_col;
    logic [$clog2(INPUTS_PER_CYCLE+1)-1:0] pix_idx;
    logic [$clog2(K_W+1)-1:0] space_left;
    logic [$clog2(K_W+1)-1:0] to_place;

    genvar g;
    generate
        for (g = 0; g < INPUTS_PER_CYCLE; g = g + 1)
            assign input_vals[g] = data_in[DATA_WIDTH*(g+1)-1 -: DATA_WIDTH];
    endgenerate

    // ------------------------------------------------------------------------
    // Fill current row safely with multi-input-per-cycle
    // ------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_counter  <= '0;
            row_complete <= 1'b0;
            for (int c = 0; c < K_W; c++)
                current_row[c] <= '0;
        end
        else begin
            row_complete <= 1'b0;
            if (valid_in) begin
                next_col = col_counter;
                pix_idx  = 0;

                while (pix_idx < INPUTS_PER_CYCLE) begin
                    space_left = K_W - next_col;
                    to_place   = (INPUTS_PER_CYCLE - pix_idx <= space_left) ? 
                                 (INPUTS_PER_CYCLE - pix_idx) : space_left;

                    for (int j = 0; j < to_place; j++) begin
                        current_row[next_col] <= input_vals[pix_idx];
                        next_col = next_col + 1;
                        pix_idx  = pix_idx + 1;
                    end

                    if (next_col >= K_W) begin
                        row_complete <= 1'b1;
                        col_counter  <= 0;
                        next_col     = 0;
                        // leave extra pixels to next loop iteration
                    end
                end

                col_counter <= next_col;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Shift completed rows into row buffer stack
    // ------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_row_count <= '0;
            valid_out       <= 1'b0;
            for (int r = 0; r < K_H; r++)
                for (int c = 0; c < K_W; c++)
                    row_buf[r][c] <= '0;
        end
        else begin
            valid_out <= 1'b0;

            if (row_complete) begin
                // shift rows up
                for (int r = 0; r < K_H-1; r++)
                    for (int c = 0; c < K_W; c++)
                        row_buf[r][c] <= row_buf[r+1][c];

                // insert new row at bottom
                for (int c = 0; c < K_W; c++)
                    row_buf[K_H-1][c] <= current_row[c];

                // count valid rows for initial filling
                if (valid_row_count < K_H)
                    valid_row_count <= valid_row_count + 1'b1;

                // window valid when stack has K_H rows
                if (valid_row_count >= K_H-1)
                    valid_out <= 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Output assignments
    // ------------------------------------------------------------------------
    generate
        for (g = 0; g < K_H; g = g + 1)
            for (genvar c = 0; c < K_W; c = c + 1)
                assign data_out[g][c] = row_buf[g][c];
    endgenerate

endmodule