// =============================================================
//  avg_pool_2x1_stride4.sv
//
//  Implements : AvgPool2d(kernel_size=(2,1), stride=(4,4))
//  Input      : 714 × 1 × 8  (H × W × C),  batch = 1
//  Output     : 179 × 1 × 8
//
//  Streaming interface:
//    data_in / valid_in  — one row per valid cycle (8 channels)
//    data_out / valid_out — one pooled row per output (8 channels)
//
//  valid_in may go low between rows inside a frame (back-pressure
//  or upstream stall).  The row counter freezes on valid_in=0 and
//  resumes when valid_in returns — mid-frame gaps are invisible.
//
//  Frame boundary is detected purely by row_cnt reaching
//  FRAME_ROWS-1.  No extra pins needed.
//
//  Window behaviour (stride=4, kernel=2):
//    cnt==0 : latch row into accumulator
//    cnt==1 : avg = (acc + data_in)>>1  →  valid_out=1
//    cnt==2 : discard row
//    cnt==3 : discard row  →  row_cnt wraps on next valid row
//
//  Output count : floor((714-2)/4)+1 = 179  ✓
//
//  Resources : 8 acc regs + 10-bit row counter + 8 adders
//  No BRAM. No modulo. No timeout. No extra control pins.
// =============================================================
module avg_pool_2x1_stride4 #(
    parameter int DATA_W     = 8,
    parameter int CHANNELS   = 8,
    parameter int FRAME_ROWS = 714,
    parameter int KERNEL     = 2,
    parameter int STRIDE     = 4
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [DATA_W-1:0] data_in  [CHANNELS],
    input  logic               valid_in,

    output logic [DATA_W-1:0] data_out [CHANNELS],
    output logic               valid_out
);

// ──────────────────────────────────────────────────────────────
// 1.  Row counter
//
//     Increments ONLY on valid_in=1.
//     Mid-frame gaps (valid_in=0) freeze the counter — they are
//     invisible to the pooling logic.
//     Resets to 0 after the last row of the frame (row 713).
//     No modulo: last_row is a constant equality compare that
//     the synthesiser maps to a carry-chain equals check.
// ──────────────────────────────────────────────────────────────
localparam int ROW_BITS = $clog2(FRAME_ROWS);   // 10 bits

logic [ROW_BITS-1:0] row_cnt;
logic                last_row;

assign last_row = (row_cnt == ROW_BITS'(FRAME_ROWS - 1));

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        row_cnt <= '0;
    else if (valid_in)                        // only advance on valid data
        row_cnt <= last_row ? '0 : row_cnt + 1'b1;
    // valid_in=0 → counter holds, mid-frame gap is absorbed
end

// ──────────────────────────────────────────────────────────────
// 2.  Window position
//
//     STRIDE=4 is a power of two so bits [1:0] of row_cnt
//     naturally cycle 0→1→2→3→0 with no extra logic.
//     cnt is a pure wire — zero flip-flops, zero LUTs.
// ──────────────────────────────────────────────────────────────
localparam int CNT_BITS = $clog2(STRIDE);       // 2
localparam int EMIT_AT  = KERNEL - 1;           // cnt==1 triggers output

logic [CNT_BITS-1:0] cnt;
assign cnt = row_cnt[CNT_BITS-1:0];

// ──────────────────────────────────────────────────────────────
// 3.  Accumulator
//
//     8 registers, one per channel.
//     Written only when a valid row arrives at cnt==0
//     (first row of each pooling window).
//     Held stable on cnt==1,2,3 and during gaps.
// ──────────────────────────────────────────────────────────────
logic [DATA_W-1:0] acc [CHANNELS];

always_ff @(posedge clk) begin
    if (valid_in && cnt == '0)
        acc <= data_in;
end

// ──────────────────────────────────────────────────────────────
// 4.  Adder + ÷2  (combinational, 8 parallel lanes)
//
//     Sum is DATA_W+1 = 9 bits wide to hold up to 510 (255+255)
//     without overflow before the right-shift.
//     avg = sum[DATA_W:1] — divide by 2 is a free bit-select,
//     costs zero LUTs.
// ──────────────────────────────────────────────────────────────
logic [DATA_W:0]   sum [CHANNELS];
logic [DATA_W-1:0] avg [CHANNELS];

always_comb
    for (int c = 0; c < CHANNELS; c++) begin
        sum[c] = {1'b0, acc[c]} + {1'b0, data_in[c]};
        avg[c] = sum[c][DATA_W:1];
    end

// ──────────────────────────────────────────────────────────────
// 5.  Output register
//
//     Captures avg[] and asserts valid_out for exactly one cycle
//     when a valid row arrives at cnt==1 (second row of window).
//     All other cycles: valid_out=0, data_out holds last value.
// ──────────────────────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out <= 1'b0;
    end else begin
        valid_out <= valid_in && (cnt == CNT_BITS'(EMIT_AT));
        if (valid_in && cnt == CNT_BITS'(EMIT_AT))
            data_out <= avg;
    end
end

endmodule