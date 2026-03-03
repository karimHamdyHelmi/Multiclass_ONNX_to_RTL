# Multiclass ONNX to RTL

Convert multiclass ONNX models (fully connected layers) to RTL and `.mem` files. Supports 3+ output classes (e.g. MNIST 10, CIFAR-100). Uses `detect_quant_type.py` for autodetection of quantization (int4, int8, int16).

**Softmax support**: When the ONNX model has a Softmax layer after the final FC, the RTL outputs probabilities (0.16 format, 1.0 = 65535) instead of raw logits. Use `--force-softmax` to add softmax even when the ONNX does not have it.

## Requirements

- **Python 3.8+**
- **PyTorch** for rtl_mapper
- **ONNX** from `convert_model_to_RTL/onnx_lib`
- **convert_model_to_RTL** sibling directory with `rtl_mapper`

## ONNX to RTL

```bash
# Convert multiclass ONNX to RTL + .mem (quantization autodetected)
python multiclass_onnx_to_rtl.py --onnx-model path/to/model.onnx --out-dir ./my_ip

# With testbench
python multiclass_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --emit-testbench

# Flattened RTL structure (single inlined module)
python multiclass_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --rtl-structure flattened

# Override quantization
python multiclass_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --weight-format int8

# Force softmax (output probabilities even when ONNX has no Softmax)
python multiclass_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --force-softmax
```

## Quantization Detection

`multiclass_onnx_to_rtl.py` uses `detect_quant_type.py` in the same folder:

```bash
# From ONNX model
python detect_quant_type.py --onnx-model path/to/model.onnx

# From .mem files
python detect_quant_type.py --mem-dir path/to/mem/files

# From checkpoint
python detect_quant_type.py --checkpoint path/to/model.pth
```

## Supported Models

- **FC only**: Gemm, MatMul, QLinearMatMul, QLinearGemm
- **CNN to FC**: MatMul after Reshape
- **Output**: 3+ classes (e.g. MNIST 10, CIFAR-10, CIFAR-100)
- **Softmax**: Auto-detected when ONNX has Softmax after final FC; use `--force-softmax` to add it manually

## Output

- `src/rtl/systemverilog/` RTL modules, quant_pkg, ROMs
- `src/rtl/systemverilog/mem/` weight and bias `.mem` files
- `mapping_report.txt`, `netlist.json`
