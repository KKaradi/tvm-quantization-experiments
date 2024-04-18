from tvm.contrib import graph_executor
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
import numpy as np
import torch
import torchvision
from tvm.relay.quantize import quantize, qconfig, current_qconfig
QUANTIZE_CONFIGS = [
    {
        "nbit_input": 1,
        "nbit_weight": 1,
        "nbit_activation": 32,
        "dtype_input": "int8",
        "dtype_weight": "int8",
        "dtype_activation": "int32",
        "calibrate_mode": "global_scale",
        "global_scale": 8.0,
        "weight_scale": "power2",
        "skip_dense_layer": True,
        "skip_conv_layers": [0],
        "do_simulation": False,
        "round_for_shift": True,
        "debug_enabled_ops": None,
        "rounding": "UPWARD",
        "calibrate_chunk_by": -1,
        "partition_conversions": "disabled",
    }
]

# QUANTIZE_CONFIGS = [
#     None, {
#         "nbit_input": 8,
#         "nbit_weight": 8,
#         "nbit_activation": 32,
#         "dtype_input": "int8",
#         "dtype_weight": "int8",
#         "dtype_activation": "int32",
#         "calibrate_mode": "global_scale",
#         "global_scale": 8.0,
#         "weight_scale": "power2",
#         "skip_dense_layer": True,
#         "skip_conv_layers": [0],
#         "do_simulation": False,
#         "round_for_shift": True,
#         "debug_enabled_ops": None,
#         "rounding": "UPWARD",
#         "calibrate_chunk_by": -1,
#         "partition_conversions": "disabled",
#     }, {
#         "nbit_input": 4,
#         "nbit_weight": 4,
#         "nbit_activation": 32,
#         "dtype_input": "int8",
#         "dtype_weight": "int8",
#         "dtype_activation": "int32",
#         "calibrate_mode": "global_scale",
#         "global_scale": 8.0,
#         "weight_scale": "power2",
#         "skip_dense_layer": True,
#         "skip_conv_layers": [0],
#         "do_simulation": False,
#         "round_for_shift": True,
#         "debug_enabled_ops": None,
#         "rounding": "UPWARD",
#         "calibrate_chunk_by": -1,
#         "partition_conversions": "disabled",
#     },
#     {
#         "nbit_input": 2,
#         "nbit_weight": 2,
#         "nbit_activation": 32,
#         "dtype_input": "int8",
#         "dtype_weight": "int8",
#         "dtype_activation": "int32",
#         "calibrate_mode": "global_scale",
#         "global_scale": 8.0,
#         "weight_scale": "power2",
#         "skip_dense_layer": True,
#         "skip_conv_layers": [0],
#         "do_simulation": False,
#         "round_for_shift": True,
#         "debug_enabled_ops": None,
#         "rounding": "UPWARD",
#         "calibrate_chunk_by": -1,
#         "partition_conversions": "disabled",
#     },
#     {
#         "nbit_input": 1,
#         "nbit_weight": 1,
#         "nbit_activation": 32,
#         "dtype_input": "int8",
#         "dtype_weight": "int8",
#         "dtype_activation": "int32",
#         "calibrate_mode": "global_scale",
#         "global_scale": 8.0,
#         "weight_scale": "power2",
#         "skip_dense_layer": True,
#         "skip_conv_layers": [0],
#         "do_simulation": False,
#         "round_for_shift": True,
#         "debug_enabled_ops": None,
#         "rounding": "UPWARD",
#         "calibrate_chunk_by": -1,
#         "partition_conversions": "disabled",
#     },
#     {
#         "nbit_input": 16,
#         "nbit_weight": 16,
#         "nbit_activation": 32,
#         "dtype_input": "int16",
#         "dtype_weight": "int16",
#         "dtype_activation": "int32",
#         "calibrate_mode": "global_scale",
#         "global_scale": 8.0,
#         "weight_scale": "power2",
#         "skip_dense_layer": True,
#         "skip_conv_layers": [0],
#         "do_simulation": False,
#         "round_for_shift": True,
#         "debug_enabled_ops": None,
#         "rounding": "UPWARD",
#         "calibrate_chunk_by": -1,
#         "partition_conversions": "disabled",
#     },
#     {
#         "nbit_input": 4,
#         "nbit_weight": 4,
#         "nbit_activation": 32,
#         "dtype_input": "int4",
#         "dtype_weight": "int4",
#         "dtype_activation": "int32",
#         "calibrate_mode": "global_scale",
#         "global_scale": 8.0,
#         "weight_scale": "power2",
#         "skip_dense_layer": True,
#         "skip_conv_layers": [0],
#         "do_simulation": False,
#         "round_for_shift": True,
#         "debug_enabled_ops": None,
#         "rounding": "UPWARD",
#         "calibrate_chunk_by": -1,
#         "partition_conversions": "disabled",
#     },
#     {
#         "nbit_input": 2,
#         "nbit_weight": 2,
#         "nbit_activation": 32,
#         "dtype_input": "int2",
#         "dtype_weight": "int2",
#         "dtype_activation": "int32",
#         "calibrate_mode": "global_scale",
#         "global_scale": 8.0,
#         "weight_scale": "power2",
#         "skip_dense_layer": True,
#         "skip_conv_layers": [0],
#         "do_simulation": False,
#         "round_for_shift": True,
#         "debug_enabled_ops": None,
#         "rounding": "UPWARD",
#         "calibrate_chunk_by": -1,
#         "partition_conversions": "disabled",
#     },
#     {
#         "nbit_input": 1,
#         "nbit_weight": 1,
#         "nbit_activation": 32,
#         "dtype_input": "int1",
#         "dtype_weight": "int1",
#         "dtype_activation": "int32",
#         "calibrate_mode": "global_scale",
#         "global_scale": 8.0,
#         "weight_scale": "power2",
#         "skip_dense_layer": True,
#         "skip_conv_layers": [0],
#         "do_simulation": False,
#         "round_for_shift": True,
#         "debug_enabled_ops": None,
#         "rounding": "UPWARD",
#         "calibrate_chunk_by": -1,
#         "partition_conversions": "disabled",
#     }
# ]


model_name = "resnet50"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

input_data = np.random.randn(*input_shape).astype("float32")
input_name = "input0"
shape_list = [(input_name, input_data.shape)]

dev = tvm.cpu(0)


for QUANTIZE_CONFIG in QUANTIZE_CONFIGS:
    try:
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        trial_name = "no_quantization" if QUANTIZE_CONFIG == None else ("nbit" +
                                                                        str(QUANTIZE_CONFIG['nbit_input'])+"_dtype"+str(QUANTIZE_CONFIG['dtype_input']))

        print(f"Running config {trial_name}")

        if (QUANTIZE_CONFIG != None):
            with qconfig(**QUANTIZE_CONFIG):
                print(current_qconfig())
                mod_quantized = quantize(mod, params)

                opt_level = 3
                target = tvm.target.Target('llvm')

                with tvm.transform.PassContext(opt_level=opt_level):
                    lib = relay.build(mod_quantized, target, params=params)
                    graph, src, params = lib
                    src_code = src.get_source()

                    with open(f'./source_code/{trial_name}', 'w') as file:
                        file.write(str(src_code))

                m = graph_executor.GraphModule((lib["default"](dev)))
                m.set_input(input_name, tvm.nd.array(input_data))

                benchmark = m.benchmark(dev, "run", 10, 10)
                print(benchmark)
                with open('benchmarks.txt', 'a') as file:
                    file.write(trial_name+"\n")
                    file.write(str(benchmark)+"\n")
        else:
            opt_level = 3
            target = tvm.target.Target('llvm')

            with tvm.transform.PassContext(opt_level=opt_level):
                lib = relay.build(mod, target, params=params)
                graph, src, params = lib
                src_code = src.get_source()

                with open(f'./source_code/{trial_name}', 'w') as file:
                    file.write(str(src_code))

            m = graph_executor.GraphModule((lib["default"](dev)))
            m.set_input(input_name, tvm.nd.array(input_data))

            benchmark = m.benchmark(dev, "run", 10, 10)
            print(benchmark)
            with open('benchmarks.txt', 'a') as file:
                file.write(trial_name+"\n")
                file.write(str(benchmark)+"\n")
    except Exception as e:
        print("Error")
        print(e)
