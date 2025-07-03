import tensorrt as trt
import sys
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, fp16=True, max_workspace_size=(1 << 30)):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Loading ONNX model from {onnx_file_path}...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX model:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Parsing failed.")

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled.")
        else:
            print("FP16 not supported on this platform.")

    print("🔧 Building engine...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build engine.")

    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"Engine saved to: {engine_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/quantize_model.py <input.onnx> <output.trt>")
        sys.exit(1)

    onnx_path = sys.argv[1]
    engine_path = sys.argv[2]
    build_engine(onnx_path, engine_path)
