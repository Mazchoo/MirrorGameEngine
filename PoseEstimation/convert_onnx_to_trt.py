import tensorrt as trt


def build_engine(input_path: str, output_path: str):
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    # we have only one image in batch
    network = builder.create_network()

    parser = trt.OnnxParser(network, trt_logger)
    with open(input_path, "rb") as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    config = builder.create_builder_config()

    serialized_network = builder.build_serialized_network(network, config)
    with open(output_path, "wb") as f:
        f.write(serialized_network)

    return serialized_network


if __name__ == "__main__":
    build_engine(
        "./PoseEstimation/models/onyx.onnx", "./PoseEstimation/models/turtwig.trt"
    )
