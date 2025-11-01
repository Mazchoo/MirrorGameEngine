import torch

from PoseEstimation.mobilenet import PoseEstimationWithMobileNet
import collections

INPUT_PATH = "./PoseEstimation/models/checkpoint.pth"
OUTPUT_PATH = "./PoseEstimation/models/onyx.onnx"
INPUT_HEIGHT = 210
INPUT_WIDTH = 280


def load_state(net, checkpoint):
    source_state = checkpoint["state_dict"]
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key in target_state.keys():
        if (
            target_key in source_state
            and source_state[target_key].size() == target_state[target_key].size()
        ):
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print(
                "[WARNING] Not found pre-trained parameters for {}".format(target_key)
            )

    net.load_state_dict(new_target_state)


def convert_to_onnx(net, output_path):
    input = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH).cuda()
    input_names = ["data"]
    output_names = [
        "stage_0_output_1_heatmaps",
        "stage_0_output_0_pafs",
        "stage_1_output_1_heatmaps",
        "stage_1_output_0_pafs",
    ]

    torch.onnx.export(
        net,
        input,
        output_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
    )


if __name__ == "__main__":
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(INPUT_PATH)
    load_state(net, checkpoint)
    net = net.cuda()

    convert_to_onnx(net, OUTPUT_PATH)
