"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse

from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def run(input_path, output_path, model_path, model_type="large", optimize=True, save_npy=True, save_png=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large": # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize==True:
        # rand_example = torch.rand(1, 3, net_h, net_w)
        # model(rand_example)
        # traced_script_module = torch.jit.trace(model, rand_example)
        # model = traced_script_module

        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

    model.to(device)

    # get input
    img_names = glob.glob(os.path.join(input_path, "im_*.jpg"))
    num_images = len(img_names)

    # create output folder
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input
        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize==True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # output

        if save_png:
            filename = os.path.join(
                output_path, os.path.splitext(os.path.basename(img_name))[0]
            )
            utils.write_depth(filename, prediction, bits=2)

        if save_npy:
            base_image_name = os.path.splitext(os.path.basename(img_name))[0]
            base_image_name = base_image_name.replace("im_", "dm_midas_")
            filename = os.path.join(output_path, base_image_name)
            utils.write_npy(filename, prediction)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path',
        default='input',
        help='folder with input images',
        required=True
    )
    parser.add_argument('-o', '--output_path',
        help='folder for output image'
    )
    parser.add_argument('-m', '--model_weights',
        default=None,
        help='path to the trained weights of model'
    )
    parser.add_argument('-t', '--model_type',
        default='dpt_large',
        help='model type: dpt_large, dpt_hybrid, midas_v21_large or midas_v21_small'
    )
    parser.add_argument('--save_npy',
        type=utils.str_to_bool,
        required=False,
        default=True,
        help='True if you want to save the infered depth to a .npy file.'
    )
    parser.add_argument('--save_png',
        type=utils.str_to_bool,
        required=False,
        default=True,
        help='True if you want to save the infered depth to a .png file.'
    )
    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false')
    parser.set_defaults(optimize=True)

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path if args.output_path else input_path

    # set extension to save depth files.
    save_npy = args.save_npy
    save_png = args.save_png

    default_models = {
        "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt",
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(args.input_path, output_path, args.model_weights, args.model_type, args.optimize, save_npy, save_png)
