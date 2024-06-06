"""Script used to train neural parts."""
import argparse

import json
import random
import os
import string
import subprocess
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")


import torch
from torch.utils.data import DataLoader
from arguments import add_dataset_parameters
from neural_parts_code.losses import get_loss,eval_loss
from neural_parts_code.metrics import get_metrics
from neural_parts_code.models import  build_network
from neural_parts_code.stats_logger import StatsLogger
from utils import load_config
from neural_parts_code.datasets import build_datasets
from neural_parts_code.models.sgaNet import  compute_predictions_from_features
from neural_parts_code.PostProcess.SGA_visualize import get_meshes
from neural_parts_code.datasets.cbct.common import save_img
import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np
import math
import nibabel as nib
def main(argv):
    parser = argparse.ArgumentParser(
        description="Do the forward pass and estimate a set of primitives"
    )
    parser.add_argument(
        "config_file",
        default="../config/default.yaml",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )

    add_dataset_parameters(parser)
    args = parser.parse_args(argv)

    config = load_config(args.config_file)
    if torch.cuda.is_available():

        device = torch.device("cuda:{}".format(config["network"]["cuda"]))
        os.environ["CUDA_VISIBLE_DEVICES"] = config["network"]["cuda"]
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    StatsLogger.instance().add_output_file(open(
        os.path.join( "stats.txt"),
        "a"
    ))

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    model_list = os.listdir(args.weight_file)
    model_list = [modelname  for modelname in model_list if "model" in modelname]
    model_list.sort()


    # Instantiate a dataloader to generate the samples for evaluation
    train_dataset, Val_dataset, test_dataset = build_datasets(config)
    # Val_dataset.readImageFromMen()

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["testing"].get("batch_size"),drop_last=True)

    loss_fn = get_loss(config["loss"]["type"], config["loss"])
    eval_loss_fn = eval_loss(config["loss"]["type"], config["loss"])
    metrics_fn = get_metrics(config["metrics"])

    for model_save_path in model_list:
        # Build the network architecture to be used for training

        model_id = int(model_save_path[6:])
        # if model_id <20:
        #     continue
        # if model_id != 1600:
        if model_id != 200:
            continue
        # network, train_on_batch, validate_on_batch = build_network(
        #     config, os.path.join(args.weight_file, model_save_path), device=device
        # )
        network, train_on_batch, validate_on_batch = build_network(
            config, "/home/yisiinfo/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/old_demo/output/sga_net/model_02990", device=device
        )
        new_network, train_on_batch, validate_on_batch = build_network(
            config, "/home/yisiinfo/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/demo/output/sga_net/model_00620",
            device=device
        )
        network.quad_seg = new_network.quad_seg
        network.eval()

        # Create the prediction input
        with torch.no_grad():
            ct_path = "/home/yisiinfo/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/scripts/cropped_heatmap.nii.gz"
            ct = sitk.ReadImage(ct_path, sitk.sitkFloat32)
            ct_image = sitk.GetArrayFromImage(ct)
            # ct_image = nib.load(ct_path)
            save_img(ct_image, os.path.join(args.weight_file, "ct_cropyuan{}.nii".format(2)))

            # 归一化
            # ct_image = np.flip(ct_image, axis=(1, 2))
            # ct_image[ct_image > 2500] = 2500
            # ct_image[ct_image < -500] = -500
            # ct_image = ct_image - 1000
            # ct_image = ct_image / 1500

            #统一resize
            max_slice = max(ct_image.shape[1], ct_image.shape[2],ct_image.shape[0])
            zoom_size = 256 / max_slice
            if ct_image.shape[0]*zoom_size >160:
                zoom_size = 160/ct_image.shape[0]
                print(zoom_size)
            resized_ct_image = zoom(ct_image, zoom=(zoom_size, zoom_size, zoom_size), order=3)

            #填补
            min_size = [160, 256, 256]
            pad = np.array(min_size) - np.array(resized_ct_image.shape)
            pad[pad <= 0] = 0
            pading = []
            for i in range(3):
                left = math.floor(pad[i] / 2.0)
                right = math.ceil(pad[i] / 2.0)
                pading.append((left, right))
            resized_ct_image=np.pad(resized_ct_image, pading, 'constant', constant_values=0)

            X = {"quad_ct":torch.from_numpy(resized_ct_image).cuda().unsqueeze(0).unsqueeze(0).float()}
            print(X['quad_ct'].shape)
            predictions = compute_predictions_from_features(network, X, None, config)
            print(predictions['get_id'].shape)
            save_img(X["quad_ct"].squeeze(0).squeeze(0).cpu().numpy(), os.path.join(args.output_directory, "ct_crop003{}.nii".format(1)))
            save_img(predictions["volume"][0,0].int().cpu().numpy(), os.path.join(args.output_directory, "seg_volume_crop003{}.nii".format(1)))
            # save_img(predictions['get_id'][0,0].int().cpu().numpy(), os.path.join(args.weight_file, "get_id{}.nii".format(1)))
            # save_img(predictions['get_id'][1,0].int().cpu().numpy(), os.path.join(args.weight_file, "get_id{}.nii".format(2)))
            # save_img(predictions['get_id'][2,0].int().cpu().numpy(), os.path.join(args.weight_file, "get_id{}.nii".format(3)))
            # save_img(predictions['get_id'][3,0].int().cpu().numpy(), os.path.join(args.weight_file, "get_id{}.nii".format(4)))
            save_img(predictions['seg_predcition'][0, 0].int().cpu().numpy(),
                     os.path.join(args.weight_file, "seg_crop_predcition_001{}.nii".format(1)))
            save_img(predictions['seg_predcition'][0, 1].int().cpu().numpy(),
                     os.path.join(args.weight_file, "seg_crop_predcition_002{}.nii".format(1)))
            save_img(predictions['seg_predcition'][0, 2].int().cpu().numpy(),
                     os.path.join(args.weight_file, "seg_crop_predcition_003{}.nii".format(1)))
            save_img(predictions['seg_predcition'][0, 3].int().cpu().numpy(),
                     os.path.join(args.weight_file, "seg_crop_predcition_004{}.nii".format(1)))

            print("1111")
            # StatsLogger.instance().print_progress(int(model_save_path[6:]), b + 1, batch_loss)

        StatsLogger.instance().clear()

if __name__ == "__main__":
    argv = ["python train_network.py", "../config/sga_net.yaml", "../demo/output", "--weight_file",
            "/home/yisiinfo/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/demo/output/sga_net"]
    main(argv[1:])
