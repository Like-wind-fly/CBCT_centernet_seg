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
import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np
import nibabel as nib
import math
from neural_parts_code.datasets.cbct.common import save_img
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

    dataloader = torch.utils.data.DataLoader(Val_dataset, batch_size=config["testing"].get("batch_size"),drop_last=True)

    loss_fn = get_loss(config["loss"]["type"], config["loss"])
    eval_loss_fn = eval_loss(config["loss"]["type"], config["loss"])
    metrics_fn = get_metrics(config["metrics"])



    for model_save_path in model_list:
        # Build the network architecture to be used for training
        model_id = int(model_save_path[6:])
        # if model_id <20:
        #     continue
        if model_id != 2970:
            continue
        network, train_on_batch, validate_on_batch = build_network(
            config, os.path.join(args.weight_file, model_save_path), device=device
        )
        network.eval()

        # Create the prediction input
        with torch.no_grad():
            ct_path = "/home/yisiinfo/cyj/cj/segmentation_teeth_3d/our_cbct_data/nii/1.nii.gz"
            ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
            ct_image = sitk.GetArrayFromImage(ct)
            ct_image[ct_image > 2500] = 2500
            ct_image[ct_image < -500] = -500
            ct_image = ct_image - 1000
            ct_image = ct_image / 1500
            ct_image = ct_image[0:ct_image.shape[0]//2,:,:]
            new_shape = (160,256,256)
            max_slice = max(ct_image.shape[1], ct_image.shape[2], ct_image.shape[0])
            zoom_size = 256 / max_slice
            if ct_image.shape[0] * zoom_size > 160:
                zoom_size = 160 / ct_image.shape[0]
            print(zoom_size)

            resized_ct_image = zoom(ct_image, zoom=(zoom_size, zoom_size, zoom_size), order=3)
            min_size = [160, 256, 256]
            pad = np.array(min_size) - np.array(resized_ct_image.shape)
            pad[pad <= 0] = 0
            pading = []
            for i in range(3):
                left = math.floor(pad[i] / 2.0)
                right = math.ceil(pad[i] / 2.0)
                pading.append((left, right))
            # if resized_ct_image.shape[0] >160:
            #     resized_ct_image = resized_ct_image[0:160,:,:]
            resized_ct_image=np.pad(resized_ct_image, pading, 'constant', constant_values=0)
            save_img(resized_ct_image, os.path.join(args.weight_file, "resized_ct_image{}.nii".format(1)))
            print(resized_ct_image.shape)
            X = torch.from_numpy(resized_ct_image).cuda().unsqueeze(0).unsqueeze(0).float()
            predictions = network(X)
            round_heatmap=predictions["heatmap"]
            print(round_heatmap.shape)
            round_heatmap[round_heatmap>0.5]=1
            round_heatmap[round_heatmap<0.5]=0

            save_img(round_heatmap[0,0].cpu(), os.path.join(args.weight_file, "round_heatmap{}.nii".format(1)))
            # round_heatmap = torch.round(predictions["heatmap"]) #0-1,1


            ######
            # 获取值为 1 的元素的索引
            # 检查是否存在值为 1 的元素
            if (round_heatmap == 1).any():
                # 获取值为 1 的元素的索引
                indices = torch.where(round_heatmap.cpu() == 1)

                # 将索引转换为 NumPy 数组，方便后续操作
                indices_np = [ind.numpy() for ind in indices]

                # 找出每个维度上的最小值和最大值
                min_depth = np.min(indices_np[2])
                max_depth = np.max(indices_np[2])
                min_width = np.min(indices_np[3])
                max_width = np.max(indices_np[3])
                min_height = np.min(indices_np[4])
                max_height = np.max(indices_np[4])

                # 左上角坐标
                top_left = (int(min_depth//zoom_size*4), int(min_width//zoom_size*4), int(min_height//zoom_size*4))
                # 右下角坐标
                bottom_right = (int(max_depth//zoom_size*4), int(max_width//zoom_size*4), int(max_height//zoom_size*4))

                print(f"左上角坐标：{top_left}")
                print(f"右下角坐标：{bottom_right}")
                print(ct_image.shape)
                # 裁剪 round_heatmap
                cropped_heatmap = ct_image[
                                  ct_image.shape[0]-bottom_right[0]-20:ct_image.shape[0]-top_left[0]+30,
                                  top_left[1]-50:bottom_right[1]+50,
                                  top_left[2]-50:bottom_right[2]+50,
                                  ]

                # 保存为 NIfTI 文件
                print(cropped_heatmap.shape)
                save_img(cropped_heatmap,
                    "cropped_heatmap.nii.gz"
                )

            else:
                print("round_heatmap 中不存在值为 1 的元素。")

            print("*********************")

            # # 计算反向缩放比例
            # inverse_scale_factors = [new_dim / old_dim for new_dim, old_dim in
            #                          zip(new_shape, predictions["heatmap"].shape)]

            # if (resized_round_heatmap == 1).any():
            #     # 获取值为 1 的元素的索引
            #     indices = torch.where(resized_round_heatmap.cpu() == 1)
            #
            #     # 将索引转换为 NumPy 数组，方便后续操作
            #     indices_np = [ind.numpy() for ind in indices]
            #
            #     # 找出每个维度上的最小值和最大值
            #     min_depth = np.min(indices_np[2])
            #     max_depth = np.max(indices_np[2])
            #     min_width = np.min(indices_np[3])
            #     max_width = np.max(indices_np[3])
            #     min_height = np.min(indices_np[4])
            #     max_height = np.max(indices_np[4])
            #
            #     # 左上角坐标
            #     top_left = (min_depth, min_width, min_height)
            #     # 右下角坐标
            #     bottom_right = (max_depth, max_width, max_height)
            #
            #     print(f"左上角坐标：{top_left}")
            #     print(f"右下角坐标：{bottom_right}")
            # # 裁剪 ct_image
            # cropped_ct_image = ct_image[
            #                    min_depth+30:max_depth + 200,
            #                    min_width+100:max_width + 300,
            #                    min_height+120:max_height + 420,
            #                    ]
            # print(cropped_ct_image.shape)
            # 保存裁剪后的 CT 图像
            # cropped_ct_image_sitk = sitk.GetImageFromArray(cropped_ct_image)
            # sitk.WriteImage(cropped_ct_image_sitk, "cropped_ct_image.nii.gz")




            # scale_factors = [new_dim / old_dim for new_dim, old_dim in zip(new_shape, predictions["heatmap"].shape)]
            # predictions["heatmap"] = zoom(predictions["heatmap"].cpu(), zoom=scale_factors, order=3)
            save_img(resized_ct_image, os.path.join(args.weight_file, "ct_0{}.nii".format(1)))
            save_img(torch.round(predictions["heatmap"]).cpu().numpy(),
                     os.path.join(args.weight_file, "heatmap_0{}.nii".format(1)))
if __name__ == "__main__":
    argv = ["python train_network.py", "../config/centernet.yaml", "../demo/output" , "--weight_file","/home/yisiinfo/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/demo/output/centerNet_blockNum_2"]
    # argv = ["python train_network.py", "../config/centernet_Unet_layer3.yaml", "../demo/output" , "--weight_file","/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/centerNet_mini"]
    main(argv[1:])
