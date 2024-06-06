"""Script used to eval neural parts."""
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
            for b,sample in enumerate(dataloader):
                keys = list(sample.keys())  #['ct', 'seg', 'heatmap', 'heatmap32', 'reg_mask', 'index_int', 'reg', 'dwh']

                print(keys)
                X = sample["ct"].to(device)
                targets = sample

                # Validate on batch
                batch_loss = validate_on_batch(
                    network, loss_fn, metrics_fn, X, targets, config
                )
                print(X.shape)
                predictions = network(X)
                save_img(sample["heatmap"][0, 0].cpu().numpy(),os.path.join(args.weight_file, "our_heatmap_{}.nii".format(b)))
                save_img(predictions["heatmap"][0, 0].cpu().numpy(),os.path.join(args.weight_file, "our_predict_heatmap_{}.nii".format(b)))
                print("heatmap.shape:", predictions["reg"][0, 0].shape)
                #save_img(prediction["reg"][0, 0].cpu().numpy(),os.path.join(args.weight_file, "our_predict_reg_{}.nii".format(b)))

                print("reg.shape:", sample["reg"].cpu().numpy().shape)
                print("dwh.shape:", sample["dwh"].cpu().numpy().shape)
                save_img(sample["ct"][0, 0].cpu().numpy(), os.path.join(args.weight_file, "our_ct_{}.nii".format(b)))
                save_img((torch.round(predictions["heatmap"])*2000-1000).cpu().numpy(),
                         os.path.join(args.weight_file, "heatmap_{}.nii".format(b)))

                StatsLogger.instance().print_progress(int(model_save_path[6:]), b + 1, batch_loss)

        StatsLogger.instance().clear()

if __name__ == "__main__":
    argv = ["python train_network.py", "../config/centernet.yaml", "../demo/output" , "--weight_file","/mnt/fourT/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/mix_centernet/output/centerNet_blockNum_2"]
    # argv = ["python train_network.py", "../config/centernet_Unet_layer3.yaml", "../demo/output" , "--weight_file","/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/centerNet_mini"]
    main(argv[1:])
