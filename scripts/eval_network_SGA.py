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
        if model_id != 1600:
            continue
        network, train_on_batch, validate_on_batch = build_network(
            config, os.path.join(args.weight_file, model_save_path), device=device
        )
        network.eval()

        # Create the prediction input
        with torch.no_grad():
            for b,sample in enumerate(dataloader):
                keys = list(sample.keys())
                X = {key: sample[key] for key in keys[:2]}
                targets = sample
                # Validate on batch
                # batch_loss = validate_on_batch(
                #     network, loss_fn, metrics_fn, X, targets, config
                # )
                predictions = compute_predictions_from_features(network, X, targets, config)
                x_classify, x_gt_meshes, x_pred_meshes = get_meshes(predictions, targets)

                pred_verts = x_pred_meshes[0][0]
                pred_faces = x_pred_meshes[0][1]

                torch.save([pred_verts, pred_faces, x_classify[0].int()], os.path.join(args.weight_file, "all_teeth_{}.pt".format(b)))
                print(os.path.join(args.weight_file, "all_teeth_{}.pt".format(b)))
                torch.save(predictions["volume"][0,0],os.path.join(args.weight_file, "volume_{}.pt".format(b)))

                save_img(predictions["volume"][0,0].int().cpu().numpy(), os.path.join(args.weight_file, "volume_{}.nii".format(b)))

                # StatsLogger.instance().print_progress(int(model_save_path[6:]), b + 1, batch_loss)

        StatsLogger.instance().clear()

if __name__ == "__main__":
    argv = ["python train_network.py", "../config/sga_net.yaml", "../demo/output", "--weight_file",
            "/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/sga_net"]
    main(argv[1:])
