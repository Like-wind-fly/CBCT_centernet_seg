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

import numpy as np
import torch
from shutil import copytree, ignore_patterns

from arguments import add_dataset_parameters
from torch.optim import lr_scheduler

from neural_parts_code.datasets.cbct.CBCT_dataset import BuildCBCTDataSet,CBCTDataset
from neural_parts_code.losses import get_loss
from neural_parts_code.metrics import get_metrics
from neural_parts_code.models import optimizer_factory, build_network
from neural_parts_code.stats_logger import StatsLogger
from utils import load_config

def set_num_threads(nt):
    nt = str(nt)
    os.environ["OPENBLAS_NUM_THREDS"] = nt
    os.environ["NUMEXPR_NUM_THREDS"] = nt
    os.environ["OMP_NUM_THREDS"] = nt
    os.environ["MKL_NUM_THREDS"] = nt


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def yield_infinite(iterable):
    while True:
        for item in iterable:
            yield item



def save_experiment_params(args, experiment_tag, directory):

    print(directory)
    if not os.path.isdir(directory):
        os.mkdir(directory)

    if not os.path.isdir(os.path.join(directory,'source_code')):
        copypath = os.getcwd()
        copypath = os.path.abspath(os.path.dirname(copypath)+"/.")
        ignore_path = os.path.join(copypath,"demo")+"/*"
        copytree(copypath, os.path.join(directory,'source_code'),
                 ignore=ignore_patterns('demo'))
    #
    # seed = trial_id + cfg.rank
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.enabled = True  # speeds up the computation
    #
    # return trial_save_path, trial_id


def load_checkpoints(model, optimizer, experiment_directory, args, device):
    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith("model_")
    ]
    if len(model_files) == 0:
        return
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(
        experiment_directory, "model_{:05d}"
    ).format(max_id)
    opt_path = os.path.join(experiment_directory, "opt_{:05d}").format(max_id)
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return

    print("Loading model checkpoint from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loading optimizer checkpoint from {}".format(opt_path))
    optimizer.optimizer.load_state_dict(
        torch.load(opt_path, map_location=device)
    )
    args.continue_from_epoch = max_id+1


def save_checkpoints(epoch, model, optimizer, experiment_directory, args):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    # The optimizer is wrapped with an object implementing gradient
    # accumulation
    torch.save(
        optimizer.optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a network to predict primitives"
    )
    parser.add_argument(
        "config_file",
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
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )

    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )

    add_dataset_parameters(parser)


    args = parser.parse_args(argv)
    set_num_threads(1)

    config = load_config(args.config_file)

    if torch.cuda.is_available():

        device = torch.device("cuda:{}".format(config["network"]["cuda"]))
        os.environ["CUDA_VISIBLE_DEVICES"] = config["network"]["cuda"]
        torch.cuda.set_device(device)

    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Get the parameters and their ordering for the spreadsheet
    save_experiment_params(args, experiment_tag, experiment_directory)
    # print("Save experiment statistics in {}".format(experiment_tag))

    # Log the training stats to a file
    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "stats.txt"),
        "a"
    ))

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    # Instantiate a dataloader to generate the samples for training
    datasets = BuildCBCTDataSet(config).quick_load_data()

    train_dataset = datasets["training"]
    train_dataset.readImageFromMen()
    Val_dataset = datasets["validation"]
    Val_dataset.readImageFromMen()

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= config["training"].get("batch_size") ,shuffle=True,drop_last=True)
    # Instantiate a dataloader to generate the samples for validation
    val_dataloader = torch.utils.data.DataLoader(Val_dataset, batch_size=config["validation"].get("batch_size") ,drop_last=True)

    epochs = config["training"].get("epochs", 150)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(
        config, args.weight_file, device=device
    )
    # Build an optimizer object to compute the gradients of the parameters
    optimizer =  optimizer_factory(config["training"], network.parameters())
    load_checkpoints(network, optimizer, experiment_directory, args, device)

    tmax = config["training"]["T_max"]
    scheduler =  lr_scheduler.CosineAnnealingLR(optimizer.optimizer, tmax, eta_min=1e-7, last_epoch=-1, verbose=False)

    # Create the loss and metrics functions
    loss_fn = get_loss(config["loss"]["type"], config["loss"])
    metrics_fn = get_metrics(config["metrics"])

    for i in range(args.continue_from_epoch, epochs):
        network.train()

        for b, sample in enumerate(dataloader):
            X = sample["ct"].to(device)
            # targets = [yi.to(device) for yi in sample[1:]] zechu code
            targets = {key: value.to(device) for key, value in sample.items() if key != 'ct'}
            # Train on batch
            batch_loss = train_on_batch(
                network, optimizer, loss_fn, metrics_fn, X, targets, config
            )

            # Print the training progress
            StatsLogger.instance().print_progress(i+1, b+1, batch_loss)

        scheduler.step()

        if i % save_every == 0:
            save_checkpoints(
                i,
                network,
                optimizer,
                experiment_directory,
                args
            )
        StatsLogger.instance().clear()

        # if True:
        if i % val_every == 0 and i > 0:
            with torch.no_grad():
                print("====> Validation Epoch ====>")

                network.eval()
                for b, sample in enumerate(val_dataloader):
                    # for b, sample in zip(range(len(val_dataloader)), val_dataloader):

                    # X = sample[0].to(device)
                    # targets = [yi.to(device) for yi in sample[1:]]
                    # 修改的代码
                    X = sample["ct"].to(device)
                    targets = {key: value.to(device) for key, value in sample.items() if key != 'ct'}
                    # Validate on batch
                    batch_loss = validate_on_batch(
                        network, loss_fn, metrics_fn, X, targets, config
                    )
                    # Print the training progress
                    StatsLogger.instance().print_progress(1, b + 1, batch_loss)

                torch.cuda.empty_cache()
                StatsLogger.instance().clear()
                print("====> Validation Epoch ====>")

    print("Saved statistics in {}".format(experiment_tag))


if __name__ == "__main__":
    # main(sys.argv[1:])
    print(torch.cuda.is_available())
    argv = ["python train_network.py", "../config/centernet.yaml", "../demo/output" , "--experiment_tag","centerNet_blockNum_2"]

    main(argv[1:])
