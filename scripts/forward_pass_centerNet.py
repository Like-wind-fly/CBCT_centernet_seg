"""Script used to train neural parts."""
import argparse

import os
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")

import torch


from torch.utils.data import DataLoader

from arguments import add_dataset_parameters

from neural_parts_code.datasets import build_datasets
from neural_parts_code.models import build_network
from neural_parts_code.models.Corticalflow import  compute_predictions_from_features
from utils import load_config
from neural_parts_code.datasets.cbct.common import save_img
from neural_parts_code.models.boxDecode import  box_decode



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

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(
        config, args.weight_file, device=device
    )
    network.eval()

    # Instantiate a dataloader to generate the samples for evaluation
    _,_,test_dataset = build_datasets(config)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["testing"].get("batch_size"),
                                                 drop_last=True)

    # Create the prediction input
    with torch.no_grad():
        for k,sample in enumerate(dataloader):
            X = sample[0].to(device)

            targets = [yi.to(device) for yi in sample[1:]]
            B = config["testing"].get("batch_size")

            predictions = network(X)
            pred_hm = predictions["heatmap"]
            pred_reg = predictions["reg"]
            pred_dwh = predictions["dwh"]

            # ct[None],  # cbct的影像      1, D，H, W
            # seg[None],  # 标注的分割的图    1, D，H, W
            # heatmap_tensor,  # heatmap         c, D, H, W
            # reg_mask,  # 用来标注到底出现了多少实例，   32
            # index_int,  # 用来标注每一个实例的中心     32,3
            # reg,  # 下采样导致的偏差            32,3

            ct = X
            gt_hm = targets[1]
            gt_reg_mask = targets[2]
            gt_index_int = targets[3]
            gt_reg = targets[4]

            bboxes, scores, clses = box_decode(pred_hm, pred_reg, pred_dwh)


            for i in range(B):
                save_img(ct[i,0].cpu().numpy(), os.path.join(args.output_directory,"{}_{}_gt_ct.nii".format(k,i)))
                save_img(gt_hm[i,0].cpu().numpy(), os.path.join(args.output_directory,"{}_{}_gt_hm.nii".format(k,i)))
                save_img(pred_hm[i,0].cpu().numpy(), os.path.join(args.output_directory,"{}_{}_pred_hm.nii".format(k,i)))

                box = box = torch.zeros(ct[i,0].shape).cuda().int()
                for boxId in range(len(bboxes[i])):
                    exist_ = scores[i][boxId] > 0.45
                    bound = bboxes[i][boxId].int()
                    if exist_:
                        box[bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]] = 1
                save_img(box.cpu().numpy(),
                         os.path.join(args.output_directory, "{}_{}_pred_box.nii".format(k, i)))


if __name__ == "__main__":
    # print(sys.argv)
    # main(sys.argv[1:])
    # argv = ['forward_pass_CBCT.py', '../config/centernet.yaml',
    #         '/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/centerNet/output', '--weight_file',
    #         '/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/centerNet/model_01530']
    #
    argv = ['forward_pass_CBCT.py', '../config/centernet.yaml',
            '/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/centerNet_blockNum_2/output', '--weight_file',
            '/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/centerNet_blockNum_2/model_02730']
    main(argv[1:])