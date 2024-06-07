import torch
from .cbct.CBCT_dataset import BuildCBCTDataSet
from .cbct.CBCT_crop_dataset import BuildCBCTDataSet as BuildCropDataSet
from .cbct.CBCT_SGA_dataset import BuildCBCTSGADataSet
from .SampleMeshs import  BuildSMDataSet
from .SampleMeshs import BuildSMDataSet

def build_datasets(config):
    dataset_type = config["data"]["dataset_type"]

    if dataset_type == "CBCT":
        datasets = BuildCBCTDataSet(config).quick_load_data()
        train_dataset = datasets["training"]
        Val_dataset = datasets["validation"]
        test_dataset = datasets["testing"]
    elif dataset_type == "CBCT_CROP":
        datasets = BuildCropDataSet(config).quick_load_data()
        train_dataset = datasets["training"]
        Val_dataset = datasets["validation"]
        test_dataset = datasets["testing"]

    elif dataset_type == "CBCT_SGA":
        datasets = BuildCBCTSGADataSet(config).quick_load_data()
        train_dataset = datasets["training"]
        Val_dataset = datasets["validation"]
        test_dataset = datasets["testing"]

    elif dataset_type == "SampleMesh":
        path = config["data"]["dataset_directory"]
        datasets = BuildSMDataSet(path).quick_load_data()
        train_dataset = datasets["training"]
        Val_dataset = datasets["validation"]
        test_dataset = datasets["testing"]

    else:
        raise NotImplementedError()
    return train_dataset, Val_dataset, test_dataset





