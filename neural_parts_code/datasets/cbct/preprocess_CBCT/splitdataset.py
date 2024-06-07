
import pickle
import os
import numpy as np
class DataModes:
    TRAINING = 'training'
    VALIDATION = 'validation'
    TESTING = 'testing'
    ALL = 'all'
    def __init__(self):
        dataset_splits = [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]

# def splitDataset(dataset_path):
#     dataset = {}
#     for mode in [DataModes.TRAINING,DataModes.VALIDATION,DataModes.TESTING]:
#         path = os.path.join(dataset_path,mode)
#         file_list = os.listdir(path)
#         file_list = [os.path.join(path,fname) for fname in file_list if "pickle" in fname]
#         dataset[mode] = file_list
#
#     with open(os.path.join(dataset_path,"splitGroup.pt"), 'wb') as handle:
#         pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def splitDataset(dataset_path):
    file_list = os.listdir(dataset_path)
    file_list = [os.path.join(dataset_path,fname) for fname in file_list if "pickle" in fname]
    toothNum = len(file_list)

    print('Total numbers of samples is :', toothNum)
    np.random.seed(0)
    perm = np.random.permutation(toothNum)
    counts = [perm[:int(toothNum * 0.8)], perm[int(toothNum * 0.8):int(toothNum * 0.9)],perm[int(toothNum * 0.9):int(toothNum)]]

    dataset = {}

    dataset[DataModes.TRAINING] = [file_list[i] for i in counts[0]]
    dataset[DataModes.VALIDATION] = [file_list[i] for i in counts[1]]
    dataset[DataModes.TESTING] = [file_list[i] for i in counts[2]]

    with open(os.path.join(dataset_path,"cbct_splitGroup.pt"), 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

dataset_path = "/home/yisiinfo/cyj/cj/Tooth-and-alveolar-bone-segmentation-from-CBCT/NC-release-data/NC-release-data-checked"
splitDataset(dataset_path)