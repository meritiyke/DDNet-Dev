import os
from torch.utils import data
from dataset.deformationDetection import DeformationDetectionDataSet, DeformationDetectionValDataSet, DeformationDetectionTestDataSet

def build_dataset_train(dataRootDir, dataset, input_size, batch_size, random_mirror, num_workers):
    """
    Build the training and validation datasets for DDNet.
    """
    if dataset == 'deformationDetection':
        data_dir = os.path.join(dataRootDir, dataset)
        train_data_list = os.path.join(data_dir, 'train.txt')
        val_data_list = os.path.join(data_dir, 'val.txt')

        # Training dataset
        trainLoader = data.DataLoader(
            DeformationDetectionDataSet(data_dir, train_data_list, crop_size=input_size,
                                       mirror=random_mirror),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        # Validation dataset
        valLoader = data.DataLoader(
            DeformationDetectionValDataSet(data_dir, val_data_list),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            drop_last=True)

        datas = None
        return datas, trainLoader, valLoader
    else:
        raise NotImplementedError(
            "Dataset not supported: %s" % dataset)

def build_dataset_test(dataRootDir, dataset, num_workers, none_gt=False):
    """
    Build the test dataset for DDNet.
    """
    if dataset == 'deformationDetection':
        data_dir = os.path.join(dataRootDir, dataset)
        test_data_list = os.path.join(data_dir, 'test.txt')

        if none_gt:
            # Test dataset without ground truth
            testLoader = data.DataLoader(
                DeformationDetectionTestDataSet(data_dir, test_data_list),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            # Test dataset with ground truth
            testLoader = data.DataLoader(
                DeformationDetectionValDataSet(data_dir, test_data_list),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        datas = None
        return datas, testLoader
    else:
        raise NotImplementedError(
            "Dataset not supported: %s" % dataset)