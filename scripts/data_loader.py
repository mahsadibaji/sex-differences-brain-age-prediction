import numpy as np
import pandas as pd
import monai
from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityd, RandRotated
)

def load_data(train_dataset_path, val_dataset_path, batch, verbose=True):
    """
    Load training and validation data, apply transformations, and create data loaders.

    Parameters:
        train_dataset_path (str): Path to the training dataset CSV file.
        val_dataset_path (str): Path to the validation dataset CSV file.
        batch (int): Batch size for the DataLoader.

    Returns:
        Tuple: Training dataset, training DataLoader, validation dataset, validation DataLoader.
    """
    
    # Load CSV data
    train_data = pd.read_csv(train_dataset_path)
    val_data = pd.read_csv(val_dataset_path)
    
    # Extract necessary columns
    imgs_list_train, age_labels_train = train_data['filename'], train_data['age']
    imgs_list_val, age_labels_val = val_data['filename'], val_data['age']

    # Create a list of dictionaries for training set
    filenames_train = [{"img": x, "age_label": z} for (x, z) in zip(imgs_list_train, age_labels_train)]
    ds_train = Dataset(filenames_train, train_transforms)
    train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, collate_fn=pad_list_data_collate)

    # Create a list of dictionaries for validation set
    filenames_val = [{"img": x, "age_label": z} for (x, z) in zip(imgs_list_val, age_labels_val)]
    ds_val = Dataset(filenames_val, val_transforms)
    val_loader = DataLoader(ds_val, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, collate_fn=pad_list_data_collate)
    
    if verbose:
        print(f"Size of Training set: {len(age_labels_train)}")
        print(f"Size of Validation set: {len(age_labels_val)}")

    return ds_train, train_loader, ds_val, val_loader


def load_test_data(test_dataset_path, verbose=True):
    """
    Load test data, apply transformations, and create a data loader.

    Parameters:
        test_dataset_path (str): Path to the test dataset CSV file.

    Returns:
        Tuple: Test dataset and its DataLoader.
    """
    
    test_data = pd.read_csv(test_dataset_path)
    
    # Extract necessary columns
    imgs_list_test = test_data['filename']
    age_labels_test = test_data['age']
    subj_ids_test = test_data['subject_id']

    # Create a list of dictionaries for test set
    filenames_test = [{"img": x, "age_label": z, "sid": s} for (x, z, s) in zip(imgs_list_test, age_labels_test, subj_ids_test)]
    ds_test = Dataset(filenames_test, test_transforms)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    if verbose:
        print(f"Size of Training set: {len(age_labels_test)}")
        
    return ds_test, test_loader 


# Transformations for training, validation, and test datasets
train_transforms = Compose([
    LoadImaged(keys=["img"]),
    EnsureChannelFirstd(keys=["img"]),
    ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
    RandRotated(keys=["img"], range_x=np.pi / 12, prob=0.5, keep_size=True, mode="nearest")
])

val_transforms = Compose([
    LoadImaged(keys=["img"]),
    EnsureChannelFirstd(keys=["img"]),
    ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
    RandRotated(keys=["img"], range_x=np.pi / 12, prob=0.5, keep_size=True, mode="nearest")
])

test_transforms = Compose([
    LoadImaged(keys=["img"]),
    EnsureChannelFirstd(keys=["img"]),
    ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0)
])
