from PIL import Image
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def transform_jpg_to_dataset(src_data):
    transform = transforms.Compose([
        transforms.Resize((48, 48)), #nice square power of 2
        transforms.Grayscale(), #color doesn't matter for expression
        transforms.ToTensor() #scales images to [0,1]
    ])
    dataset = datasets.ImageFolder(root = src_data, transform = transform)

    return dataset

def save_dataset(dataset, destination):

    N = len(dataset)
    C, H, W = dataset[0][0].shape
    X = torch.zeros(N, C, H, W)
    y = torch.zeros(N)
    for i in range(N):
        image, label = dataset[i]
        X[i] = image
        y[i] = label

    data = {
        "X": X,
        "y": y,
        "class_to_idx": dataset.class_to_idx
        }
    torch.save(data, destination)
    print(f"saved dataset of length {N} to {destination}")

def create_train_test_split(data, destination):
    X = data["X"]
    y = data["y"].long()
    class_to_idx = data["class_to_idx"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                    test_size = 0.2, random_state = 67, stratify = y)
    dataset = {
        "train": {
            "X": X_train,
            "y": y_train
            },
        "test": {
            "X": X_test,
            "y": y_test
            },
        "class_to_idx": class_to_idx
    }

    torch.save(dataset, destination)
    print(f"saved train/test split to {destination}")
