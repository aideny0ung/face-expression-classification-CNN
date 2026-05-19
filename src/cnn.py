import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt



class CNN(nn.Module):
    def __init__(self,classes):
        super().__init__()

        channels = 16

        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, 2*channels, 3, padding=1)
        self.conv4 = nn.Conv2d(2*channels, 2*channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(2*channels)

        self.conv5 = nn.Conv2d(2*channels, 4*channels, 3, padding=1)
        self.conv6 = nn.Conv2d(4*channels, 4*channels, 3, padding=1)


        self.dropout = nn.Dropout(0.5)

        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4*channels*6*6,32)
        self.fc2 = nn.Linear(32,classes)

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv2(F.relu(self.conv1(x))))))
        x = self.pool(F.relu(self.bn2(self.conv4(F.relu(self.conv3(x))))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc2(F.relu(self.fc1(x)))

        return x
    


def load_data(source):

    data = torch.load(source)

    train_data = data["train"]
    test_data = data["test"]
    #class_to_idx = data["class_to_idx"] not sure what to do with this. might be useful for testing

    train_dataset = TensorDataset(train_data["X"], train_data["y"])
    test_dataset = TensorDataset(test_data["X"], test_data["y"])

    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=32) #loaders are useful for some reason. not sure exactly why.

    return train_loader, test_loader


def train_cnn(model, train_loader, criterion, optimizer):
    model.train() #apparently this is useful for pytorch models
    #total_loss = 0 #for average loss. useful for testing
    correct = 0
    total = 0

    for X, y in train_loader: #loops over entire dataset [batchsize] at a time
        optimizer.zero_grad() #reset gradients

        X = X.float()
        y = y.long() #paranoid about the datatype
        output = model(X) #runs forward pass
        
        loss = criterion(output, y) #loss function
        loss.backward() #compute gradients
        optimizer.step() #gradient descent
        #total_loss += loss.item()

        prediction = output.argmax(dim=1)
        correct += (prediction == y).sum().item()
        total += y.size(0)

    return correct / total #reports train accuracy to monitor for overfitting
    #return total_loss / len(train_loader) #average loss


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.float() #ensure datatypes are correct
            y = y.long()
            outputs = model(X)

            prediction = outputs.argmax(dim=1) #returns index of max element for each tuple in y
            correct += (prediction == y).sum().item() #sums up all the correct argmaxxes 1 for correct 0 for wrong
            total += y.size(0)

    return correct / total

def findHighLoss(source, data):
    model = CNN(6)
    weights = torch.load(source)

    model.load_state_dict(weights)

    data_path = data
    dataset = torch.load(data_path)

    # Extract just the training data
    X_train = dataset["train"]["X"]
    y_train = dataset["train"]["y"]
    class_to_idx = dataset["class_to_idx"]

    # Reverse dictionary for plotting
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # --- 2. Initialize DataLoader ---
    # CRITICAL: shuffle=False ensures our tracked losses map perfectly back to X_train indices
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)

    # --- 3. Load the Trained Model ---
    # (Instantiate your model class here first, e.g., model = MyCNN())
    # model.load_state_dict(torch.load("path_to_your_saved_weights.pth"))

    # CRITICAL: Put model in evaluation mode so dropout and batch norm are disabled
    model.eval() 

    # --- 4. Setup Loss Function ---
    # reduction='none' gives us the loss for each individual image, not the batch average
    criterion = nn.CrossEntropyLoss(reduction='none')

    all_losses = []
    all_predictions = []

    print("Scanning training dataset for high-loss images...")

    # Disable gradient calculation to save memory and speed up inference
    with torch.no_grad():
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            
            # Calculate individual losses: shape is [batch_size]
            loss_per_image = criterion(outputs, labels)
            
            # Grab the network's actual guess (highest probability index)
            _, predicted = torch.max(outputs, 1)
            
            # Move the data back to CPU and append to our tracking lists
            all_losses.extend(loss_per_image.numpy())
            all_predictions.extend(predicted.numpy())

            all_losses = np.array(all_losses)
            all_predictions = np.array(all_predictions)

            num_to_inspect = 15
            highest_loss_indices = np.argsort(all_losses)[-num_to_inspect:][::-1]

            fig, axes = plt.subplots(3, 5, figsize=(15, 8))
            axes = axes.flatten()

            for i, idx in enumerate(highest_loss_indices):
                img_tensor = X_train[idx]
                true_label = int(y_train[idx].item())
                pred_label = all_predictions[idx]
                loss_val = all_losses[idx]
                
                # Format for Matplotlib
                img_display = img_tensor.squeeze().numpy()
                
                axes[i].imshow(img_display, cmap='gray', vmin=0, vmax=1)
                
                # Title shows the Truth, the CNN's Guess, and the exact Loss score
                axes[i].set_title(f"True: {idx_to_class[true_label]}\nPred: {idx_to_class[pred_label]}\nLoss: {loss_val:.2f}")
                axes[i].axis('off')

            plt.suptitle("Top 15 Highest Loss Training Images", fontsize=16)
            plt.tight_layout()
            plt.show()