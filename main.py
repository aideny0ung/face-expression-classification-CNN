from src import preprocess_data as pre
from src import cnn as cnn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


t = True
f = False

PREPROCESS = f
MODEL = f
PLOT = f
LOSS = t
RANDOM_TESTS = f

def main():
    if PREPROCESS:
        dataset = pre.transform_jpg_to_dataset(
                            "data/raw/Expressions")
        pre.save_dataset(dataset, 
                            "data/processed/expression_tensors.pt")
        data = torch.load(
                            "data/processed/expression_tensors.pt")
        pre.create_train_test_split(data,
                            "data/processed/train_test.pt")
        
    if MODEL:
        train_loader, test_loader = cnn.load_data(
                                    "data/processed/final_fer2013_train_test.pt")

        model = cnn.CNN(6)
        criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum = 0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        epochs = 20

        best_acc = 0.0
        train_accuracies = [0]
        test_accuracies = [0]

        for epoch in range(epochs):

            train_acc = cnn.train_cnn(
                model, train_loader, criterion, optimizer)
            test_acc = cnn.evaluate(model, test_loader)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), "models/cleaned_fer2013.pth")

            print(f"epoch {epoch+1}: train acc = {train_acc:.3f}: test acc = {test_acc:.3f}")


        if PLOT:

            epochs_range = range(0, epochs + 1)

            plt.figure(figsize=(10, 6))
            plt.plot(epochs_range, train_accuracies, label='Train Accuracy', color='blue')
            plt.plot(epochs_range, test_accuracies, label='Test Accuracy', color='orange')

            plt.ylim(0, 1)

            plt.title('Training and Testing Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.show()

    if LOSS:
        cnn.findHighLoss("models/cleaned_fer2013.pth","data/processed/final_fer2013_train_test.pt")
        


#/////////////////////////////////////////////////////////////////////////////////////////////////////
    if RANDOM_TESTS:
        dataset = torch.load(
                                        "data/processed/final_fer2013_tensors.pt")
        pre.create_train_test_split(dataset,
                           "data/processed/final_fer2013_train_test.pt")

if __name__ == '__main__':
    main()