import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            # TODO: complete this part
        )
        self.classifier = nn.Sequential(
            # TODO: complete this part
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features), features
    

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # NOTE: this is optional
        )
        self.classifier = nn.Sequential(
            # NOTE: this is optional
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 12 * 12)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features), features
    

def load_mnist_dataset(batch_size=64, val_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)

    # Split dataset into train and validation
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    # TODO: define cross-entropy loss here
    criterion = None

    # TODO: define Adam optimizer here with given learning rate (lr)
    optimizer = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device)
                targets = targets.to(device)

                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss / len(val_loader):.4f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model.get_features(images)  # Extract final layer outputs, ignore intermediate features
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append predictions and true labels for confusion matrix
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

    return predictions, true_labels


def plot_confusion_matrix(cm, classes, model_name, batch_size, lr):
    # Create directory for saving plots
    plot_dir = f'{model_name}/bs{batch_size}_lr{lr}'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Add labels to the plot
    thresh = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/confusion.png')
    plt.close()


def visualize_tsne(model, test_loader, class_labels, model_name, batch_size, lr):
    # Create directory for saving plots
    plot_dir = f'{model_name}/bs{batch_size}_lr{lr}'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    model.eval()
    features = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            _, intermediate_features = model.get_features(images)  # Extract intermediate features
            features.append(intermediate_features.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    features = np.concatenate(features, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    for i in range(len(class_labels)):
        indices = true_labels == i
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=class_labels[i])
    plt.legend()
    plt.title('t-SNE Visualization')
    plt.savefig(f'{plot_dir}/t-SNE.png')
    plt.close()


def plot_loss(train_losses, val_losses, model_name, batch_size, lr):
    # Create directory for saving plots
    plot_dir = f'{model_name}/bs{batch_size}_lr{lr}'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{plot_dir}/loss.png')
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='MNIST Classification')
    parser.add_argument('--model', choices=['lenet', 'alexnet'], help='Model name (lenet or alexnet)', required=True)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio (default: 0.1)')
    return parser.parse_args()

def run():
    # Parse command-line arguments
    args = parse_arguments()

    # Read the args
    model_name = args.model.lower()
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_split = args.val_split

    # Load MNIST dataset
    train_loader, val_loader, test_loader = load_mnist_dataset(batch_size, val_split)

    # Create the model
    if model_name == 'lenet':
        model = LeNet().to(device)
    elif model_name == 'alexnet':
        model = AlexNet().to(device)
    else:
        print("Invalid model name. Please choose 'lenet' or 'alexnet'")
        return

    # Create directory for model
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, lr)

    # Plot Train/Val loss
    plot_loss(train_losses, val_losses, model_name, batch_size, lr)

    # Evaluate the model
    predictions, true_labels = evaluate_model(model, test_loader)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Define class labels
    class_labels = [str(i) for i in range(10)]

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_labels, model_name, batch_size, lr)

    # Visualize t-SNE
    visualize_tsne(model, test_loader, class_labels, model_name, batch_size, lr)


if __name__ == '__main__':
    run()
