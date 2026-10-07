import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# Each image in mnist is 28x28 pixels, grayscale, and contains handwritten digits from 0 through 9. 
# We will flatten each image into 784 input values. 
# The neural network will be: Input: 784 neurons, Hidden: 128 neurons, Hidden: 64 neurons, Output: 10 neurons. 
# The output contains one value for each digit (0-9).

seed = 1

# initialize all four random seeds 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# setting up device, using gpu if availale, else cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# loading and normalizing mnist data set

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # mean and sd for mnist data, according to std info found online. 
    # source: https://mint.westdri.ca/ai/pt/pt_mnist
])


# loading training data 
full_train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True, # mnist has 60k training samples and 10k test samples
    download=True,
    transform=transform
)


# loading test data 
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# split training data into training and validation sets (90% training, 10% validation, arbitrary split)
train_size = int(0.9 * len(full_train_dataset))
validation_size = len(full_train_dataset) - train_size

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(seed)
)

# checking everything looks as expected
print("Training samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Test samples:", len(test_dataset))

# creating our neural network class
class NeuralNetwork(nn.Module):
    """
    NN inheriting pytorch nn.Module class.
    Architecture: 784 -> 128 -> 64 -> 10
    """

    def __init__(self, activation="relu", dropout_rate=0.0):
        super().__init__()

        self.activation_name = activation # storing the activation function
        self.fc1 = nn.Linear(28 * 28, 128) # first fully connected layer, output = input × weights + bias
        self.fc2 = nn.Linear(128, 64) # second fully connected layer
        self.fc3 = nn.Linear(64, 10)  # output layer
        self.dropout = nn.Dropout(dropout_rate) # dropout later. when dropout_rate = 0, dropout is effectively disabled

        # to select the activation function.
        if activation == "relu":
            self.activation = nn.ReLU()

        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

        elif activation == "tanh":
            self.activation = nn.Tanh()

        else:
            raise ValueError("Unknown activation function.")

    # our forward pass. 
    def forward(self, x):

        # first, flatten the 28 x 28 image into 784 values
        x = x.view(x.size(0), -1)

        # first hidden layer
        x = self.fc1(x)

        # activation function
        x = self.activation(x)

        # dropout
        x = self.dropout(x)

        # second hidden layer
        x = self.fc2(x)

        # activation function
        x = self.activation(x)

        # dropout
        x = self.dropout(x)

        # output layer
        x = self.fc3(x)

        return x # ie the final ten digit output


# Helper functions, defined below

def create_dataloader(dataset, batch_size, shuffle=True):
    """
    To make batches of data for training and evaluation depending on specified batch size.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def calculate_accuracy(model, data_loader):
    """
    Calculate classification accuracy for a dataset - what percent of predictions are correct?
    """

    model.eval() # dropout off
    correct = 0
    total = 0

    # we don't need gradient calculation for accuracy
    with torch.no_grad():

        for images, labels in data_loader: 

            images = images.to(device)
            labels = labels.to(device)

            # get model predictions with forward pass
            outputs = model(images)

            # find the class with the largest output
            _, predicted = torch.max(outputs, 1)

            # count correct predictions to return accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def evaluate_model(model, data_loader, criterion):
    """
    Calculating both loss and accuracy for a dataset - how wrong are my predictions?
    """

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / total
    accuracy = 100 * correct / total

    return average_loss, accuracy


# training function 

def train_model(
    model,
    train_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=10,
    l1_lambda=0.0
):
    """
    Train the neural network: model: NN, train laoder: criterion: loss function, l1_lambda: strength of L1 regularization.
    Returns a dictionary containing: training loss, validation loss, training accuracy, validation accuracy.
    """

    # Lists to store results for every epoch.
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:

            # Move data to approp device
            images = images.to(device)
            labels = labels.to(device)

            # Clear old gradients
            optimizer.zero_grad()

            # Do a forward pass
            outputs = model(images)

            # Calculate cross-entropy loss
            loss = criterion(outputs, labels)

            # L1 reg: for sparsity, adds lambda * sum(|weights|), encouraging some weights to become exactly 0. 
            if l1_lambda > 0:

                l1_penalty = 0.0
                for parameter in model.parameters():

                    # only apply L1 to weight parameters.
                    if parameter.requires_grad:
                        l1_penalty += torch.sum(
                            torch.abs(parameter)
                        )

                loss = loss + l1_lambda * l1_penalty

            # backward pass
            loss.backward()

            # update weights, learning rate chosen arbitrarily 
            optimizer.step() 

            # keep track of total loss
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        # average training loss for current epoch
        average_train_loss = running_loss / total_samples

        # training and validation performance
        train_loss, train_accuracy = evaluate_model(
            model,
            train_loader,
            criterion
        )

        validation_loss, validation_accuracy = evaluate_model(
            model,
            validation_loader,
            criterion
        )

        # storing results
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        # showing progress 
        print(
            f"Epoch {epoch + 1:2d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Acc: {validation_accuracy:.2f}%"
        )

    return {
        "train_loss": train_losses,
        "validation_loss": validation_losses,
        "train_accuracy": train_accuracies,
        "validation_accuracy": validation_accuracies
    }


# plotting functions 

def plot_training_loss(results_dict, title):
    """
    Plot training loss for different optimizers/ combinations. 
    """

    plt.figure(figsize=(10, 6))

    for name, results in results_dict.items():

        plt.plot(
            results["train_loss"],
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(title)

    plt.legend()
    plt.grid(True)

    plt.show()


def plot_loss_and_accuracy(results_dict, title):
    """
    Plot training + validation loss and accuracy.
    """

    plt.figure(figsize=(10, 6))

    for name, results in results_dict.items():

        plt.plot(
            results["train_loss"],
            label=f"{name} - Train Loss"
        )

        plt.plot(
            results["validation_loss"],
            linestyle="--",
            label=f"{name} - Validation Loss"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title + " - Loss")

    plt.legend()
    plt.grid(True)

    plt.show()

    # plot accuracy
    plt.figure(figsize=(10, 6))

    for name, results in results_dict.items():

        plt.plot(
            results["train_accuracy"],
            label=f"{name} - Train Accuracy"
        )

        plt.plot(
            results["validation_accuracy"],
            linestyle="--",
            label=f"{name} - Validation Accuracy"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title + " - Accuracy")

    plt.legend()
    plt.grid(True)

    plt.show()

# OPTIMIZATION: Batch Gradient Descent, SGD, Momentum, Adam. 
# Plot Training Loss vs. Epoch and identify which optimizer converges fastest.

# both arbitrary 
EPOCHS = 10
LEARNING_RATE = 0.001


# Batch Gradient Descent: use entire training dataset in one batch
batch_gd_loader = create_dataloader(
    train_dataset,
    batch_size=len(train_dataset),
    shuffle=True
)

# SGD: use one training example at a time
sgd_loader = create_dataloader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

# Momentum: mini-batch is used for practical training speed
momentum_loader = create_dataloader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Adam: use the same mini-batch size as Momentum
adam_loader = create_dataloader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# Validation loader does not need to shuffle
validation_loader = create_dataloader(
    validation_dataset,
    batch_size=256,
    shuffle=False
)

# loss function & optimization results
criterion = nn.CrossEntropyLoss()
optimization_results = {}

# BATCH GRADIENT DESCENT
print("\n" + "=" * 60)
print("BATCH GRADIENT DESCENT")
print("=" * 60)

model = NeuralNetwork(activation="relu").to(device)

optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)

optimization_results["Batch Gradient Descent"] = train_model(
    model,
    batch_gd_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=EPOCHS
)


# SGD
print("\n" + "=" * 60)
print("SGD")
print("=" * 60)

model = NeuralNetwork(activation="relu").to(device)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.001 # learning rate
)

optimization_results["SGD"] = train_model(
    model,
    sgd_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=EPOCHS
)

# MOMENTUM
print("\n" + "=" * 60)
print("MOMENTUM")
print("=" * 60)

model = NeuralNetwork(activation="relu").to(device)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.001, # learning rate
    momentum=0.9
)

optimization_results["Momentum"] = train_model(
    model,
    momentum_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=EPOCHS
)

# ADAM
print("\n" + "=" * 60)
print("ADAM")
print("=" * 60)

model = NeuralNetwork(activation="relu").to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001 # learning rate
)

optimization_results["Adam"] = train_model(
    model,
    adam_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=EPOCHS
)


# Plotting Training Loss vs. Epoch

plot_training_loss(
    optimization_results,
    "Optimization Comparison"
)

# Print final optimization results
print("\nFinal Optimization Results")
print("-" * 60)

for name, results in optimization_results.items():

    final_loss = results["train_loss"][-1]
    final_accuracy = results["train_accuracy"][-1]

    print(
        f"{name:25s} "
        f"Loss = {final_loss:.4f}, "
        f"Accuracy = {final_accuracy:.2f}%"
    )

# REGULARIZATION: Compare no regularization, L1, L2 and dropout
# Compare Training/Validation Loss & Accuracy and analyze generalization.

# arbitrary settings for regularization experiments
REGULARIZATION_EPOCHS = 10
REGULARIZATION_BATCH_SIZE = 64

regularization_loader = create_dataloader(
    train_dataset,
    batch_size = REGULARIZATION_BATCH_SIZE,
    shuffle = True
)

# to store results
regularization_results = {}


# no regularization
print("\n" + "=" * 60)
print("NO REGULARIZATION")
print("=" * 60)

model = NeuralNetwork(
    activation="relu",
    dropout_rate=0.0
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001 # learning rate
)

regularization_results["No Regularization"] = train_model(
    model,
    regularization_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=REGULARIZATION_EPOCHS
)


# L2 REGULARIZATION: pytorch implements L2 regularization through weight_decay, penality for squared weight magnitude,
# shrinkage but no sparsity generally. 

print("\n" + "=" * 60)
print("L2 REGULARIZATION")
print("=" * 60)

model = NeuralNetwork(
    activation="relu",
    dropout_rate=0.0
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001, # learning rate
    weight_decay=0.0001
)

regularization_results["L2"] = train_model(
    model,
    regularization_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=REGULARIZATION_EPOCHS
)

# L1 REGULARIZATION: implemented manually in train_model(), adds lambda * sum(abs(weights)), encouraging sparsity.

print("\n" + "=" * 60)
print("L1 REGULARIZATION")
print("=" * 60)

model = NeuralNetwork(
    activation="relu",
    dropout_rate=0.0
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001 # learning rate
)

regularization_results["L1"] = train_model(
    model,
    regularization_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=REGULARIZATION_EPOCHS,
    l1_lambda=0.00001
)


# DROPOUT: randomly sets some activations to zero during training, prevents the network from depending too heavily on
# particular neurons and can improve generalization.
print("\n" + "=" * 60)
print("DROPOUT")
print("=" * 60)

model = NeuralNetwork(
    activation="relu",
    dropout_rate=0.5
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001 # learning rate
)

regularization_results["Dropout"] = train_model(
    model,
    regularization_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=REGULARIZATION_EPOCHS
)


# Comparing reg results 

plot_loss_and_accuracy(
    regularization_results,
    "Regularization Comparison"
)

print("\nFinal Regularization Results")
print("-" * 75)

for name, results in regularization_results.items():

    train_loss = results["train_loss"][-1]
    validation_loss = results["validation_loss"][-1]

    train_accuracy = results["train_accuracy"][-1]
    validation_accuracy = results["validation_accuracy"][-1]

    print(
        f"{name:20s} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {validation_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {validation_accuracy:.2f}%"
    )


# Effect of L1 and L2 on weights, looking at average absolute weight and number of weights close to zero
def analyze_weights(model):
    """
    Calculate simple statistics about the model weights.
    """

    total_weights = 0
    near_zero_weights = 0
    absolute_weight_sum = 0.0

    for parameter in model.parameters():

        if parameter.requires_grad:

            weights = parameter.detach().cpu()

            total_weights += weights.numel()

            near_zero_weights += (
                torch.abs(weights) < 0.001
            ).sum().item()

            absolute_weight_sum += (
                torch.abs(weights).sum().item()
            )

    average_absolute_weight = (
        absolute_weight_sum / total_weights
    )

    percent_near_zero = (
        100 * near_zero_weights / total_weights
    )

    return average_absolute_weight, percent_near_zero


# Training separate models for weight analysis

# No regularization
model_no_reg = NeuralNetwork().to(device)

optimizer_no_reg = optim.Adam(
    model_no_reg.parameters(),
    lr=0.001 # learning rate
)

train_model(
    model_no_reg,
    regularization_loader,
    validation_loader,
    optimizer_no_reg,
    criterion,
    epochs=REGULARIZATION_EPOCHS
)


# L2
model_l2 = NeuralNetwork().to(device)

optimizer_l2 = optim.Adam(
    model_l2.parameters(),
    lr=0.001, # learning rate
    weight_decay=0.0001
)

train_model(
    model_l2,
    regularization_loader,
    validation_loader,
    optimizer_l2,
    criterion,
    epochs=REGULARIZATION_EPOCHS
)


# L1
model_l1 = NeuralNetwork().to(device)

optimizer_l1 = optim.Adam(
    model_l1.parameters(),
    lr=0.001 # learning rate
)

train_model(
    model_l1,
    regularization_loader,
    validation_loader,
    optimizer_l1,
    criterion,
    epochs=REGULARIZATION_EPOCHS,
    l1_lambda=0.00001
)


# Weight stats analysis: 

no_reg_stats = analyze_weights(model_no_reg)
l2_stats = analyze_weights(model_l2)
l1_stats = analyze_weights(model_l1)


print("\nWeight Analysis")
print("-" * 70)

print(
    f"No Regularization: "
    f"Average |weight| = {no_reg_stats[0]:.6f}, "
    f"Near-zero weights = {no_reg_stats[1]:.2f}%"
)

print(
    f"L2: "
    f"Average |weight| = {l2_stats[0]:.6f}, "
    f"Near-zero weights = {l2_stats[1]:.2f}%"
)

print(
    f"L1: "
    f"Average |weight| = {l1_stats[0]:.6f}, "
    f"Near-zero weights = {l1_stats[1]:.2f}%"
)

# ACTIVATION FUNCTIONS
ACTIVATION_EPOCHS = 10

activation_loader = create_dataloader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# same optimizer for every activation function (adam) to provide a consistent optimizer for the activation comparison.

activation_results = {}

# RELU
print("\n" + "=" * 60)
print("RELU")
print("=" * 60)

model = NeuralNetwork(
    activation="relu"
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001 # learning rate
)

activation_results["ReLU"] = train_model(
    model,
    activation_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=ACTIVATION_EPOCHS
)

# SIGMOID
print("\n" + "=" * 60)
print("SIGMOID")
print("=" * 60)

model = NeuralNetwork(
    activation="sigmoid"
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001 # learning rate
)

activation_results["Sigmoid"] = train_model(
    model,
    activation_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=ACTIVATION_EPOCHS
)


# TANH
print("\n" + "=" * 60)
print("TANH")
print("=" * 60)

model = NeuralNetwork(
    activation="tanh"
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001 # learning rate
)

activation_results["Tanh"] = train_model(
    model,
    activation_loader,
    validation_loader,
    optimizer,
    criterion,
    epochs=ACTIVATION_EPOCHS
)


# Comparing activation functions

plot_loss_and_accuracy(
    activation_results,
    "Activation Function Comparison"
)

print("\nFinal Activation Function Results")
print("-" * 75)

for name, results in activation_results.items():

    train_loss = results["train_loss"][-1]
    validation_loss = results["validation_loss"][-1]

    train_accuracy = results["train_accuracy"][-1]
    validation_accuracy = results["validation_accuracy"][-1]

    print(
        f"{name:10s} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {validation_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {validation_accuracy:.2f}%"
    )

# Analysis info

print("\n" + "=" * 70)
print("ANALYSIS: OPTIMIZER")
print("=" * 70)


def convergence_epoch(results, target_loss):
    """
    Find the first epoch where training loss reaches
    or falls below a predefined target loss
    """

    for epoch, loss in enumerate(results["train_loss"]):

        if loss <= target_loss:
            return epoch + 1

    return None


# Using the same target loss for every optimizer, this makes the convergence comparison fair.
TARGET_LOSS = 0.10


for name, results in optimization_results.items():

    final_loss = results["train_loss"][-1]

    convergence = convergence_epoch(
        results,
        TARGET_LOSS
    )

    if convergence is None:
        convergence_text = "Did not reach target"
    else:
        convergence_text = f"Epoch {convergence}"

    print(
        f"{name:25s}: "
        f"Final loss = {final_loss:.4f}, "
        f"Convergence = {convergence_text}"
    )

# How did l1 & l2 affect the weights?

print("\n" + "=" * 70)
print("ANALYSIS: L1 VS L2")
print("=" * 70)

print(
    "No Regularization average |weight|:",
    f"{no_reg_stats[0]:.6f}"
)

print(
    "L2 average |weight|:",
    f"{l2_stats[0]:.6f}"
)

print(
    "L1 average |weight|:",
    f"{l1_stats[0]:.6f}"
)

print()

print(
    "No Regularization near-zero weights:",
    f"{no_reg_stats[1]:.2f}%"
)

print(
    "L2 near-zero weights:",
    f"{l2_stats[1]:.2f}%"
)

print(
    "L1 near-zero weights:",
    f"{l1_stats[1]:.2f}%"
)


# Dropout effect on training versus validation

print("\n" + "=" * 70)
print("ANALYSIS: DROPOUT")
print("=" * 70)

no_reg_train_acc = regularization_results[
    "No Regularization"
]["train_accuracy"][-1]

no_reg_val_acc = regularization_results[
    "No Regularization"
]["validation_accuracy"][-1]


dropout_train_acc = regularization_results[
    "Dropout"
]["train_accuracy"][-1]

dropout_val_acc = regularization_results[
    "Dropout"
]["validation_accuracy"][-1]


print(
    "No Regularization - Training Accuracy:",
    f"{no_reg_train_acc:.2f}%"
)

print(
    "No Regularization - Validation Accuracy:",
    f"{no_reg_val_acc:.2f}%"
)

print()

print(
    "Dropout - Training Accuracy:",
    f"{dropout_train_acc:.2f}%"
)

print(
    "Dropout - Validation Accuracy:",
    f"{dropout_val_acc:.2f}%"
)


# Activation function performance

print("\n" + "=" * 70)
print("ANALYSIS: ACTIVATION FUNCTIONS")
print("=" * 70)

for name, results in activation_results.items():

    print(
        f"{name:10s}: "
        f"Final Train Accuracy = "
        f"{results['train_accuracy'][-1]:.2f}%, "
        f"Final Validation Accuracy = "
        f"{results['validation_accuracy'][-1]:.2f}%"
    )


# Which config generalized best?

def calculate_generalization_gap(results):
    """
    Calculate the difference between training and validation
    accuracy.
    """

    train_accuracy = results["train_accuracy"][-1]
    validation_accuracy = results["validation_accuracy"][-1]

    return train_accuracy - validation_accuracy


print("\n" + "=" * 70)
print("ANALYSIS: GENERALIZATION")
print("=" * 70)


for name, results in regularization_results.items():

    gap = calculate_generalization_gap(results)

    validation_accuracy = results[
        "validation_accuracy"
    ][-1]

    print(
        f"{name:20s}: "
        f"Validation Accuracy = {validation_accuracy:.2f}%, "
        f"Train-Val Gap = {gap:.2f}%"
    )
