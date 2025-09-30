import torch
import pickle
import tqdm 
from dataset.utils import mnist_dataloader

from atomgrad.tensor import atom
import atomgrad.nn as nn
import atomgrad.optim as optim

DEVICE = 'cuda'

def atom_runner():
    MAX_EPOCHS = 100
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    BATCH_SIZE = 4096
    LEARNING_RATE = 0.001

    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH
    
    # Initialize Model
    parameters = []
    linear_1, params_1 = nn.linear(784, 2000, device=DEVICE)
    activation = nn.relu()
    linear_2, params_2 = nn.linear(2000, 10, device=DEVICE)
    parameters.extend(params_1)
    parameters.extend(params_2)

    loss_fn = nn.cross_entropy()
    step, zero_grad = optim.Adam(parameters, lr=LEARNING_RATE)

    for _ in (t := tqdm.trange(MAX_EPOCHS)):
        train_loader = mnist_dataloader(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = mnist_dataloader(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=True)

        # Training Loop
        train_loss = []
        for batched_image, batched_label in train_loader:
            batched_image = atom(batched_image, device=DEVICE)
            batched_label = atom(batched_label, device=DEVICE)
            batch = batched_image.shape[0]
            # Forward pass: linear 1 -> activation fn -> linear 2
            model_prediction = linear_2(activation(linear_1(batched_image)))
            avg_loss = loss_fn(model_prediction, batched_label)

            zero_grad()
            avg_loss.backward()
            step(batch)

            train_loss.append(avg_loss.data.item())

        # Test Loop
        accuracies = []
        for batched_image, batched_label in test_loader:
            batched_image = atom(batched_image, requires_grad=True, device=DEVICE)
            batched_label = atom(batched_label, device=DEVICE)
            model_pred_probabilities = linear_2(activation(linear_1(batched_image))).data
            batch_accuracy = (model_pred_probabilities.argmax(axis=-1) == batched_label.data).mean()
            accuracies.append(batch_accuracy)
        train_loss = sum(train_loss) / len(train_loss)
        accuracies = sum(accuracies) / len(accuracies)

        t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracies:.4f}')

def torch_runner():
    MAX_EPOCHS = 100
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    BATCH_SIZE = 4096
    LEARNING_RATE = 0.001

    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    # Initialize Model
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_1 = torch.nn.Linear(784, 2000, device=DEVICE)
            self.activation = torch.nn.ReLU()
            self.lin_2 = torch.nn.Linear(2000, 10, device=DEVICE)

        def forward(self, x):
            return self.lin_2(self.activation(self.lin_1(x)))

    mlp = MLP()
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE)

    for _ in (t := tqdm.trange(MAX_EPOCHS)):
        train_loader = mnist_dataloader(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = mnist_dataloader(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=True)

        # Training Loop
        train_loss = []
        for batched_image, batched_label in train_loader:
            batched_image = torch.tensor(batched_image, requires_grad=True, device=DEVICE)
            batched_label = torch.tensor(batched_label, device=DEVICE)

            # Forward pass: linear 1 -> activation fn -> linear 2
            model_prediction = mlp(batched_image)

            loss = loss_fn(model_prediction, batched_label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss.append(loss.item())

        # Test Loop
        accuracies = []
        for batched_image, batched_label in test_loader:
            batched_image = torch.tensor(batched_image, requires_grad=True, device=DEVICE)
            batched_label = torch.tensor(batched_label, device=DEVICE)

            model_pred_probabilities = mlp(batched_image)
            batch_accuracy = (model_pred_probabilities.argmax(axis=-1) == batched_label).float().mean()

            accuracies.append(batch_accuracy.item())

        train_loss = sum(train_loss) / len(train_loss)
        accuracies = sum(accuracies) / len(accuracies)

        t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracies:.4f}')

# Atom previous GPU behaves:
# index, name, memory.total [MiB], memory.used [MiB], memory.free [MiB], temperature.gpu, pstate, utilization.gpu [%], utilization.memory [%]
# 0, Quadro RTX 4000, 8192 MiB, 4644 MiB, 3352 MiB, 71, P0, 78 %, 44 %

# Atom refactored v1 GPU behaves:
# index, name, memory.total [MiB], memory.used [MiB], memory.free [MiB], temperature.gpu, pstate, utilization.gpu [%], utilization.memory [%]
# 0, Quadro RTX 4000, 8192 MiB, 3241 MiB, 4755 MiB, 71, P0, 72 %, 43 %

# Atom refactored v2 GPU behaves:
# index, name, memory.total [MiB], memory.used [MiB], memory.free [MiB], temperature.gpu, pstate, utilization.gpu [%], utilization.memory [%]
# 0, Quadro RTX 4000, 8192 MiB, 2582 MiB, 5414 MiB, 78, P0, 74 %, 46 %

# Torch GPU behaves:
# index, name, memory.total [MiB], memory.used [MiB], memory.free [MiB], temperature.gpu, pstate, utilization.gpu [%], utilization.memory [%]
# 0, Quadro RTX 4000, 8192 MiB, 1577 MiB, 6419 MiB, 76, P0, 76 %, 24 %

# TODO: Make atomgrad effecient as Pytorch
# TODO: How can I make atomgrad memory efficient same as pytorch

atom_runner()
# torch_runner()
