import tqdm
import pickle
import torch
import MLP.config as configs
from dataset.utils import mnist_dataloader

def torch_mlp_runner():
    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == configs.IMAGE_HEIGHT*configs.IMAGE_WIDTH

    # Initialize Model    
    model_parameters = []
    layer_1 = torch.nn.Linear(784, 2000, device=configs.DEVICE)
    layer_1_activation = torch.nn.ReLU()
    layer_2 = torch.nn.Linear(2000, 10, device=configs.DEVICE)

    # Accumulate parameters
    model_parameters.extend([layer_1.weight, layer_1.bias])
    model_parameters.extend([layer_2.weight, layer_2.bias])

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model_parameters, lr=configs.LEARNING_RATE)
    
    # Loop through EPOCHS
    for _ in (t := tqdm.trange(configs.MAX_EPOCHS)):
        train_loader = mnist_dataloader(train_images, train_labels, batch_size=configs.BATCH_SIZE, shuffle=True)
        test_loader = mnist_dataloader(test_images, test_labels, batch_size=configs.BATCH_SIZE, shuffle=True)

        # Training Loop
        train_loss = []
        for batched_image, batched_label in train_loader:
            # Converting from numpy array to torch tensor
            batched_image = torch.tensor(batched_image, requires_grad=True, device=configs.DEVICE)
            batched_label = torch.tensor(batched_label, device=configs.DEVICE)

            # Model forward pass
            model_output = layer_2(layer_1_activation(layer_1(batched_image)))

            loss = loss_fn(model_output, batched_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # Test Loop
        accuracies = []
        for batched_image, batched_label in test_loader:
            batched_image = torch.tensor(batched_image, requires_grad=True, device=configs.DEVICE)
            batched_label = torch.tensor(batched_label, device=configs.DEVICE)

            model_prediction = layer_2(layer_1_activation(layer_1(batched_image)))
            batch_accuracy = (model_prediction.argmax(axis=-1) == batched_label).float().mean()

            accuracies.append(batch_accuracy.item())

        train_loss = sum(train_loss) / len(train_loss)
        accuracies = sum(accuracies) / len(accuracies)

        t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracies:.4f}')
