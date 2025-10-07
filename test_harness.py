import tqdm
import torch
import pickle
from standard_mlp import StandardMLP
from dataset.utils import mnist_dataloader
from multi_agents import MultiAgents, AgentMarginLoss

def standard_mlp_update(model, model_prediction, expected_output, learning_rate):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = loss_fn(model_prediction, expected_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def multi_agent_update(model, model_prediction, expected_output, learning_rate):
    loss_fn = AgentMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = loss_fn(model_prediction, expected_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss 

def model_runner(model, model_update_function, model_name):
    DEVICE = 'cuda'
    MAX_EPOCHS = 100
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    BATCH_SIZE = 4096
    LEARNING_RATE = 0.001

    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    for _ in (t := tqdm.trange(MAX_EPOCHS)):
        train_loader = mnist_dataloader(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = mnist_dataloader(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=True)

        # Training Loop
        train_loss = []
        for batched_image, batched_label in train_loader:
            batched_image = torch.tensor(batched_image, requires_grad=True, device=DEVICE)
            batched_label = torch.nn.functional.one_hot(torch.tensor(batched_label, device=DEVICE), num_classes=10).float()

            model_output = model(batched_image)
            loss = model_update_function(model, model_output, batched_label, LEARNING_RATE)

            train_loss.append(loss.item())

        # Test Loop
        accuracies = []
        for batched_image, batched_label in test_loader:
            batched_image = torch.tensor(batched_image, requires_grad=True, device=DEVICE)
            batched_label = torch.tensor(batched_label, device=DEVICE)

            model_prediction = model(batched_image)

            batch_accuracy = (model_prediction.argmax(axis=-1) == batched_label).float().mean()
            accuracies.append(batch_accuracy.item())

        train_loss = sum(train_loss) / len(train_loss)
        accuracies = sum(accuracies) / len(accuracies)

        t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracies:.4f}')
        
    # torch.save(model, f'{model_name}.pth')

# Multi agents architecture
multi_agents = MultiAgents()
multi_agents_params_update = multi_agent_update

# Standard MLP architecture
mlp_model = StandardMLP()
mlp_model_params_update = standard_mlp_update


model_runner(multi_agents, multi_agents_params_update, 'multi_agents')
print()
model_runner(mlp_model, mlp_model_params_update, 'standard_mlp')
