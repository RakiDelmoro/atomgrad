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

def test_two_different_model():
    MAX_EPOCHS = 20
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    BATCH_SIZE = 4096
    LEARNING_RATE = 0.001
    DEVICE = 'cuda'

    multi_agents_model = MultiAgents()
    standard_mlp_model = StandardMLP()

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

            # # Multi agent architecture training
            # model1_prediction = multi_agents_model(batched_image)
            # loss = multi_agent_update(multi_agents_model, model1_prediction, batched_label, LEARNING_RATE)
            
            # Standard MLP architecture training
            model2_prediction = standard_mlp_model(batched_image)
            loss = standard_mlp_update(standard_mlp_model, model2_prediction, batched_label, LEARNING_RATE)

            train_loss.append(loss.item())

        # Test Loop
        accuracies = []
        for batched_image, batched_label in test_loader:
            batched_image = torch.tensor(batched_image, requires_grad=True, device=DEVICE)
            batched_label = torch.tensor(batched_label, device=DEVICE)

            # # Multi agent architecture prediction
            # model_pred_probabilities = torch.linalg.norm(multi_agents_model(batched_image), dim=-1)

            # Standard MLP archictecture prediction
            model_pred_probabilities = standard_mlp_model(batched_image)

            batch_accuracy = (model_pred_probabilities.argmax(axis=-1) == batched_label).float().mean()

            accuracies.append(batch_accuracy.item())

        train_loss = sum(train_loss) / len(train_loss)
        accuracies = sum(accuracies) / len(accuracies)

        t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracies:.4f}')

test_two_different_model()
