import tqdm 
import torch
import pickle
import torch.nn as nn
from dataset.utils import mnist_dataloader

def squash(tensor, axis=-1):
    """
    Squash function: makes vector length between 0 and 1
    - Short vectors -> nearly zero
    - Long vectors -> nearly 1 (but keeps direction)
    """
    squared_norm = torch.sum(torch.square(tensor), axis=axis, keepdims=True)
    scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
    return scale * tensor

class AgentMarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, agents_outputs, targets):
        agents_probability = torch.linalg.norm(agents_outputs, dim=-1)
        present_loss = torch.nn.functional.relu(self.m_plus - agents_probability) ** 2
        absent_loss = torch.nn.functional.relu(agents_probability - self.m_minus) ** 2
        loss = targets * present_loss + self.lambda_ * (1 - targets) * absent_loss
        total_loss = loss.sum(dim=-1).mean()

        return total_loss

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 64, device='cuda')
        self.linear2 = nn.Linear(64, 64, device='cuda')

    def forward(self, x):
        return squash(self.linear2(self.linear1(x).relu()))

class MultiAgents(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.first_agents = nn.ModuleList([Agent() for _ in range(4)])
        self.W = nn.Parameter(torch.randn(1, 4, 10, 32, 64, device='cuda'))

    def forward(self, x):
        first_agent_outputs = []
        for each in self.first_agents:
            output = each(x)
            first_agent_outputs.append(output)

        stack_first_agents_output = torch.stack(first_agent_outputs, dim=1).unsqueeze(2).unsqueeze(-1)
        self.w_batch = self.W.repeat(x.shape[0], 1, 1, 1, 1)
        each_agent_pred = torch.matmul(self.w_batch, stack_first_agents_output).squeeze(-1)

        logits_for_routing = torch.zeros((x.shape[0], 4, 10, 1), device='cuda')
        for iteration in range(3):
            probabilities_to_route = torch.nn.functional.softmax(logits_for_routing, dim=2)

            second_agents_logit_outputs = (probabilities_to_route * each_agent_pred).sum(dim=1, keepdim=True)
            second_agents_outputs = squash(second_agents_logit_outputs, axis=-1)

            if iteration < 3 - 1:  # Don't update on last iteration
                agreement = (each_agent_pred * second_agents_outputs).sum(dim=-1, keepdim=True)
                logits_for_routing += agreement

        return second_agents_outputs.squeeze(1)
    
def torch_runner():
    MAX_EPOCHS = 20
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    BATCH_SIZE = 4096
    LEARNING_RATE = 0.001

    DEVICE = 'cuda'

    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    model = MultiAgents()
    loss_fn = AgentMarginLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for _ in (t := tqdm.trange(MAX_EPOCHS)):
        train_loader = mnist_dataloader(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = mnist_dataloader(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=True)

        # Training Loop
        train_loss = []
        for batched_image, batched_label in train_loader:
            batched_image = torch.tensor(batched_image, requires_grad=True, device=DEVICE)
            batched_label = torch.nn.functional.one_hot(torch.tensor(batched_label, device=DEVICE), num_classes=10)

            model_prediction = model(batched_image)

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

            model_pred_probabilities = torch.linalg.norm(model(batched_image), dim=-1)
            batch_accuracy = (model_pred_probabilities.argmax(axis=-1) == batched_label).float().mean()

            accuracies.append(batch_accuracy.item())

        train_loss = sum(train_loss) / len(train_loss)
        accuracies = sum(accuracies) / len(accuracies)

        t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracies:.4f}')

torch_runner()
