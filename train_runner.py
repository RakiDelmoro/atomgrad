import torch
import numpy as np
import torch.nn as nn
import atomgrad.atom as atom
from dataset.utils import text_label_one_hot

def train(torch_model, atom_model, loader):
    optimizer = torch.optim.AdamW(torch_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    each_batch_losses = []
    for input_image, digits in loader:
        input_image = input_image.view(input_image.size(0), -1, 28*28).repeat(1, 5, 1)
        digits = text_label_one_hot(digits)

        # '''TORCH outputs'''
        torch_outputs, torch_frame_pred, torch_frame_error = torch_model(input_image)
        torch_frame_error.retain_grad()
        torch_frame_pred.retain_grad()
        torch_loss_pred_frame = torch_outputs['prediction_error']

        '''ATOM outputs'''
        atom_outputs, atom_prediction, atom_frame_error, atom_frame_pred, parameters = atom_model(atom.tensor(input_image.numpy()))
        atom_prediction = atom.tensor(atom_outputs['prediction'], requires_grad=True)
        atom_loss_pred_frame = atom_outputs['prediction_frame_error']
        atom_frame_grad = (-2 * (atom_frame_error['data'] / (atom_frame_error['shape'][1]*atom_frame_error['shape'][0]*5)))
        atom_pred_grad = ((atom.softmax(atom_prediction) - digits.numpy()))

        torch_loss_digit = loss_fn(torch_outputs['digit_prediction'], digits)
        torch_loss_pred = torch_outputs['prediction_error']
        # Combine losses with regularization
        loss = (torch_loss_digit + 0.1 * torch_loss_pred + 0.01 * (torch_model.lower_level_network.weight.pow(2).mean()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        each_batch_losses.append(loss.item())
        atom.backward(atom_prediction, atom_pred_grad)
        atom.backward(atom_frame_pred, atom_frame_grad)
        print()

    return torch.mean(torch.tensor(each_batch_losses)).item()
