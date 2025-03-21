import numpy as np
import torch.nn as nn
import atomgrad.atom as atom
import dpc.forward_pass as f_pass
from dataset.utils import text_label_one_hot

def train_runner(model, dataloader, torch_model):
    for batched_image, batched_label in dataloader:
        # From (Batch, height*width) to (Batch, seq_len, height*width)
        batched_image = atom.tensor(batched_image.view(batched_image.size(0), -1, 28*28).repeat(1, 5, 1).numpy())
        batched_label = atom.tensor(text_label_one_hot(batched_label.numpy()))

        model_outputs = model(batched_image)
        digit_prediction = model_outputs['prediction']
        prediction_error = model_outputs['prediction_frame_error']

def dynamic_predictive_coding(torch_model):
    """Initialize model parameters (from torch model)"""
    # Spatial decoder
    lower_level_network_parameters = atom.tensor(torch_model.lower_level_network.weight.data.numpy(), requires_grad=True)
    # Transition matrices
    Vk_parameters = [atom.tensor(vk.data.numpy(), requires_grad=True) for vk in torch_model.Vk]
    # Hypernetwork
    hyper_network_parameters = [[atom.tensor(layer.weight.data.numpy(), requires_grad=True), atom.tensor(layer.bias.data.numpy(), requires_grad=True)] for layer in torch_model.hyper_network if isinstance(layer, nn.Linear)]
    # Higher-level dynamics
    higher_rnn_parameters = [[atom.tensor(torch_model.higher_rnn.weight_ih.data.numpy(), requires_grad=True), atom.tensor(torch_model.higher_rnn.bias_ih.data.numpy(), requires_grad=True)], [atom.tensor(torch_model.higher_rnn.weight_hh.data.numpy(), requires_grad=True), atom.tensor(torch_model.higher_rnn.bias_hh.data.numpy(), requires_grad=True)]]
    # Digit classifier only
    digit_classifier_parameters = [atom.tensor(torch_model.digit_classifier.weight.data.numpy(), requires_grad=True), atom.tensor(torch_model.digit_classifier.bias.data.numpy(), requires_grad=True)]

    parameters = {'lower_network_parameters': lower_level_network_parameters, 'Vk_parameters': Vk_parameters, 'hyper_network_parameters': hyper_network_parameters, 'higher_rnn_parameters': higher_rnn_parameters, 'digit_classifier_parameters': digit_classifier_parameters}

    def forward(batched_image):
            batch_size, seq_len, _ = batched_image['data'].shape

            # Initialize states
            lower_level_state = atom.tensor(np.zeros(shape=(batch_size, torch_model.lower_dim), dtype=np.float32), requires_grad=True)
            higher_level_state = atom.tensor(np.zeros(shape=(batch_size, torch_model.higher_dim), dtype=np.float32), requires_grad=True)

            # Storage for outputs
            pred_errors = []
            digit_logits = []

            for t in range(seq_len):
                each_frame = batched_image['data'][:, t]

                predicted_frame = f_pass.lower_network_forward(lower_level_state, lower_level_network_parameters)
                avg_error, error = f_pass.prediction_frame_error(predicted_frame, each_frame)

                # Use RnnCell to update the higher level state
                higher_level_state = f_pass.rnn_forward(error, higher_level_state, higher_rnn_parameters, )

                # Generate transition weights
                generated_weights = f_pass.hyper_network_forward(higher_level_state, hyper_network_parameters, )
                value = f_pass.combine_transitions(generated_weights, Vk_parameters)

                # Update lower state with ReLU and noise
                lower_level_state = f_pass.lower_net_state_update(lower_level_state, value)

                # Collect digit logits
                model_prediction = f_pass.classifier_forward(lower_level_state, digit_classifier_parameters)

                digit_logits.append(model_prediction['data'])
                # Store frame prediction error
                pred_errors.append(avg_error)
            
            model_digit_prediction = np.stack(digit_logits).mean(0)
            prediction_error = np.stack(pred_errors).mean()

            return {'prediction': model_digit_prediction, 'prediction_frame_error': prediction_error}, model_prediction, value, parameters
    
    return forward