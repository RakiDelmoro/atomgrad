import numpy as np
import torch.nn as nn
import atomgrad.atom as atom

def lower_network_forward(input_data, parameters):
    activation = atom.matmul(input_data, parameters)
    return activation

def prediction_frame_error(predicted, expected):
    error = expected - predicted['data']
    return np.mean(error**2), atom.tensor(error, requires_grad=True)

def rnn_forward(input_data, hidden_state, parameters):
    input_to_hidden_params = parameters[0]
    hidden_to_hidden_params = parameters[1]

    input_to_hidden_activation = atom.add(atom.matmul(input_data, input_to_hidden_params[0]), input_to_hidden_params[1])
    hidden_to_hidden_activation = atom.add(atom.matmul(hidden_state, hidden_to_hidden_params[0]), hidden_to_hidden_params[1])
    output = atom.relu(atom.add(input_to_hidden_activation, hidden_to_hidden_activation), requires_grad=True)
    return output

def hyper_network_forward(input_data, parameters):
    activation = input_data
    for each in range(len(parameters)):
        last_layer = each == len(parameters)-1
        weights = parameters[each][0]
        bias = parameters[each][1]
        pre_activation = atom.add(atom.matmul(activation, weights), bias)
        activation = pre_activation if last_layer else atom.relu(pre_activation, requires_grad=True)
    return activation

def combine_transitions_weights(weights, Vk_parameters):
    combined_transitions = atom.broadcasted_mul(weights, Vk_parameters)
    return atom.sum_tensors(combined_transitions)

def lower_net_state_update(lower_net_state, value):
    activation = atom.matmul_3d(value, lower_net_state)
    noise = 0.01 * np.random.randn(*lower_net_state['shape'])
    atom_noise = atom.tensor(noise)
    updated_lower_net_state = atom.add(activation, atom_noise)
    return atom.relu(updated_lower_net_state, requires_grad=True)

def classifier_forward(input_data, parameters):
    weights = parameters[0]
    bias = parameters[1]
    activation = atom.add(atom.matmul(input_data, weights), bias)
    return activation

def dynamic_predictive_coding(torch_model):
    """Initialize model parameters (from torch model)"""
    # Spatial decoder
    lower_level_network_parameters = atom.tensor(torch_model.lower_level_network.weight.data.numpy(), requires_grad=False)
    # Transition matrices
    Vk_parameters = [atom.tensor(vk.data.numpy(), requires_grad=True) for vk in torch_model.Vk]
    # Hypernetwork
    hyper_network_parameters = [[atom.tensor(layer.weight.data.numpy(), requires_grad=True), atom.tensor(layer.bias.data.numpy(), requires_grad=True)] for layer in torch_model.hyper_network if isinstance(layer, nn.Linear)]
    # Higher-level dynamics
    higher_rnn_parameters = [[atom.tensor(torch_model.higher_rnn.weight_ih.data.numpy(), requires_grad=True), atom.tensor(torch_model.higher_rnn.bias_ih.data.numpy(), requires_grad=True)], [atom.tensor(torch_model.higher_rnn.weight_hh.data.numpy(), requires_grad=True), atom.tensor(torch_model.higher_rnn.bias_hh.data.numpy(), requires_grad=True)]]
    # Digit classifier only
    digit_classifier_parameters = [atom.tensor(torch_model.digit_classifier.weight.data.numpy(), requires_grad=True), atom.tensor(torch_model.digit_classifier.bias.data.numpy(), requires_grad=True)]

    parameters = {'lower_network': lower_level_network_parameters, 'vk': Vk_parameters, 'hyper_network': hyper_network_parameters, 'higher_rnn': higher_rnn_parameters, 'digit_classifier': digit_classifier_parameters}

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

                predicted_frame = lower_network_forward(lower_level_state, lower_level_network_parameters)
                avg_error, frame_error = prediction_frame_error(predicted_frame, each_frame)

                # Use RnnCell to update the higher level state
                higher_level_state = rnn_forward(frame_error, higher_level_state, higher_rnn_parameters)

                # Generate transition weights
                generated_weights = hyper_network_forward(higher_level_state, hyper_network_parameters)
                
                value = combine_transitions_weights(generated_weights, Vk_parameters)

                # Update lower state with ReLU and noise
                lower_level_state = lower_net_state_update(lower_level_state, value)

                # Collect digit logits
                model_prediction = classifier_forward(lower_level_state, digit_classifier_parameters)

                digit_logits.append(model_prediction)
                # Store frame prediction error
                pred_errors.append(avg_error)

            model_digit_prediction = atom.mean_tensor(digit_logits, axis=0)
            prediction_error = np.stack(pred_errors).mean()

            return {'prediction': model_digit_prediction, 'prediction_frame_error': prediction_error}, parameters

    return forward
