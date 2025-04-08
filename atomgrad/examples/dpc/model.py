import random
import numpy as np
import torch.nn as nn
import atomgrad.atom as atom
import atomgrad.activations_fn.atom_activations as act
import atomgrad.examples.dpc.optimizer as opt

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'

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
    output = act.relu()(atom.add(input_to_hidden_activation, hidden_to_hidden_activation))
    return output

def hyper_network_forward(input_data, parameters):
    activation = input_data
    for each in range(len(parameters)):
        last_layer = each == len(parameters)-1
        weights = parameters[each][0]
        bias = parameters[each][1]
        pre_activation = atom.add(atom.matmul(activation, weights), bias)
        activation = pre_activation if last_layer else act.relu()(pre_activation)
    return activation

def combine_transitions_weights(weights, Vk_parameters):
    combined_transitions = atom.broadcasted_mul(weights, Vk_parameters)
    return atom.sum_tensors(combined_transitions)

def lower_net_state_update(lower_net_state, value):
    activation = atom.matmul(value, lower_net_state)
    noise = 0.01 * np.random.randn(*lower_net_state['shape'])
    atom_noise = atom.tensor(noise)
    updated_lower_net_state = atom.add(activation, atom_noise)
    return act.relu()(updated_lower_net_state)

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

            return {'prediction': model_digit_prediction, 'prediction_frame_error': prediction_error}

    # Handle this
    def cross_entropy_loss(prediction, expected):
        def log_softmax(prediction):
            shifted = prediction['data'] - np.max(prediction['data'], axis=-1, keepdims=True)
            return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        # if function used is from atom it always have a type of dict we need the data of that tensor
        prediction_probs = act.softmax()(prediction)['data']
        grad = (prediction_probs - expected.numpy())
        avg_loss = -np.mean(np.sum(expected.numpy() * log_softmax(prediction), axis=-1))

        return avg_loss, grad

    def training_phase(dataloader):
        each_batch_loss = []
        for input_image, digits in dataloader:
            input_image = input_image.view(input_image.size(0), -1, 28*28).repeat(1, 5, 1)
            atom_outputs = forward(atom.tensor(input_image.numpy()))
            atom_loss, atom_pred_grad = cross_entropy_loss(atom_outputs['prediction'], digits)

            opt.zero_grad(parameters)
            atom.backward(atom_outputs['prediction'], atom_pred_grad)
            opt.step(input_image.shape[0], parameters)
            each_batch_loss.append(atom_loss)
            print(atom_loss)

        return np.mean(np.array(each_batch_loss))

    def testing_phase(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            batched_image = batched_image.view(batched_image.size(0), -1, 28*28).repeat(1, 5, 1)
            batched_label = batched_label.numpy()

            model_pred_probabilities = forward(atom.tensor(batched_image, requires_grad=False))['prediction']
            # clean computation graph
            cleaner(model_pred_probabilities)
            batch_accuracy = (model_pred_probabilities['data'].argmax(axis=-1) == batched_label.argmax(axis=-1)).mean()
            for each in range(len(batched_label)//10):
                model_prediction = model_pred_probabilities['data'][each].argmax()
                if model_prediction == batched_label[each].argmax(axis=-1): correctness.append((model_prediction.item(), batched_label[each].argmax(axis=-1).item()))
                else: wrongness.append((model_prediction.item(), batched_label[each].argmax(axis=-1).item()))
            print(f'Number of samples: {i+1}\r', end='', flush=True)
            accuracy.append(np.mean(batch_accuracy))
        random.shuffle(correctness)
        random.shuffle(wrongness)
        print(f'{GREEN}Model Correct Predictions{RESET}')
        [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correctness) if i < 5]
        print(f'{RED}Model Wrong Predictions{RESET}')
        [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(wrongness) if i < 5]

        return np.mean(np.array(accuracy)).item()

    return training_phase, testing_phase

#Create a file and a function to computional graph for cleaning (Important for resolving memory problem)
def deepwalk(nodes):
    """Build topological order starting from the given node."""
    visited = set()
    topo = []

    def visit(node):
        node_identity = id(node)
        if node_identity not in visited:
            visited.add(node_identity)
            if type(node) == list:
                for each in node:
                    for depends_on in each['depends_on']: visit(depends_on)
            else:
                for depends_on in node['depends_on']: visit(depends_on)
            topo.append(node)
    visit(nodes)

    return topo

# After forward pass throw away all the compution graph for memory efficiency
def cleaner(atom_tensor):
    """Clean compution graph"""
    topo = deepwalk(atom_tensor)
    for node in reversed(topo):
        if type(node) == list:
            for each in node:
                each['depends_on'] = []
                each['grad_fn'] = None
        else:
            node['depends_on'] = []
            node['grad_fn'] = None
