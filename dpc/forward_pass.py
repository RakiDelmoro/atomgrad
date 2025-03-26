import numpy as np
import numpy as np
import atomgrad.atom as atom

def lower_network_forward(input_data, parameters):
    activation = atom.matmul(input_data, parameters)

    return activation

def hyper_network_forward(input_data, parameters):
    activation = input_data
    for each in range(len(parameters)):
        last_layer = each == len(parameters)-1
        weights = parameters[each][0]
        bias = parameters[each][1]

        pre_activation = atom.add(atom.matmul(activation, weights), bias)
        activation = pre_activation if last_layer else atom.relu(pre_activation)

    return activation

def rnn_forward(input_data, hidden_state, parameters):
    input_to_hidden_params = parameters[0]
    hidden_to_hidden_params = parameters[1]

    input_to_hidden_activation = atom.add(atom.matmul(input_data, input_to_hidden_params[0]), input_to_hidden_params[1])
    hidden_to_hidden_activation = atom.add(atom.matmul(hidden_state, hidden_to_hidden_params[0]), hidden_to_hidden_params[1])

    output = atom.relu(atom.add(input_to_hidden_activation, hidden_to_hidden_activation))

    return output

def combine_transitions_weights(weights, Vk_parameters):
    combined_transitions = []
    for k in range(len(Vk_parameters)):
        transition = atom.mul(weights['data'][:, k].reshape(-1, 1, 1), Vk_parameters[k], weights, Vk_parameters)
        combined_transitions.append(transition)

    return atom.sum_tensor(combined_transitions)

def prediction_frame_error(predicted, expected):
    error = expected - predicted['data']

    return np.mean(error**2), atom.tensor(error, requires_grad=True)

def lower_net_state_update(lower_net_state, value, noise):
    activation = atom.matmul_3d(value, lower_net_state)
    #+ 0.01 * np.random.randn(*lower_net_state['shape'])
    updated_lower_net_state = activation['data'] + noise.numpy()

    return atom.relu(updated_lower_net_state, requires_grad=True)

def classifier_forward(input_data, parameters):
    weights = parameters[0]
    bias = parameters[1]
    activation = atom.add(atom.matmul(input_data, weights), bias)

    return activation
