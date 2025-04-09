import random
import numpy as np
import atomgrad.atom as atom

# ATOM for CPU
# import atomgrad.cpu.atom as cpu_atom
import atomgrad.cpu.nn_ops as cpu_nn_ops
import atomgrad.cpu.optimizer as cpu_optimizer
import atomgrad.cpu.loss_fn.loss_fn_nn as cpu_loss_ops
import atomgrad.cpu.activations_fn.activations as cpu_act_ops
# ATOM for GPU
# import atomgrad.cuda.atom as cuda_atom
import atomgrad.cuda.nn_ops as cuda_nn_ops
import atomgrad.cuda.optimizer as cuda_optimizer
import atomgrad.cuda.loss_fn.loss_fn_nn as cuda_loss_ops
import atomgrad.cuda.activations_fn.activations as cuda_act_ops

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'

def mlp(device='cpu'):
    parameters = [] 
    linear_1, params_1 = cpu_nn_ops.linear_layer(784, 2000) if device == 'cpu' else cuda_nn_ops.linear_layer(784, 2000)
    activation = cpu_act_ops.relu() if device == 'cpu' else cuda_act_ops.relu()
    linear_2, params_2 = cpu_nn_ops.linear_layer(2000, 10) if device == 'cpu' else cuda_nn_ops.linear_layer(2000, 10)

    parameters.extend(params_1)
    parameters.extend(params_2)

    loss_fn = cpu_loss_ops.cross_entropy_loss() if device == 'cpu' else cuda_loss_ops.cross_entropy_loss()
    step, zero_grad = cpu_optimizer.adam(parameters, lr=0.001) if device == 'cpu' else cuda_optimizer.adam(parameters, lr=0.001)

    model_pred_as_prob = cpu_act_ops.softmax() if device == 'cpu' else cuda_act_ops.softmax()

    def forward(data):
        linear_1_out = linear_1(data)
        neuron_activation = activation(linear_1_out)
        linear_2_out = linear_2(neuron_activation)

        return linear_2_out

    def training_phase(dataloader):
        each_batch_loss = []
        
        for input_batched, label_batched in dataloader:
            input_batched = atom.tensor(input_batched, requires_grad=True, device=device)
            model_prediction = forward(input_batched)
            avg_loss, gradients = loss_fn(model_prediction, label_batched)
            # print(avg_loss)
            
            zero_grad(parameters)
            atom.backward(model_prediction, gradients)
            step(input_batched['data'].shape[0])
            
            each_batch_loss.append(avg_loss.item())

        return np.mean(np.array(each_batch_loss))

    def testing_phase(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            batched_image = atom.tensor(batched_image, requires_grad=True, device=device)
            batched_label = atom.tensor(batched_label, device=device)

            model_pred_probabilities = model_pred_as_prob(forward(batched_image))['data']
            batch_accuracy = (model_pred_probabilities.argmax(axis=-1) == batched_label['data'].argmax(axis=-1)).mean()
            for each in range(len(batched_label['data'])//10):
                model_prediction = model_pred_probabilities[each].argmax()
                if model_prediction == batched_label['data'][each].argmax(axis=-1): correctness.append((model_prediction.item(), batched_label['data'][each].argmax(axis=-1).item()))
                else: wrongness.append((model_prediction.item(), batched_label['data'][each].argmax(axis=-1).item()))
            # print(f'Number of samples: {i+1}\r', end='', flush=True)
            accuracy.append(np.mean(batch_accuracy.item()))
        # random.shuffle(correctness)
        # random.shuffle(wrongness)
        # print(f'{GREEN}Model Correct Predictions{RESET}')
        # [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correctness) if i < 5]
        # print(f'{RED}Model Wrong Predictions{RESET}')
        # [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(wrongness) if i < 5]
        return np.mean(np.array(accuracy)).item()

    return training_phase, testing_phase
