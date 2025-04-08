import random
import numpy as np
import atomgrad.atom as a
import atomgrad.nn_ops as nn_ops
import atomgrad.optimizer as optimizer
import atomgrad.loss_fn.atom_loss_fn as loss_ops
import atomgrad.activations_fn.atom_activations as act_ops
from atomgrad.tensor import atom

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'

def mlp():
    parameters = [] 
    linear_1, params_1 = nn_ops.linear_layer(784, 2000, device='cuda')
    activation = act_ops.relu()
    linear_2, params_2 = nn_ops.linear_layer(2000, 10, device='cuda')

    parameters.extend(params_1)
    parameters.extend(params_2)

    loss_fn = loss_ops.cross_entropy_loss()

    def forward(data):
        linear_1_out = linear_1(data)
        neuron_activation = activation(linear_1_out)
        linear_2_out = linear_2(neuron_activation)

        return linear_2_out

    def training_phase(dataloader):
        each_batch_loss = []
        step, zero_grad = optimizer.adam(parameters, lr=0.001, device='cuda')
        for input_batched, label_batched in dataloader:
            input_batched = atom(input_batched, requires_grad=True, device='cuda')
            model_prediction = forward(input_batched)
            avg_loss, gradients = loss_fn(model_prediction, label_batched)

            zero_grad(parameters)
            a.backward(model_prediction, gradients)
            step(input_batched['data'].shape[0])
            each_batch_loss.append(avg_loss.item())

        return np.mean(np.array(each_batch_loss))

    def testing_phase(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            batched_image = atom(batched_image, requires_grad=True, device='cuda')
            batched_label = atom(batched_label.numpy(), device='cuda')

            model_pred_probabilities = act_ops.softmax()(forward(batched_image))['data']
            batch_accuracy = (model_pred_probabilities.argmax(axis=-1) == batched_label['data'].argmax(axis=-1)).mean()
            for each in range(len(batched_label['data'])//10):
                model_prediction = model_pred_probabilities[each].argmax()
                if model_prediction == batched_label['data'][each].argmax(axis=-1): correctness.append((model_prediction.item(), batched_label['data'][each].argmax(axis=-1).item()))
                else: wrongness.append((model_prediction.item(), batched_label['data'][each].argmax(axis=-1).item()))
            print(f'Number of samples: {i+1}\r', end='', flush=True)
            accuracy.append(np.mean(batch_accuracy.item()))
        random.shuffle(correctness)
        random.shuffle(wrongness)
        print(f'{GREEN}Model Correct Predictions{RESET}')
        [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correctness) if i < 5]
        print(f'{RED}Model Wrong Predictions{RESET}')
        [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(wrongness) if i < 5]
        return np.mean(np.array(accuracy)).item()

    return training_phase, testing_phase
