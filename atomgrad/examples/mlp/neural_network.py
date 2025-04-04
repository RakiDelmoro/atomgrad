import random
import numpy as np
import atomgrad.atom as atom
import atomgrad.nn_ops as nn_ops
import atomgrad.optimizer as optimizer
import atomgrad.activations_fn.atom_activations as act_ops

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'

def mlp():
    parameters = [] 
    linear_1, params_1 = nn_ops.linear_layer(784, 2000)
    activation = act_ops.relu()
    linear_2, params_2 = nn_ops.linear_layer(2000, 10)

    parameters.extend(params_1)
    parameters.extend(params_2)

    def forward(data):
        linear_1_out = linear_1(data)
        neuron_activation = activation(linear_1_out)
        linear_2_out = linear_2(neuron_activation)

        return linear_2_out
    
    def cross_entropy_loss(prediction, expected):
        act_probabilities = act_ops.softmax()

        def log_softmax(prediction):
            shifted = prediction['data'] - np.max(prediction['data'], axis=-1, keepdims=True)
            return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

        prediction_probs = act_probabilities(prediction)
        grad = (prediction_probs['data'] - expected.numpy())
        avg_loss = -np.mean(np.sum(expected.numpy() * log_softmax(prediction), axis=-1))

        return avg_loss, grad

    def training_phase(dataloader):
        each_batch_loss = []
        step, zero_grad = optimizer.adam(parameters, lr=0.001)
        for input_batched, label_batched in dataloader:
            input_batched = atom.tensor(input_batched, requires_grad=True)
            model_prediction = forward(input_batched)
            avg_loss, gradients = cross_entropy_loss(model_prediction, label_batched)
            
            zero_grad(parameters)
            atom.backward(model_prediction, gradients)
            step(input_batched['data'].shape[0])
            print(avg_loss)
            each_batch_loss.append(avg_loss)

        return np.mean(np.array(each_batch_loss))
    
    def testing_phase(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            batched_image = atom.tensor(batched_image, requires_grad=True)
            batched_label = batched_label.numpy()

            model_pred_probabilities = act_ops.softmax()(forward(batched_image))['data']
            batch_accuracy = (model_pred_probabilities.argmax(axis=-1) == batched_label.argmax(axis=-1)).mean()
            for each in range(len(batched_label)//10):
                model_prediction = model_pred_probabilities[each].argmax()
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
