import numpy as np
import cupy
import atomgrad.activations_fn.atom_activations as act_ops

def _cross_entropy_loss(prediction, expected):
    device = prediction['device']
    act_probabilities = act_ops.softmax()
    log_softmax_act = act_ops.log_softmax()

    prediction_probs = act_probabilities(prediction)
    grad = (prediction_probs['data'] - expected.numpy()) if device == 'cpu' else (prediction_probs['data'] - cupy.array(expected.numpy()))
    avg_loss = -np.mean(np.sum(expected.numpy() * log_softmax_act(prediction), axis=-1)) if device == 'cpu' else -cupy.mean(cupy.sum(cupy.array(expected.numpy()) * log_softmax_act(prediction), axis=-1))

    return avg_loss, grad

def _mean_squared_loss(prediction, expected):
    pass
