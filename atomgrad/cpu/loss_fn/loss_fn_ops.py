import numpy as np
import atomgrad.cpu.activations_fn.activations as act_ops

def _cross_entropy_loss(prediction, expected):
    act_probabilities = act_ops.softmax()
    log_softmax_act = act_ops.log_softmax()

    prediction_probs = act_probabilities(prediction)
    grad = (prediction_probs['data'] - expected.numpy())
    avg_loss = -np.mean(np.sum(expected.numpy() * log_softmax_act(prediction), axis=-1))

    return avg_loss, grad

def _mean_squared_loss(prediction, expected):
    pass
