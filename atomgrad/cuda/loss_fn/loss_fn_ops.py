import cupy as cp
import atomgrad.cuda.activations_fn.activations as act_ops

def _cross_entropy_loss(prediction, expected):
    act_probabilities = act_ops.softmax()
    log_softmax_act = act_ops.log_softmax()

    prediction_probs = act_probabilities(prediction)
    grad = (prediction_probs['data'] - cp.array(expected))
    avg_loss = -cp.mean(cp.sum(cp.array(expected) * log_softmax_act(prediction), axis=-1))

    return avg_loss, grad

def _mean_squared_loss(prediction, expected):
    pass
