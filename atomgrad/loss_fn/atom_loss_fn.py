import atomgrad.loss_fn.losses_ops as loss

def cross_entropy_loss():
    def forward(prediction, expected): return loss._cross_entropy_loss(prediction, expected)

    return forward

def mean_squared_loss():
    pass
