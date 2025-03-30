import atomgrad.atom as atom

def softmax():
    pass

def relu():
    def forward(data):
        return atom.relu(data, data['requires_grad'])
    
    return forward

def leaky_relu(data):
    pass

def tanh(data):
    pass

def sigmoid(data):
    pass
