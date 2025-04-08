import atomgrad.activations_fn.activations_ops as act

def relu():
    def forward(atom_tensor): return act.relu_ops(atom_tensor)

    return forward

def softmax():
    def forward(atom_tensor): return act.softmax_ops(atom_tensor)

    return forward

def leaky_relu():
    def forward(atom_tensor): return act.leaky_relu_ops(atom_tensor)

    return forward

def tanh():
    def forward(atom_tensor): return act.tanh_ops(atom_tensor)

    return forward

def log_softmax():
    def forward(atom_tensor): return act.log_softmax(atom_tensor)

    return forward
