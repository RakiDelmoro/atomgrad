import atomgrad.activations_fn.activations_ops as act

def relu():
    def forward(atom_tensor): return act.relu_ops(atom_tensor)

def softmax():
    def forward(atom_tensor): return act.softmax_ops(atom_tensor)

def leaky_relu():
    def forward(atom_tensor): return act.leaky_relu_ops(atom_tensor)

def tanh():
    def forward(atom_tensor): return act.tanh_ops(atom_tensor)

