from MLP.atom_mlp import atom_mlp_runner
from MLP.torch_mlp import torch_mlp_runner

# MLP Comparison
torch_mlp_runner()
# Torch GPU behaves:
# memory.total [MiB], memory.used [MiB], memory.free [MiB]
#    8192 MiB,            1238 MiB,          6758 MiB

atom_mlp_runner()
# Atom GPU behaves:
# memory.total [MiB], memory.used [MiB], memory.free [MiB]
#    8192 MiB,            3467 MiB,          4529 MiB
