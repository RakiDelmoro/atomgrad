import atom
import torch
import activations_fn.activations as act

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'
UNDERLINE = "\033[4m"

def deriv_softmax():
    # Init
    logits = torch.randn(3, 3)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    grad = torch.randn_like(probs)

    # TORCH deriv softmax
    t_logits = logits.clone().detach().requires_grad_(True)
    t_probs = torch.nn.functional.softmax(t_logits, dim=-1)
    t_loss_scalar = (t_probs * grad).sum()
    t_loss_scalar.backward()
    t_calculated_grad = t_logits.grad

    # ATOM deriv softmax
    a_logits = atom.cuda_tensor(logits.numpy(), requires_grad=True)
    a_probs = act.softmax()(a_logits)
    a_grad = atom.cuda_tensor(grad.numpy())['data']
    # call backward in atom
    a_probs['grad_fn'](a_grad)
    a_calculated_grad = a_logits['grad']

    # for double checking
    # print(t_calculated_grad)
    # print(a_calculated_grad)

    satisfied = torch.allclose(torch.tensor(a_calculated_grad), t_calculated_grad)

    if satisfied:
        print(f"softmax derivative --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"softmax derivative --->>> {RED}PASSED{RESET}")

deriv_softmax()
