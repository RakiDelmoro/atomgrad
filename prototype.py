import torch

def squash(tensor, axis=-1):
    """
    Squash function: makes vector length between 0 and 1
    - Short vectors -> nearly zero
    - Long vectors -> nearly 1 (but keeps direction)
    """
    squared_norm = torch.sum(torch.square(tensor), axis=axis, keepdims=True)
    scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
    return scale * tensor

def dynamic_routing(first_agents_output, each_agents_w, num_iterations=3):
    # Each agents has MLP inside
    # first_agents_ouput, shape (3, 16) first dim represent how many agents each has 16 feature size
    # each_agents_w, shape (3, 10, 5, 16) this represent each agent weight use to forward to the next agents. where it need to pass it's info through 10 agents each has a 5 feature size

    # each agents in first, has prediction of what each agents output in the second would be
    # first agents -> second agents prediction
    # shape, (3, 10, 5)
    predict_second_agents_outputs = torch.einsum('ld,lhod->lho', first_agents_output, each_agents_w)
    
    # This step determine how much each agents in first should route to each agents in second
    # You can think of it as a voting array where it start with equal vote to all second agents
    # shape, (first num agents, second num agents)
    logits_for_routing = torch.zeros((3, 10)) # Initially start with zeros this means (equal routing to all agents in the second)
    for iteration in range(num_iterations):
        # Softmax to get routing coefficients (c)
        # Convert logits to probabilities
        # Each row sums to 1 (first agents distributes to second agents)
        probabilities_to_route = torch.exp(logits_for_routing) / torch.sum(torch.exp(logits_for_routing), axis=1, keepdims=True)
        
        # each second agents receive all the prediction made by first agents
        # shape, (10, 5)
        second_agents_logit_outputs = torch.einsum('lh,lhd->hd', probabilities_to_route, predict_second_agents_outputs)

        # Apply squash function 
        # Make vectors have length between 0 and 1
        # Length represents probability that the feature exists
        second_agents_outputs = squash(second_agents_logit_outputs, axis=-1)
        # UNPRINT: to see the example lengths of each agents in the second
        # print(f"Example lengths: {[torch.linalg.norm(second_agents_outputs[i]) for i in range(3)]}")

        # Increase routing to second agents that "agree" with predictions from first agents
        # Agreement = dot product between prediction and actual output
        if iteration < num_iterations - 1:  # Don't update on last iteration
            for i in range(3):  # For each first agents
                for j in range(10):  # For each second agents
                    # If prediction is close to output, increase routing
                    agreement = torch.dot(predict_second_agents_outputs[i, j], second_agents_outputs[j])
                    logits_for_routing[i, j] += agreement
            
            print(f"Updated logits for routing based on agreement between predictions and outputs")

    return second_agents_outputs, logits_for_routing

if __name__ == "__main__":   
    # Initialize random lower capsule outputs (3 capsules, 16D each)
    first_agents_outputs = torch.randn(3, 16)
    
    # Initialize random weight matrices
    # Shape: (3 lower, 10 higher, 5 output dims, 16 input dims)
    each_agents_weights = torch.randn(3, 10, 5, 16) * 0.1
    
    # Run dynamic routing
    output, voting = dynamic_routing(first_agents_outputs, each_agents_weights, num_iterations=3)


    # x = torch.randn(2, 10)
    # print(x)
    # print(torch.linalg.norm(squash(x), keepdim=True, dim=-1))
    # print(torch.norm(squash(x), keepdim=True, dim=-1))

    # print(voting)

# Example voting array, shape (first num agents, second num agents):
# Example in row 0(first agent in first) should route more of it's information to 9th agent in second
# row 1(second agent in first) should route more of it's information to 2nd agent in second
"""tensor([[ 0.0384,  0.0220, -0.0051,  0.0104,  0.0368, -0.0030,  0.0106,  0.0042,
          0.0783,  0.0512],
        [ 0.0157,  0.1286,  0.0227,  0.0212,  0.0414,  0.0246,  0.0179,  0.0088,
          0.0670,  0.0768],
        [ 0.0196,  0.0629,  0.0386,  0.0081,  0.0470,  0.0098,  0.0138,  0.0045,
          0.0637, -0.0110]])
"""
# first dim represent total agents each representing 0-9 digits
# second dim represent the activation of each agent this activation is a vector we can translate this to probabilities by
# computing the length of this vector and keeping it's direction
last_agents = torch.randn(10, 5)

agents_as_pred = torch.norm(squash(last_agents), keepdim=True, dim=-1).T # Translate to probabilities
expected_output = torch.nn.functional.one_hot(torch.randint(0, 10, size=(1,)), num_classes=10).float()

print(expected_output) # tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
print(agents_as_pred) # tensor([[0.8632, 0.5487, 0.7186, 0.8359, 0.6455, 0.6512, 0.8465, 0.7687, 0.5604,0.8465]])
