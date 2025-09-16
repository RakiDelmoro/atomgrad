import tqdm
import pickle
import MLP.config as configs
import atomgrad.atom as atom

import atomgrad.cuda.nn_ops as cuda_nn_ops
import atomgrad.cuda.activations_fn.activations as cuda_act_ops
import atomgrad.cuda.loss_fn.loss_fn_nn as cuda_loss_ops
import atomgrad.cuda.optimizer as cuda_optimizer

from dataset.utils import mnist_dataloader

def atom_mlp_runner():

    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == configs.IMAGE_HEIGHT*configs.IMAGE_WIDTH

    # Initialize Model
    model_parameters = []
    layer_1, layer_1_params = cuda_nn_ops.linear_layer(784, 2000)
    layer_1_activation = cuda_act_ops.relu()
    layer_2, layer_2_params = cuda_nn_ops.linear_layer(2000, 10)

    # Accumulate parameters
    model_parameters.extend(layer_1_params)
    model_parameters.extend(layer_2_params)

    # Loss function
    loss_fn = cuda_loss_ops.cross_entropy_loss()

    # Optimizer
    update_params, zero_grad = cuda_optimizer.adam(model_parameters, lr=configs.LEARNING_RATE)

    # Loop through EPOCHS
    for _ in (t := tqdm.trange(configs.MAX_EPOCHS)):
        train_loader = mnist_dataloader(train_images, train_labels, batch_size=configs.BATCH_SIZE, shuffle=True)
        test_loader = mnist_dataloader(test_images, test_labels, batch_size=configs.BATCH_SIZE, shuffle=True)

        # Training Loop
        each_batch_loss = []
        for batched_image, batched_label in train_loader: 
            # Converting from numpy array to atom tensor
            batched_image = atom.tensor(batched_image, requires_grad=True, device=configs.DEVICE)
            batched_label = atom.tensor(batched_label, device=configs.DEVICE)
            
            # Model forward pass
            model_output = layer_2(layer_1_activation(layer_1(batched_image)))

            # Calculate Model loss
            avg_loss_per_batch, gradients_per_batch = loss_fn(model_output, batched_label)

            # Update model parameters
            zero_grad(model_parameters) # set parameters gradient to zero
            atom.backward(model_output, gradients_per_batch) # Perform backpropagation
            update_params(configs.BATCH_SIZE)

            each_batch_loss.append(avg_loss_per_batch.item())

        # Test Loop
        each_batch_accuracy = []
        for batched_image, batched_label in test_loader:
            # Converting from numpy array to atom tensor
            batched_image = atom.tensor(batched_image, requires_grad=True, device=configs.DEVICE)
            batched_label = atom.tensor(batched_label, device=configs.DEVICE)

            model_prediction = layer_2(layer_1_activation(layer_1(batched_image)))['data']
            accuracy_per_batch = (model_prediction.argmax(axis=-1) == batched_label['data']).mean()

            each_batch_accuracy.append(accuracy_per_batch)

        model_loss = sum(each_batch_loss) / len(each_batch_loss)
        model_accuracy = sum(each_batch_accuracy) / len(each_batch_accuracy)

        t.set_description(f'Loss: {model_loss:.4f} Accuracy: {model_accuracy:.4f}')
