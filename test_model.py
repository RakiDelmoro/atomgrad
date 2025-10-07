import pickle
import torch
import random
import numpy as np
from PIL import Image

def get_correct_label():
    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    img = Image.open('output_image.png')
    image_as_arr = np.asarray(img).reshape(28*28)

    for each in range(len(test_images)):
        np_array = test_images[each]
        scaled_array = ((np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array))) * 255
        if np.allclose(image_as_arr, scaled_array):
            print(test_labels[each])

def fetch_data_from_mnist():
    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    random_int = random.randint(0, test_images.shape[0])
    np_array = test_images[random_int].reshape(28, 28)
    scaled_arr = ((np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array))) * 255
    image = Image.fromarray(scaled_arr.astype(np.uint8))

    # Save the image to a file
    image.save("output_image.png")

def test_model(model, model_path: str):
    # Load Image
    split = model_path.split('.')

    img = Image.open('output_image.png')
    image_as_arr = np.asarray(img).reshape(1, 28*28)
    # normalize pixel value to range [0, 1]
    normalized_arr = (image_as_arr - np.min(image_as_arr)) / (np.max(image_as_arr) - np.min(image_as_arr))

    # Model prediction
    prediction = model(torch.tensor(normalized_arr, dtype=torch.float32, device='cuda'))
    print(f'MODEL: {split[0]} Predicted: {prediction.argmax(-1).item()}')

# fetch_data_from_mnist()

# Multi Agents
multi_agents = torch.load('multi_agents.pth') # Multi agent parameters save with 97.1% accuracy
# print(sum(param.numel() for param in multi_agents.parameters()))

# Standard MLP
standard_mlp = torch.load('standard_mlp.pth') # Standard MLP parameters save with 97.5% accuracy
# print(sum(param.numel() for param in standard_mlp.parameters()))

test_model(multi_agents, 'multi_agents.pth')
test_model(standard_mlp, 'standard_mlp.pth')
