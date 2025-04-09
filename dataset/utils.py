import torch
import numpy as np

def text_label_one_hot(label_arr):
    one_hot_expected = np.zeros(shape=(label_arr.shape[0], 10), dtype=np.float32)
    one_hot_expected[np.arange(len(label_arr)), label_arr] = 1
    return torch.tensor(one_hot_expected)

def mnist_dataloader(img_arr, label_arr, batch_size, shuffle):
    num_train_samples = img_arr.shape[0]    
    # Total samples
    train_indices = np.arange(num_train_samples)
    if shuffle: np.random.shuffle(train_indices)

    for start_idx in range(0, num_train_samples, batch_size):
        end_idx = start_idx + batch_size
        # yield torch.tensor(img_arr[train_indices[start_idx:end_idx]]), torch.tensor(text_label_one_hot(label_arr[train_indices[start_idx:end_idx]]))
        yield torch.tensor(img_arr[train_indices[start_idx:end_idx]]), text_label_one_hot(label_arr[train_indices[start_idx:end_idx]])
