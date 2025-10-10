import torch
import pickle
import numpy as np
import atomgrad.tensor as atom

def normalization(input_for_model):
    return (input_for_model / 9) - 0.5

def sudoku_dataloader(quizzes_arr, solutions_arr, batch_size, shuffle):
    num_train_samples = quizzes_arr.shape[0]    
    # Total samples
    train_indices = np.arange(num_train_samples)
    if shuffle: np.random.shuffle(train_indices)

    for start_idx in range(0, num_train_samples, batch_size):
        end_idx = start_idx + batch_size

        # Subract 1 to solution array since it output 1-9 to get 0-8
        yield normalization(quizzes_arr[train_indices[start_idx:end_idx]]), solutions_arr[train_indices[start_idx:end_idx]]-1

def load_data_from_csv_file():
    quizzes = np.zeros((1000000, 81), np.int32)
    solutions = np.zeros((1000000, 81), np.int32)
    for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s

    return quizzes, solutions

def save_data_to_pkl_file():
    # Load the data
    quizzes, solutions = load_data_from_csv_file()

    # Split into train/test (e.g., 90% train, 10% test)
    split_idx = int(len(quizzes) * 0.9)

    train_quizzes = quizzes[:split_idx]
    train_solutions = solutions[:split_idx]
    test_quizzes = quizzes[split_idx:]
    test_solutions = solutions[split_idx:]

    # Create the structure matching your unpacking format
    data = (
        (train_quizzes, train_solutions),
        (test_quizzes, test_solutions),
        None  # or any additional metadata you want
    )

    # Save to pickle file
    with open('./dataset/sudoku_task.pkl', 'wb') as f:
        pickle.dump(data, f)

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
        # yield torch.tensor(img_arr[train_indices[start_idx:end_idx]]), text_label_one_hot(label_arr[train_indices[start_idx:end_idx]])
        yield img_arr[train_indices[start_idx:end_idx]], label_arr[train_indices[start_idx:end_idx]]