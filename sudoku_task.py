import torch
import numpy as np
import torch.nn as nn

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

def sudoku_dataloader(quizzes_arr, solutions_arr, batch_size, shuffle):
    num_train_samples = quizzes_arr.shape[0]    
    # Total samples
    train_indices = np.arange(num_train_samples)
    if shuffle: np.random.shuffle(train_indices)

    for start_idx in range(0, num_train_samples, batch_size):
        end_idx = start_idx + batch_size

        yield quizzes_arr[train_indices[start_idx:end_idx]], solutions_arr[train_indices[start_idx:end_idx]]

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 81*9, kernel_size=(1,1), padding='same')
        )

        self.mlp_layers = nn.Sequential(
            nn.Linear(81*9, 81*9)
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        mlp_out = self.mlp_layers(conv_out)

        return mlp_out

x = torch.randn(2, 81)
model = Model()
print(model(x))
