import torch
import tqdm
import numpy as np
import torch.nn as nn

def normalization(input_for_model):
    return (input_for_model / 9) - 0.5

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

        # Subract 1 to solution array since it output 1-9 to get 0-8
        yield normalization(quizzes_arr[train_indices[start_idx:end_idx]]), solutions_arr[train_indices[start_idx:end_idx]]-1

class Modelv1(nn.Module):
    def __init__(self, in_channels, grid_row, grid_column, num_classes=9):
        super().__init__()
        
        self.in_channels = in_channels
        self.grid_row = grid_row
        self.grid_column = grid_column
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), padding=1, device='cuda'),
            nn.ReLU(),
            nn.BatchNorm2d(64, device='cuda'),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, device='cuda'),
            nn.ReLU(),
            nn.BatchNorm2d(64, device='cuda'),
            nn.Conv2d(64, 128, kernel_size=(1,1), padding=0, device='cuda'),
            nn.ReLU())

        self.mlp_layer = nn.Sequential(
            nn.Linear(128*grid_row*grid_column, grid_row*grid_column*num_classes, device='cuda'))

    def forward(self, x):
        conv_out = self.conv_layers(x)
        flattened_conv = conv_out.flatten(1, -1)
        mlp_out = self.mlp_layer(flattened_conv)

        return mlp_out.view(-1, self.grid_row*self.grid_column, self.num_classes)
    
class Modelv2(nn.Module):
    def __init__(self, grid_row, grid_column, num_classes=9):
        super().__init__()
        
        self.grid_row = grid_row
        self.grid_column = grid_column
        self.num_classes = num_classes

        self.mlp_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_column*grid_row, 1536),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1536, grid_row*grid_column*num_classes))

    def forward(self, x):
        mlp_out = self.mlp_layers(x)

        return mlp_out.reshape(x.shape[0], self.grid_row*self.grid_column, self.num_classes)

def model_runner():
    DEVICE = 'cuda'
    MAX_EPOCHS = 100
    IN_CHANNELS = 1
    GRID_ROW = 9
    GRID_COLUMN = 9
    BATCH_SIZE = 640
    LEARNING_RATE = 0.001

    quizzes, solutions = load_data_from_csv_file()

    # Split into 90% training and 10% test
    split_idx = int(len(quizzes) * 0.9)

    # Split data to training and test
    train_quizzes, train_solutions = quizzes[:split_idx], solutions[:split_idx]
    test_quizzes, test_solutions = quizzes[split_idx:], solutions[split_idx:]
    
    # model = Modelv1(IN_CHANNELS, GRID_ROW, GRID_COLUMN)
    model = Modelv2(GRID_ROW, GRID_COLUMN)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.to('cuda')

    for _ in (t := tqdm.trange(MAX_EPOCHS)):
        train_loader = sudoku_dataloader(train_quizzes, train_solutions, BATCH_SIZE, shuffle=True)
        test_loader = sudoku_dataloader(test_quizzes, test_solutions, BATCH_SIZE, shuffle=True)

        train_loss = []
        for train_batch_quizzes, train_batch_solutions in train_loader:
            input_for_model = torch.tensor(train_batch_quizzes, device=DEVICE, dtype=torch.float32).view(-1, IN_CHANNELS*GRID_ROW*GRID_COLUMN)
            expected_model_out = torch.tensor(train_batch_solutions, device=DEVICE, dtype=torch.long)

            model_output = model(input_for_model)
            model_loss = loss_fn(model_output.view(-1, 9), expected_model_out.view(-1))

            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            train_loss.append(model_loss.item())

        accuracies = []
        for test_batch_quizzes, test_batch_solutions in test_loader:
            input_for_model = torch.tensor(test_batch_quizzes, device=DEVICE, dtype=torch.float32).view(-1, IN_CHANNELS*GRID_ROW*GRID_COLUMN)
            expected_model_out = torch.tensor(test_batch_solutions, device=DEVICE, dtype=torch.long)

            logits = model(input_for_model)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

            avg_batch_accuracy = (predictions == expected_model_out).float().mean()
            accuracies.append(avg_batch_accuracy.item())

        train_loss = sum(train_loss) / len(train_loss)
        accuracies = sum(accuracies) / len(accuracies)

        t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracies:.4f}')

    torch.save(model, f'sudoku_solver_nn.pth')

def solve_sudoku_task(model, puzzle):
    # Preprocess the input Sudoku puzzle
    puzzle = puzzle.replace('\n', '').replace(' ', '')
    initial_board = torch.tensor([int(j) for j in puzzle], dtype=torch.float32, device='cuda').reshape((1, 1, 9, 9))
    initial_board = normalization(initial_board)

    while True:
        # Use the neural network to predict values for empty cells
        model_output = model(initial_board).softmax(dim=-1)

        prediction = (model_output.argmax(-1) + 1).view(9, 9)
        probabilities = torch.round(torch.max(model_output, dim=-1)[0], decimals=2).view(9, 9)

        initial_board = ((initial_board + 0.5) * 9).reshape((9, 9))
        mask = (initial_board == 0)

        if mask.sum() == 0:
            # Puzzle is solved
            break

        prob_new = probabilities * mask

        indices = torch.argmax(prob_new)
        x, y = (indices // 9), (indices % 9)

        val = prediction[x][y]
        initial_board[x][y] = val
        initial_board = normalization(initial_board.unsqueeze(0).unsqueeze(0))

    initial_board = ((initial_board + 0.5) * 9).reshape((9, 9))
    # Convert the solved puzzle back to a string representation
    solved_puzzle = ''.join(map(str, initial_board.flatten().type(torch.int).tolist()))

    return solved_puzzle

def print_sudoku_grid(puzzle):
    puzzle = puzzle.replace('\n', '').replace(' ', '')
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-"*21)

        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(puzzle[i*9 + j], end=" ")
        print()

game = '''
          0 0 0 7 0 0 0 9 6
          0 0 3 0 6 9 1 7 8
          0 0 7 2 0 0 5 0 0
          0 7 5 0 0 0 0 0 0
          9 0 1 0 0 0 3 0 0
          0 0 0 0 0 0 0 0 0
          0 0 9 0 0 0 0 0 1
          3 1 8 0 2 0 4 0 7
          2 4 0 0 0 5 0 0 0'''

# sudo  ku_solver = torch.load('./SaveModel/sudoku_solver_nn.pth')
# solve_puzzle = solve_sudoku_task(model=sudoku_solver, puzzle=game)
# print_sudoku_grid(solve_puzzle)

model_runner()
