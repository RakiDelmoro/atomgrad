import torch
import tqdm
import pickle
import numpy as np
import torch.nn as nn
from multi_agents import MultiAgents
from dataset.utils import sudoku_dataloader, normalization

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
    def __init__(self, grid_size):
        super().__init__()
        
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size    

        self.value_embedding = nn.Embedding(10, 64)
        self.row_embedding = nn.Embedding(9, 32)
        self.col_embedding = nn.Embedding(9, 32)
        self.box_embedding = nn.Embedding(9, 32)

        # Combined embedding dimension per cell
        cell_dim = 64 + 3 * 32  # 64 + 32 + 32 + 32 = 160
        input_size = self.n_cells * cell_dim  # 81 * 160 = 12,960

        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 550),
            nn.LayerNorm(550),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second hidden layer
            nn.Linear(550, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Output layer: 81 cells × 9 possible values
            nn.Linear(1024, self.n_cells * grid_size))

    def forward(self, x):
        # x shape: (batch, 9, 9)
        batch_size = x.shape[0]
        device = x.device
        
        # Create position indices
        rows = torch.arange(self.grid_size, device=device).repeat(self.grid_size)
        cols = torch.arange(self.grid_size, device=device).repeat_interleave(self.grid_size)
        boxes = (rows // 3) * 3 + (cols // 3)
        
        # Expand for batch
        rows = rows.unsqueeze(0).expand(batch_size, -1)  # (batch, 81)
        cols = cols.unsqueeze(0).expand(batch_size, -1)
        boxes = boxes.unsqueeze(0).expand(batch_size, -1)
        
        # Flatten puzzle
        flat_x = x.view(batch_size, self.n_cells).long()  # (batch, 81)
        
        # Get embeddings
        value_emb = self.value_embedding(flat_x)  # (batch, 81, 64)
        row_emb = self.row_embedding(rows)        # (batch, 81, 32)
        col_emb = self.col_embedding(cols)        # (batch, 81, 32)
        box_emb = self.box_embedding(boxes)       # (batch, 81, 32)

        # Concatenate all embeddings
        combined = torch.cat([value_emb, row_emb, col_emb, box_emb], dim=-1)
        # Shape: (batch, 81, 160

        # Flatten for MLP
        flat_input = combined.view(batch_size, -1)  # (batch, 10368)
        
        # Forward through network
        logits = self.network(flat_input)  # (batch, 729)
        
        # Reshape to (batch, 81, 9) for output
        return logits.view(batch_size, self.n_cells, self.grid_size)

def model_runner():
    DEVICE = 'cuda'
    MAX_EPOCHS = 50
    IN_CHANNELS = 1
    GRID_ROW = 9
    GRID_COLUMN = 9
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001

    with open('./dataset/sudoku_task.pkl', 'rb') as f:
        ((train_quizzes, train_solutions), (test_quizzes, test_solutions), _) = pickle.load(f)

    # model = Modelv1(IN_CHANNELS, GRID_ROW, GRID_COLUMN)
    model = Modelv2(grid_size=9)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    model.to('cuda')

    for _ in (t := tqdm.trange(MAX_EPOCHS)):
        train_loader = sudoku_dataloader(train_quizzes, train_solutions, BATCH_SIZE, shuffle=True)
        test_loader = sudoku_dataloader(test_quizzes, test_solutions, BATCH_SIZE, shuffle=True)

        train_loss = []
        for train_batch_quizzes, train_batch_solutions in train_loader:
            input_for_model = torch.tensor(train_batch_quizzes, device=DEVICE, dtype=torch.long).view(-1, IN_CHANNELS*GRID_ROW*GRID_COLUMN)
            expected_model_out = torch.tensor(train_batch_solutions, device=DEVICE, dtype=torch.long)

            model_output = model(input_for_model)

            # Calculate loss
            outputs_flat = model_output.view(-1, 9)
            expected_flat = expected_model_out.view(-1)

            # Only calculate loss on empty cells
            train_mask = (input_for_model.view(-1) == 0)
            model_loss = loss_fn(outputs_flat[train_mask], expected_flat[train_mask] - 1)

            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            train_loss.append(model_loss.item())

        accuracies = []
        for test_batch_quizzes, test_batch_solutions in test_loader:
            input_for_model = torch.tensor(test_batch_quizzes, device=DEVICE, dtype=torch.float32).view(-1, IN_CHANNELS*GRID_ROW*GRID_COLUMN)
            expected_model_out = torch.tensor(test_batch_solutions, device=DEVICE, dtype=torch.long)

            logits = model(input_for_model)

            test_mask = (input_for_model.view(-1) == 0)
            flattened_prediction = logits.view(-1, 9)
            flattened_expected = expected_model_out.view(-1)
            
            predictions = torch.argmax(flattened_prediction[test_mask], dim=-1) + 1
            avg_batch_accuracy = (predictions == flattened_expected[test_mask]).float().mean()
            accuracies.append(avg_batch_accuracy.item())

        train_loss = sum(train_loss) / len(train_loss)
        accuracies = sum(accuracies) / len(accuracies)

        t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracies:.4f}')

    torch.save(model, f'./SaveModel/sudoku_solver_nn_v2.pth')

def solve_sudoku_task(model, puzzle):
    # Preprocess the input Sudoku puzzle
    puzzle = puzzle.replace('\n', '').replace(' ', '')
    initial_board = torch.tensor([int(j) for j in puzzle], dtype=torch.float32, device='cuda').view(1, 9, 9)

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

sudoku_solver = torch.load('./SaveModel/sudoku_solver_nn_v2.pth').to('cuda')
sudoku_solver.eval()
solve_puzzle = solve_sudoku_task(model=sudoku_solver, puzzle=game)
print_sudoku_grid(solve_puzzle)

# model_runner()
# print(sum(param.numel() for param in Modelv2(grid_size=9).parameters()))
