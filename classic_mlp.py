import pickle
import tqdm 
from dataset.utils import mnist_dataloader
from atomgrad.examples.mlp.neural_network import mlp

def main_runner():
    MAX_EPOCHS = 100
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    BATCH_SIZE = 4096

    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    train_runner, test_runner = mlp(device='cuda')

    for _ in (t := tqdm.trange(MAX_EPOCHS)):
        train_loader = mnist_dataloader(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = mnist_dataloader(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=True)

        train_loss = train_runner(train_loader)
        accuracy = test_runner(test_loader)

        t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracy:.4f}')

main_runner()
