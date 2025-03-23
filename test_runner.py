import gzip
import pickle
from dataset.utils import image_data_batching
from torch_dpc.model import DPC
from dpc.model import dynamic_predictive_coding
from train_runner import train

def main_runner():
    MAX_EPOCHS = 100
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28

    with gzip.open('./dataset/mnist.pkl.gz', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    torch_model = DPC(IMAGE_HEIGHT*IMAGE_WIDTH)
    atom_model = dynamic_predictive_coding(torch_model)

    for epoch in range(10):
        train_loader = image_data_batching(train_images, train_labels, batch_size=128, shuffle=True)
        test_loader = image_data_batching(test_images, test_labels, batch_size=128, shuffle=True)
        # train_runner(atom_model, train_loader, torch_model)
        train(torch_model, atom_model, train_loader)

        print(f'EPOCH: {epoch+1}')

main_runner()
