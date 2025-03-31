import gzip
import pickle
from dataset.utils import image_data_batching
from atomgrad.examples.torch_dpc.model import DPC
from atomgrad.examples.dpc.model import dynamic_predictive_coding
# from train_runner import debugger_runner, atom_train, atom_test

def main_runner():
    MAX_EPOCHS = 100
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28

    with gzip.open('./dataset/mnist.pkl.gz', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    torch_model = DPC(IMAGE_HEIGHT*IMAGE_WIDTH)
    train_runner, test_runner = dynamic_predictive_coding(torch_model)
    for epoch in range(MAX_EPOCHS):
        train_loader = image_data_batching(train_images, train_labels, batch_size=128, shuffle=True)
        test_loader = image_data_batching(test_images, test_labels, batch_size=128, shuffle=True)

        loss = train_runner(train_loader)
        accuracy = test_runner(test_loader)

        print(f'EPOCH: {epoch+1} Loss: {loss} Accuracy: {accuracy}')

main_runner()
