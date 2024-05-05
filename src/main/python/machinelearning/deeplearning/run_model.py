import time

from src.main.python.machinelearning.deeplearning import mnist_loader
from src.main.python.machinelearning.deeplearning.network import Network

start = time.process_time()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
print(time.process_time() - start)