# ----------------------
# - read the input data:

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)


# ----------------------
# - network2.py example:
import network2

net = network2.Network([784,30,10])
net.SGD(training_data[:10000], 30, 10, 0.1,lmbda = 0.0, 
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_evaluation_cost=True)