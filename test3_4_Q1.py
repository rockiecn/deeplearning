# ----------------------
# - read the input data:

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)


# ----------------------
# - network2.py example:
import network2_L1

net = network2_L1.Network([784, 30, 10], cost=network2_L1.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1,
    evaluation_data=validation_data,
    lmbda = 2.0, # this is a regularization parameter
    monitor_evaluation_accuracy=True,)