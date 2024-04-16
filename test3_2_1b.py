# ----------------------
# - read the input data:

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)


# ----------------------
# - network2.py example:
import network2

net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5,
    evaluation_data=validation_data,
    lmbda = 5.0, # this is a regularization parameter
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True)