import numpy as np


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 3 * np.random.random((3, 1)) - 2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derive(self, x):
        return x * (1 - x)

    def propagation(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def apprentissage(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.propagation(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derive(output))
            self.synaptic_weights += adjustments


if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("random synaptic_weights: ")
    print(neural_network.synaptic_weights)

    entrees = np.array([[0, 0, 1],
                        [1, 1, 1],
                        [1, 0, 1],
                        [0, 1, 1]])

    sorties_d = np.array([[1, 0, 0, 1]]).T

    neural_network.apprentissage(entrees, sorties_d, 10000)

    print("synaptic_weights after training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("Test: entrees = ", A, B, C)
    print("sortie: ")
    print(neural_network.propagation(np.array([A, B, C])))
