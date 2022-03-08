import numpy as np


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.poids_synaptiques = 3 * np.random.random((3, 1)) - 2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def propagation(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.poinds_synaptiques))
        return output

    def apprentissage(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.propagation(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derive(output))
            self.poids_synaptiques += adjustments


if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Poids_synaptiques Aléatoires: ")
    print(neural_network.poids_synaptiques)

    entrees = np.array([[0, 0, 1],
                        [1, 1, 1],
                        [1, 0, 1],
                        [0, 1, 1]])

    sorties_d = np.array([[1, 0, 0, 1]]).T

    neural_network.apprentissage(entrees, sorties_d, 10000)

    print("Les poids_synaptiques après l'apprentissage: ")
    print(neural_network.poids_synaptiques)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("Test: entrees = ", A, B, C)
    print("sortie: ")
    print(neural_network.propagation(np.array([A, B, C])))