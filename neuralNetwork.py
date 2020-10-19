import numpy as np
import scipy.special
import json


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes if hasattr(hiddennodes, "__iter__") else [hiddennodes]
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes[0], self.inodes))
        self.whh = []
        self.who = np.random.normal(0.0, pow(self.hnodes[-1], -0.5), (self.onodes, self.hnodes[-1]))
        self.lr = learningrate
        if len(self.hnodes) > 1:
            self._generate_hidden_weights()

        self.activation_function = lambda x: scipy.special.expit(x)

    def _generate_hidden_weights(self):
        print(self.hnodes)
        for index in range(0, len(self.hnodes) - 1):
            weights = np.random.normal(0.0, pow(self.hnodes[index], -0.5),
                                          (self.hnodes[index], self.hnodes[index + 1])).tolist()
            self.whh.append(weights)
        self.whh = np.array(self.whh)
        print(self.whh)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        outputs = [hidden_outputs]
        hidden_layer_outputs = hidden_outputs
        for index, hidden_layer in enumerate(self.whh):
            hidden_layer_inputs = np.dot(hidden_layer, hidden_layer_outputs)
            hidden_layer_outputs = self.activation_function(hidden_layer_inputs)
            outputs.append(hidden_layer_outputs.tolist())
        outputs = np.array(outputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_layer_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # print(final_outputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        hidden_layer_errors = hidden_errors
        errors = [hidden_errors]
        for hidden_layer in self.whh:
            hidden_layer_errors = np.dot(hidden_layer.T, hidden_layer_errors)
            errors.append(hidden_layer_errors.tolist())
        errors = np.array(errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_layer_outputs))
        for index, hidden_layer in enumerate(self.whh):
            # print("len", len(outputs))
            self.whh[index] += self.lr * np.dot(
                (errors[index + 1] * outputs[index + 1] * (1.0 - outputs[index + 1])),
                np.transpose(outputs[index]))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((errors[0] * outputs[0] * (1.0 - outputs[0])),
                                        np.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        hidden_layer_outputs = hidden_outputs
        for index, hidden_layer in enumerate(self.whh):
            hidden_layer_inputs = np.dot(hidden_layer, hidden_layer_outputs)
            hidden_layer_outputs = self.activation_function(hidden_layer_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_layer_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def save(self):
        np.save("save/wih.npy", self.wih)
        np.save("save/whh.npy", self.whh)
        np.save("save/who.npy", self.who)
        with open("save/metadata.json", "w") as file:
            json.dump({
                "inputnodes": self.inodes,
                "hiddennodes": self.hnodes[0],
                "hiddenlayers": len(self.hnodes),
                "outputnodes": self.onodes,
                "learningrate": self.lr
            }, file)


def load():
    wih = np.load("save/wih.npy")
    whh = np.load("save/whh.npy")
    who = np.load("save/who.npy")
    with open("save/metadata.json", "r") as file:
        metadata = json.load(file)
        n = NeuralNetwork(metadata["inputnodes"], metadata["hiddennodes"] * int(metadata["hiddenlayers"]),
                          metadata["outputnodes"], metadata["learningrate"])
        n.wih = wih
        n.whh = whh
        n.who = who
    return n
