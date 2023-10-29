from __future__ import annotations
from typing import List, Union
import numpy as np
import scipy.special
import json
import random


# NeuralNetwork
# - number of inputnodes
# - number of hiddennodes or list of number of hidden nodes
# - number of outputnodes
# - weights between input-layer and hidden-layer
# - weights between hidden-layers (not existent for one hidden layer)
# - weights between hidden-layer and output-layer
# - learning rate
# - activation function to use across neural network
class NeuralNetwork:
    def __init__(self, inputnodes: int, hiddennodes: Union[int, List[int]], outputnodes: int, learningrate: float):
        # Nodes are initialized as integers
        self.inodes = inputnodes
        # hidden nodes as list (one list element is one layer)
        self.hnodes = hiddennodes if hasattr(hiddennodes, "__iter__") else [hiddennodes]
        self.onodes = outputnodes

        # init of the weights
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes[0], self.inodes))
        self.whh = []
        self.who = np.random.normal(0.0, pow(self.hnodes[-1], -0.5), (self.onodes, self.hnodes[-1]))
        self.lr = learningrate

        # init biases
        self.biasH = np.random.normal(-1, 1, (self.hnodes[0], 1))
        self.biasesH = []
        self.biasO = np.random.normal(-1, 1, (self.onodes, 1))

        # if there are multiple hidden layers
        if len(self.hnodes) > 1:
            self._generate_hidden_weights()
            self._generate_hidden_biases()

        self.activation_function = lambda x: scipy.special.expit(x)


    # Generates the weights between hidden layers if there are multiple hidden layers
    def _generate_hidden_weights(self) -> None:
        for index in range(0, len(self.hnodes) - 1):
            weights = np.random.normal(0.0, pow(self.hnodes[index], -0.5),
                                          (self.hnodes[index], self.hnodes[index + 1])).tolist()
            self.whh.append(weights)
        self.whh = np.array(self.whh)


    # Generates the biases for every hidden layer if there are multiple hidden layers
    def _generate_hidden_biases(self) -> None:
        # generate one bias for each layer
        for index in range(0, len(self.hnodes) - 1):
            bias = np.random.normal(-1, 1, (self.hnodes[index + 1], 1))
            self.biasesH.append(bias)
        self.biasesH = np.array(self.biasesH)


    # train the neural network
    # Input: input data, Numpy ndarray
    #        Target data, Numpy dnarray
    def train(self, inputs_list: np.ndarray, targets_list: np.ndarray) -> None:
        # convert inputs list to 2d array
        inputsToNeuralNetwork = np.array(inputs_list, ndmin=2).T
        targetOutputs = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        inputsToHidden = np.dot(self.wih, inputsToNeuralNetwork)
        biasHCopy = self.biasH.copy()
        np.concatenate((biasHCopy, self.biasH), axis=1)
        inputsToHidden += biasHCopy
        # calculate the signals emerging from hidden layer
        outputsFromFirstHidden = self.activation_function(inputsToHidden)

        outputsFromAllHidden = [outputsFromFirstHidden]
        outputsFromLastHidden = outputsFromFirstHidden
        for index, hidden_layer in enumerate(self.whh):
            inputsToCurrentHidden = np.dot(hidden_layer, outputsFromLastHidden)
            # add bias to input
            biasCopy = self.biasesH[index].copy()
            np.concatenate((biasCopy, self.biasesH[index]), axis=1)
            inputsToCurrentHidden += biasCopy

            outputsFromLastHidden = self.activation_function(inputsToCurrentHidden)
            outputsFromAllHidden.append(outputsFromLastHidden.tolist())
        outputsFromAllHidden = np.array(outputsFromAllHidden)

        # calculate signals into final output layer
        inputsToOutput = np.dot(self.who, outputsFromLastHidden)
        biasOCopy = self.biasO.copy()
        np.concatenate((biasOCopy, self.biasO), axis=1)
        inputsToOutput += biasOCopy
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(inputsToOutput)
        # print(final_outputs)

        # output layer error is the (target - actual)
        output_errors = targetOutputs - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        hidden_layer_errors = hidden_errors
        errorsFromAllHidden = [hidden_errors]
        for hidden_layer in self.whh:
            hidden_layer_errors = np.dot(hidden_layer.T, hidden_layer_errors)
            errorsFromAllHidden.append(hidden_layer_errors.tolist())
        errorsFromAllHidden = np.array(errorsFromAllHidden)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                      np.transpose(outputsFromLastHidden))
        
        # update weights for the links between hidden layers and biases for hidden layers
        for index, hidden_layer in enumerate(self.whh):
            self.whh[index] += self.lr * np.dot(
                (errorsFromAllHidden[index + 1] * outputsFromAllHidden[index + 1] * (1.0 - outputsFromAllHidden[index + 1])),
                np.transpose(outputsFromAllHidden[index]))
            
            # update biases
            self.biasesH[index] += (errorsFromAllHidden[index + 1]  * (1.0 - outputsFromAllHidden[index + 1])) * self.lr
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((errorsFromAllHidden[0] * outputsFromAllHidden[0] * (1.0 - outputsFromAllHidden[0])),
                                        np.transpose(inputsToNeuralNetwork))

        # update biases
        self.biasO += (output_errors  * (1.0 - final_outputs)) * self.lr
        self.biasH += (hidden_errors * (1.0 - outputsFromFirstHidden)) * self.lr


    # query the neural network for specific input
    # Input: List of input values, np.ndarray with only one dimension (gets converted to 2d array)
    # Output: List of all possible outputs with possibility, Numpy
    def query(self, inputs_list: np.ndarray) -> np.ndarray:
        # convert inputs list to 2d array
        inputsToNeuralNetwork = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        inputsToHidden = np.dot(self.wih, inputsToNeuralNetwork)
        # add bias to hidden_inputs
        biasHCopy = self.biasH.copy()
        np.concatenate((biasHCopy, self.biasH), axis=1)
        inputsToHidden += biasHCopy

        # calculate the signals emerging from hidden layer
        outputsFromFirstHidden = self.activation_function(inputsToHidden)

        # only executes with multiple hidden layers
        outputsFromLastHidden = outputsFromFirstHidden
        # TODO zip?
        for index, hidden_layer in enumerate(self.whh):
            inputsToCurrentHidden = np.dot(hidden_layer, outputsFromLastHidden)
            # add bias to input
            biasCopy = self.biasesH[index].copy()
            np.concatenate((biasCopy, self.biasesH[index]), axis=1)
            inputsToCurrentHidden += biasCopy

            outputsFromLastHidden = self.activation_function(inputsToCurrentHidden)

        # calculate signals into final output layer
        inputsToOutput = np.dot(self.who, outputsFromLastHidden)
        # add bias to inputsToOutput
        biasOCopy = self.biasO.copy()
        np.concatenate((biasOCopy, self.biasO), axis=1)
        inputsToOutput += biasOCopy
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(inputsToOutput)

        return final_outputs


    # save neural network to npy files and metadata to json
    def save(self) -> None:
        np.save("save/wih.npy", self.wih)
        np.save("save/whh.npy", self.whh)
        np.save("save/who.npy", self.who)
        np.save("save/biasH.npy", self.biasH)
        np.save("save/biasO.npy", self.biasO)
        np.save("save/biasesH.npy", self.biasesH)
        with open("save/metadata.json", "w") as file:
            json.dump({
                "inputnodes": self.inodes,
                "hiddennodes": self.hnodes[0],
                "hiddenlayers": len(self.hnodes),
                "outputnodes": self.onodes,
                "learningrate": self.lr
            }, file)

    # Neuroevolution
    
    # Function to (deep) copy the neural networks properties (equals crossover for now)
    # Output: Neural Network with the same properties as self
    def copy(self) -> NeuralNetwork:
        copy = NeuralNetwork(self.inodes, self.hnodes, self.onodes, self.lr)
        # copy weights of self
        copy.whh = self.whh.copy()
        copy.wih = self.wih.copy()
        copy.who = self.who.copy()
        copy.biasH = self.biasH.copy()
        copy.biasesH = self.biasesH.copy()
        copy.biasO = self.biasO.copy()

        return copy


    # Function to create child out of two parent neural networks (including self) by crossing over their weights
    # This implementation chooses every weight off a 50/50 chance, TODO only for one hidden layer right now
    # Input: parent2, as NeuralNetwork
    # Output: new child, as NeuralNetwork
    def crossover(self, parent2: NeuralNetwork) -> NeuralNetwork:
        child = self.copy()
        for i in range(self.hnodes[0]-1):
            for j in range(self.inodes-1):
                child.wih[i,j] = self.wih[i,j] if random.uniform(0,1) <= 0.5 else parent2.wih[i,j]
        for i in range(self.onodes - 1):
            for j in range(self.hnodes[0]-1):
                child.who[i,j] = self.who[i,j] if random.uniform(0,1) <= 0.5 else parent2.who[i,j]

        for i in range(self.hnodes[0]):
            child.biasH[i] = self.biasH[i] if random.uniform(0,1) <= 0.5 else parent2.biasH[i]

        for i in range(self.onodes):
            child.biasO[i] = self.biasO[i] if random.uniform(0,1) <= 0.5 else parent2.biasO[i]


        return child


    # Alter every weight by MUTATION_RATE chance to a new float having a maximum difference to previous value of 0.1
    # Input: chance of mutation, as float
    def mutate(self, MUTATION_RATE: float) -> None:
        # either change value of weight or leave it, depending on randomness and mutation_rate
        # Input: weight value, as float
        # Output: (new) weight value, as float
        def changeValue(value: float) -> float:
            if random.uniform(0.0,1.0) < MUTATION_RATE:
                # re-initialize weight
                return value + random.uniform(-value * 0.1, value * 0.1)
            else:
                return value

        changeValueFunction = np.vectorize(changeValue)
        self.wih = changeValueFunction(self.wih)
        # if whh not empty <=> multiple hidden layers
        if not(not self.whh):
            self.whh = changeValueFunction(self.whh)
        self.who = changeValueFunction(self.who)
        self.biasO = changeValueFunction(self.biasO)
        self.biasH = changeValueFunction(self.biasH)


# Load neural network from npy files and json
# Output: Neural Network with metadata and weights from files
def load() -> NeuralNetwork:
    wih = np.load("save/wih.npy")
    whh = np.load("save/whh.npy")
    who = np.load("save/who.npy")
    biasH = np.load("save/biasH.npy")
    biasesH = np.load("save/biasesH.npy")
    biasO = np.load("save/biasO.npy")
    with open("save/metadata.json", "r") as file:
        metadata = json.load(file)
        n = NeuralNetwork(metadata["inputnodes"], metadata["hiddennodes"] * int(metadata["hiddenlayers"]),
                          metadata["outputnodes"], metadata["learningrate"])
        n.wih = wih
        n.whh = whh
        n.who = who
        n.biasH = biasH
        n.biasesH = biasesH
        n.biasO = biasO
    return n