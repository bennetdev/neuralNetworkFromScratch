import numpy as np
from neuralNetwork import NeuralNetwork

# Print progress bar
# Input: Progress as percentage, as float
def printProgressBar(progressPercentage: float) -> None:
    numberOfHastags = int(progressPercentage * 100 // 5)
    bar = "[" + numberOfHastags * "#" + (20 - numberOfHastags) * " " + "] " + str(round(progressPercentage * 100)) + "%"
    print(bar)

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# learning rate
learning_rate = 0.001

# create instance of neural network
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("data/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 15

nextProgressPrint = 0.05
for e in range(epochs):
    # go through all records in the training data set
    for i, record in enumerate(training_data_list):
        progress = (i + 1 + len(training_data_list) * e) / (len(training_data_list) * epochs)
        if progress > nextProgressPrint or progress == 1:
            printProgressBar(progress)
            nextProgressPrint += 0.05
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
# load the mnist test data CSV file into a list
test_data_file = open("data/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)

# scorecard for how well the network performs, initially empty
scorecardTrain = []

# go through all the records in the test data set
for record in training_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecardTrain.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecardTrain.append(0)

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
scorecard_array_train = np.asarray(scorecardTrain)
print("performanceTest = ", scorecard_array.sum() / scorecard_array.size)
print("performanceTrain = ", scorecard_array_train.sum() / scorecard_array_train.size)
if input("Do you want to save the model?[Y/N] ") == "Y":
    n.save()
