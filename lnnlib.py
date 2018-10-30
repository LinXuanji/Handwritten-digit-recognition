# 神经网络试作版
import numpy
import scipy.special


class NeuralNetwork:
    """initialize the neural network"""
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """set number of nodes in each input, hidden, output layer"""
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        """learning rate"""
        self.lr = learning_rate

        """set link weight matrices, wih and who"""
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        """active function"""
        self.active_function = lambda x: scipy.special.expit(x)
        pass

    """train the neural network"""
    def train(self, inputs_list, targets_list):
        """convert inputs list to 2d array"""
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        """calculate signals into hidden layer"""
        hidden_inputs = numpy.dot(self.wih, inputs)
        """calculate signals emerging from hidden layer"""
        hidden_outputs = self.active_function(hidden_inputs)

        """calculate signals into final output layer"""
        final_inputs = numpy.dot(self.who, hidden_outputs)
        """calculate signals emerging from final output layer"""
        final_outputs = self.active_function(final_inputs)

        """errors"""
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    """query the neural network"""
    def query(self, input_list):
        """convert inputs list to 2d array"""
        inputs = numpy.array(input_list, ndmin=2).T
        """calculate signals into hidden layer"""
        hidden_inputs = numpy.dot(self.wih, inputs)
        """calculate signals emerging from hidden layer"""
        hidden_outputs = self.active_function(hidden_inputs)
        """calculate signals into final output layer"""
        final_inputs = numpy.dot(self.who, hidden_outputs)
        """calculate signals emerging from final output layer"""
        final_outputs = self.active_function(final_inputs)
        return final_outputs


# save the neural network
def save(weightih, weightho):
    wih = ','.join(str(x) for x in weightih)
    who = ','.join(str(y) for y in weightho)
    filename = "NNmodelinputw.csv"
    with open(filename, 'w') as file_object:
        file_object.writelines(wih)
    filename = "NNmodelhiddenw.csv"
    with open(filename, 'w') as file_object:
        file_object.writelines(who)


# load the neural network
def load(address):
    addr = address
    array_back = []
    model = open(addr, 'r')
    model_list = model.read()
    model.close()
    list_back = model_list.split(',')
    for m in list_back:
        m = m.replace('[', '')
        m = m.replace(']', '')
        z = m.split()
        z = [float(x) for x in z]
        array_back.append(z)
    callback = numpy.array(array_back)
    return callback


# transfer the neural network
def transfer(wih, who, input_list):
    """convert inputs list to 2d array"""
    inputs = numpy.array(input_list, ndmin=2).T
    """calculate signals into hidden layer"""
    hidden_inputs = numpy.dot(wih, inputs)
    """calculate signals emerging from hidden layer"""
    hidden_outputs = scipy.special.expit(hidden_inputs)
    """calculate signals into final output layer"""
    final_inputs = numpy.dot(who, hidden_outputs)
    """calculate signals emerging from final output layer"""
    final_outputs = scipy.special.expit(final_inputs)
    return final_outputs
