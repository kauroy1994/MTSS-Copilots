import math

class Activations:

    @staticmethod
    def sigmoid(x):

        return math.exp(x)/float(1+math.exp(x))