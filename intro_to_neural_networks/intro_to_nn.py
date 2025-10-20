import numpy as np


class NeuralNetwork:

    def __init__(self, number_of_inputs, number_of_outputs, epochs=200):
        self.a = number_of_inputs
        self.b = number_of_outputs
        self.epochs = epochs

    def __call__(self, x, y):
        x, y = np.array(x), np.array(y)
        b = -1
        for epoch in range(self.epochs):
            
            # Forward Propigation
            for layer in self.axis:
                if layer == self.axis[0]:
                    self.L[layer] = x.T.dot(self.W[layer]) + b
                else:
                    self.L[layer] = self.L[layer+1].T.dot(self.W[layer]) + b
                self.SL[layer] = self.sigmoid(self.L[layer])

            # Backward Propigation
            gradient = {}
            for layer in self.raxis:
                if layer == self.raxis[0]:
                    error = pow(y - self.SL[layer], 2)
                    print(error)
                    delta = error*self.sigmoid(self.L[layer], dv=True)
                    gradient[layer] = delta
                else:
                    #print(self.W[layer-1].shape, delta.shape)
                    error = self.W[layer-1] @ delta
                    delta = error*self.sigmoid(self.L[layer], dv=True)
                    gradient[layer] = delta

            for layer in self.raxis:
                self.W[layer] -= gradient[layer]

            
            

    def build(self):
        a, b = self.a, self.b

        self.axis = list(range(a, b, -1))
        self.raxis = self.axis[::-1]

        self.W = {}
        self.L = {}
        self.SL = {}

        for i in self.axis:
            self.W[i] = np.random.random((i, i-1))
            self.L[i] = np.zeros(i-1)
            self.SL[i] = np.zeros(i-1)

    def sigmoid(self, x, dv=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if dv:
            return f*(1 - f)
        return f



nnet = NeuralNetwork(5, 2)
nnet.build()

x = [0, 1, 1, 0, 1]
y = [0.47, 0.22]

nnet(x, y)




