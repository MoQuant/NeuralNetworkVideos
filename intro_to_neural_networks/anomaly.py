import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

def pyautoencoder(x, epochs=200, lr=0.1):
    input_size = len(x)
    x = torch.tensor(x, dtype=torch.float32)
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, input_size-1),
                nn.Sigmoid(),
                nn.Linear(input_size-1, input_size-2),
                nn.Sigmoid(),
                nn.Linear(input_size-2, input_size-3),
                nn.Sigmoid()
            )
            self.decoder = nn.Sequential(
                nn.Linear(3, 4),
                nn.Sigmoid(),
                nn.Linear(4, 5),
                nn.Sigmoid(),
                nn.Linear(5, 6),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))
        
    
    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    anomaly = []
    for epoch in range(epochs):
        out = model(x)
        loss = criterion(out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epochs Left: ", epochs - epoch)
        anomaly.append(loss.item())
    
    return anomaly

class autoencoder:

    def __init__(self, input_size, epochs=200, lr=0.1):
        self.in_size = input_size
        self.lr = lr
        self.epochs = epochs
        self.anomaly = []

    def __call__(self, x):
        x = np.array(x)
        self.anomaly = []
        for epoch in range(self.epochs):
            # Forward Prop
            for i in self.axis:
                if i == self.axis[0]:
                    self.L1[i] = x.T.dot(self.W1[i]) + self.b1[i]
                else:
                    self.L1[i] = self.SL1[i+1].T @ self.W1[i] + self.b1[i]
                self.SL1[i] = self.activate(self.L1[i])
            
            for i, j in zip(self.raxis, self.axis):
                if i == self.raxis[0]:
                    self.L2[j] = self.W2[j] @ self.SL1[i] + self.b2[j]
                else:
                    self.L2[j] = self.W2[j] @ self.SL2[j+1] + self.b2[j]
                self.SL2[j] = self.activate(self.L2[j])
            
            # BackProp
            gradient1 = {}
            for i, j in zip(self.axis, self.raxis):
                if i == self.axis[0]:
                    error = (x - self.SL2[j])**2
                    self.anomaly.append(sum(error))
                    delta = error*self.activate(self.L2[j], dv=True)
                else:
                    #print(delta.shape, self.W2[j-1].shape)
                    error = delta.T @ self.W2[j-1]
                    delta = error*self.activate(self.L2[j], dv=True)
                self.b1[i] -= self.lr*sum(error)
                gradient1[i] = delta

            gradient2 = {}   
            for i, j in zip(self.raxis, self.axis):
                #print(self.L1[i].shape, self.W1[i].shape, delta.shape)
                if i == self.raxis[0]:
                    error = delta.T @ self.W2[j]
                    delta = error*self.activate(self.L1[i], dv=True)
                else:
                    error = self.W1[i-1] @ delta
                    delta = error*self.activate(self.L1[i], dv=True)
                gradient2[j] = delta 
                self.b2[j] -= self.lr*sum(error)
            
            for i in self.axis:
                #print(self.W1[i].shape, gradient1[i].shape)
                #print(self.W2[i].shape, gradient2[i].shape)
                XW1 = self.W1[i]
                XW1 = (XW1.T - self.lr*gradient1[i]).T
                self.W1[i] = XW1
                self.W2[i] -= self.lr*gradient2[i]
            
            print("Epochs Left: ", self.epochs - epoch)
            

    def activate(self, x, dv=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if dv:
            return f*(1 - f)
        return f

    def build(self):
        self.axis = list(range(self.in_size, 2, -1))
        self.raxis = self.axis[::-1]

        self.W1 = {}
        self.L1 = {}
        self.SL1 = {}
        self.b1 = {}

        self.W2 = {}
        self.L2 = {}
        self.SL2 = {}
        self.b2 = {}

        for u in self.axis:
            self.W1[u] = np.random.random((u, u-1))
            self.L1[u] = np.zeros(u-1)
            self.SL1[u] = np.zeros(u-1)
            self.b1[u] = -1.0

        for u, v in zip(self.axis, self.raxis):
            self.W2[u] = np.random.random((v, v-1))
            self.L2[u] = np.zeros(v-1)
            self.SL2[u] = np.zeros(v-1)
            self.b2[u] = -1.0


fig = plt.figure()
ax = fig.add_subplot(121)
ay = fig.add_subplot(122)

x1 = [0.23, 0.66, 0.45, 0.03, 0.15, 0.8]
lr = 0.001

c = len(x1)
ai = autoencoder(c, lr=lr)
ai.build()

ai(x1)

pyanomaly = pyautoencoder(x1, lr=lr)

anom = []
pnom = []
for anomaly, panomaly in zip(ai.anomaly, pyanomaly):
    anom.append(anomaly)
    pnom.append(panomaly)
    ax.cla()
    ay.cla()
    ax.set_title('My Autoencoder')
    ay.set_title('PyTorch Autoencoder')
    ax.plot(anom, color='red', label='Homebrew')
    ay.plot(pnom, color='blue', label='PyTorch')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Anomaly')
    plt.pause(0.1)

plt.show()


