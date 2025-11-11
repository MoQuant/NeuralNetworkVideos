import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class memencoder:

    def __init__(self, input_size, epochs=200, lr=1):
        self.in_size = input_size
        self.epochs = epochs
        self.lr = 1
        self.anomaly = []

    def __call__(self, x):
        x = np.array(x)

        for epoch in range(self.epochs):
            # Forward Prop
            for i in self.axis:
                if i == self.axis[0]:
                    self.L1[i] = x.T.dot(self.W1[i])
                else:
                    self.L1[i] = self.SL1[i+1].T @ self.W1[i]
                self.SL1[i] = self.activate(self.L1[i])
            
            for i, j in zip(self.raxis, self.axis):
                if i == self.raxis[0]:
                    self.L2[j] = self.W2[j] @ self.SL1[i]
                else:
                    self.L2[j] = self.W2[j] @ self.SL2[j+1]
                self.SL2[j] = self.activate(self.L2[j])

            for i in self.memaxis:
                if i == self.memaxis[0]:
                    self.mL[i] = self.SL2[j]
                else:
                    self.mL[i] = self.mSL[i] @ self.mW[i]
                self.mSL[i] = self.activate(self.mL[i])
            
            # BackProp
            gradient1 = {}
            for i, j in zip(self.axis, self.raxis):
                if i == self.axis[0]:
                    error = (x - self.SL2[j])**2
                    self.anomaly.append(np.mean(error))
                    delta = error*self.activate(self.L2[j], dv=True)
                else:
                    #print(delta.shape, self.W2[j-1].shape)
                    error = delta.T @ self.W2[j-1]
                    delta = error*self.activate(self.L2[j], dv=True)
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
            
            gradient3 = {}
            for i in self.rmemaxis[:-1]:
                if i == self.rmemaxis[0]:
                    #print(self.mW[i].shape, delta.shape)
                    error = self.mW[i] @ delta
                    delta = error*self.activate(self.mL[i], dv=True)
                else:
                    error = self.mW[i] @ delta
                    delta = error*self.activate(self.mL[i], dv=True)
                gradient3[i] = delta
                    
            for i in self.axis:
                #print(self.W1[i].shape, gradient1[i].shape)
                #print(self.W2[i].shape, gradient2[i].shape)
                XW1 = self.W1[i]
                XW1 = (XW1.T - self.lr*gradient1[i]).T
                self.W1[i] = XW1
                self.W2[i] -= self.lr*gradient2[i]

            for i in self.memaxis[1:]:
                self.mW[i] -= self.lr*gradient3[i]
            
            #print("Epochs Left: ", self.epochs - epoch)
            

    def activate(self, x, dv=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if dv:
            return f*(1 - f)
        return f

    def build(self):
        self.axis = list(range(self.in_size, 2, -1))
        self.raxis = self.axis[::-1]

        self.memaxis = list(range(2*len(self.axis)))
        self.rmemaxis = self.memaxis[::-1]

        self.W1 = {}
        self.L1 = {}
        self.SL1 = {}

        self.W2 = {}
        self.L2 = {}
        self.SL2 = {}

        self.mW = {}
        self.mL = {}
        self.mSL = {}

        for u in self.axis:
            self.W1[u] = np.random.random((u, u-1))
            self.L1[u] = np.zeros(u-1)
            self.SL1[u] = np.zeros(u-1)

        for u, v in zip(self.axis, self.raxis):
            self.W2[u] = np.random.random((v, v-1))
            self.L2[u] = np.zeros(v-1)
            self.SL2[u] = np.zeros(v-1)

        for a in self.memaxis:
            self.mW[a] = np.random.random((self.in_size-1, self.in_size-1))
            self.mL[a] = np.zeros(self.in_size-1)
            self.mSL[a] = np.zeros(self.in_size-1)


data = [0.13, 0.21, 0.67, 0.12, 0.26, 0.84]

epochs = [10, 50, 100, 200, 500]
learn = [0.0001, 0.001, 0.01, 0.1, 1.0]

x, y = np.meshgrid(epochs, learn)
rmse = np.zeros((len(learn), len(learn)))

n = len(learn)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(n):
    for j in range(n):
        mem_ai = memencoder(len(data), epochs=x[i, j], lr=y[i, j])
        mem_ai.build()
        mem_ai(data)
        root_anomaly = np.sqrt(sum([(i - np.mean(mem_ai.anomaly))**2 for i in mem_ai.anomaly]) / (len(mem_ai.anomaly) - 1))
        rmse[i, j] = root_anomaly
        print(i, j)

ax.set_title('Memory Autoencoder Root Anomaly')
ax.set_xlabel('Epochs')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('Anomaly')

ax.plot_surface(x, y, rmse, cmap='hsv')

plt.show()