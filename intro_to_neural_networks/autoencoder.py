import numpy as np
import matplotlib.pyplot as plt

class autoencoder:

    def __init__(self, input_size, epochs=200):
        self.in_size = input_size
        self.epochs = epochs

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
            
            # BackProp
            gradient1 = {}
            for i, j in zip(self.axis, self.raxis):
                if i == self.axis[0]:
                    error = (x - self.SL2[j])**2
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
            
            for i in self.axis:
                #print(self.W1[i].shape, gradient1[i].shape)
                #print(self.W2[i].shape, gradient2[i].shape)
                XW1 = self.W1[i]
                XW1 = (XW1.T - gradient1[i]).T
                self.W1[i] = XW1
                self.W2[i] -= gradient2[i]
            
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

        self.W2 = {}
        self.L2 = {}
        self.SL2 = {}

        for u in self.axis:
            self.W1[u] = np.random.random((u, u-1))
            self.L1[u] = np.zeros(u-1)
            self.SL1[u] = np.zeros(u-1)

        for u, v in zip(self.axis, self.raxis):
            self.W2[u] = np.random.random((v, v-1))
            self.L2[u] = np.zeros(v-1)
            self.SL2[u] = np.zeros(v-1)

    def play_game(self, x):
        x = np.array(x)
        L1 = self.L1
        W1 = self.W1
        SL1 = self.SL1
        W2 = self.W2
        L2 = self.L2
        SL2 = self.SL2 
        for i in self.axis:
            if i == self.axis[0]:
                L1[i] = x.T.dot(W1[i])
            else:
                L1[i] = SL1[i+1].T @ W1[i]
            L1[i] = self.activate(L1[i])
        
        for i, j in zip(self.raxis, self.axis):
            if i == self.raxis[0]:
                L2[j] = W2[j] @ SL1[i]
            else:
                L2[j] = W2[j] @ SL2[j+1]
            SL2[j] = self.activate(L2[j])

        return SL2[self.axis[-1]]

fig = plt.figure()
ax = fig.add_subplot(111)

c = 7
X = np.random.random((100, c))
XT = X[:80]
XY = X[80:]

ai = autoencoder(c)
ai.build()

for xi in XT:
    ai(xi)

for xi in XY:
    ax.cla()
    prediction = ai.play_game(xi)
    ax.set_title('Pre vs. Post Autoencoder')
    ax.plot(xi, color='red')
    ax.plot(prediction, color='limegreen')
    plt.pause(0.15)

plt.show()

