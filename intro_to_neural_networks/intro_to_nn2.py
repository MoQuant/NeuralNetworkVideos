import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('ML Model Performance')
ax.set_xlabel('Epochs')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('RMSE')

class ai:

    def __init__(self, inputs, outputs, epochs=100, lr=1):
        self.m = inputs
        self.n = outputs
        self.epochs = epochs
        self.lr = lr

    def __call__(self, x, y):
        x, y = np.array(x), np.array(y)
        b = -1
        
        for epoch in range(self.epochs):
            # Forwad Propigation
            for i in self.axis:
                if i == self.axis[0]:
                    self.L[i] = x.T.dot(self.W[i]) + b
                else:
                    self.L[i] = self.L[i+1].T @ self.W[i] + b
                self.SL[i] = self.sigmoid(self.L[i])

            # Backward Propigation
            gradient = {}
            for i in self.raxis:
                if i == self.raxis[0]:
                    error = (y - self.SL[i])**2
                    delta = error*self.sigmoid(self.L[i], dv=True)
                else:
                    error = self.W[i-1] @ delta
                    delta = error*self.sigmoid(self.L[i], dv=True)
                    
                gradient[i] = delta
                
            for u in self.axis:
                self.W[u] -= self.lr*gradient[u]

    
    def predict(self, x):
        x = np.array(x)
        b = -1
        W = self.W
        L = self.L
        SL = self.SL
        for i in self.axis:
            if i == self.axis[0]:
                L[i] = x.T.dot(self.W[i]) + b
            else:
                L[i] = L[i+1].T @ self.W[i] + b
            SL[i] = self.sigmoid(L[i])
        return SL[self.axis[-1]]


    def sigmoid(self, x, dv=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if dv:
            return f*(1 - f)
        return f

    def build_parameters(self):
        m, n = self.m, self.n

        self.axis = list(range(m, n, -1))
        self.raxis = self.axis[::-1]

        self.W = {}
        self.L = {}
        self.SL = {}

        for i in self.axis:
            self.W[i] = np.random.random((i, i-1))
            self.L[i] = np.zeros(i-1)
            self.SL[i] = np.zeros(i-1)


X = np.random.rand(50, 7)
y = np.random.rand(50, 3)

prop = 0.8
I = int(prop*len(X))

trainX = X[:I]
trainY = y[:I]
testX = X[I:]
testY = y[I:]

H = 12

epochs_list = np.linspace(10, 100, H)
learning_rate = np.linspace(0.0001, 1, H)

Xh, Yh = np.meshgrid(epochs_list, learning_rate)
Zh = np.zeros((H, H))

for i in range(H):
    for j in range(H):
        print(i, j)
        model = ai(7, 3, epochs=int(Xh[i, j]), lr=Yh[i, j])
        model.build_parameters()

        for tx, ty in zip(trainX, trainY):
            model(tx, ty)

        RMSE = 0
        for ttx, tty in zip(testX, testY):
            pred = model.predict(ttx)
            RMSE += sum((tty - pred)**2)

        RMSE = np.sqrt(RMSE / len(testX))
        Zh[i, j] = RMSE

ax.plot_surface(Xh, Yh, Zh, cmap='jet_r')
plt.show()