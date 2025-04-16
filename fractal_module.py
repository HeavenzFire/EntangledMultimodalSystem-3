import torch
import torch.nn as nn
import numpy as np

class FractalBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x + x3  # Fractal connection

class FractalNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([FractalBlock(input_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def mandelbrot_set(self, x_min, x_max, y_min, y_max, width, height, max_iter):
        r1 = np.linspace(x_min, x_max, width)
        r2 = np.linspace(y_min, y_max, height)
        return np.array([[self._mandelbrot(c, max_iter) for c in r1] for r in r2])

    def _mandelbrot(self, c, max_iter):
        z = 0
        n = 0
        while abs(z) <= 2 and n < max_iter:
            z = z*z + c
            n += 1
        return n

    def julia_set(self, c, x_min, x_max, y_min, y_max, width, height, max_iter):
        r1 = np.linspace(x_min, x_max, width)
        r2 = np.linspace(y_min, y_max, height)
        return np.array([[self._julia(complex(x, y), c, max_iter) for x in r1] for y in r2])

    def _julia(self, z, c, max_iter):
        n = 0
        while abs(z) <= 2 and n < max_iter:
            z = z*z + c
            n += 1
        return n

    def sierpinski_triangle(self, size):
        triangle = np.zeros((size, size), dtype=int)
        for y in range(size):
            for x in range(size):
                if x & y == 0:
                    triangle[y, x] = 1
        return triangle

    def barnsley_fern(self, n):
        x, y = 0, 0
        points = np.zeros((n, 2))
        for i in range(n):
            r = np.random.random()
            if r < 0.01:
                x, y = 0, 0.16*y
            elif r < 0.86:
                x, y = 0.85*x + 0.04*y, -0.04*x + 0.85*y + 1.6
            elif r < 0.93:
                x, y = 0.2*x - 0.26*y, 0.23*x + 0.22*y + 1.6
            else:
                x, y = -0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44
            points[i] = [x, y]
        return points
