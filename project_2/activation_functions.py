import numpy as np

class ReLU():
    def __init__(self):
        self.name = "ReLU"

    def __call__(self, input):
        return mp.maximum(0, input)

    def derivative(self, input):
        return np.where(self(input) > 1, 1, self(input))

class LeakyReLU():
    def __init__(self):
        self.name = "LeakyReLU"

    def __call__(self, input):
        np.where(input > 0, input, input*0.01)

    def derivative(self, input):
        np.where(inputs > 0, 1, 0.01)

class Sigmoid():
    def __init__(self):
        self.name = "Sigmoid"

    def lt_zero(input):
        e = np.exp(input.astype(float))
        return e / (e + 1)

    def gt_zero(input):
        return 1 / (np.exp(-input.astype(float)) + 1)

    def __call__(self, input):
        return (np.where(input > 0, gt_zero(input), lt_zero(input)))

    def derivative(self, input):
        return self(input) * (1-self(input))

class Identity():
    def __init__(self):
        self.name = "Identity"

    def __call__(self, input):
        return input

    def derivative(self, input):
        return np.ones(input.shape)