from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


# Inside your mynn library implement a scheduler like:
class MultiStepLR:
    def __init__(self, optimizer, milestones, gamma):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count in self.milestones:
            self.optimizer.lr *= 0.95

class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
        self.lr = init_lr  # 必须显式定义 lr 属性

    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        pass
    
    def step(self):
        pass