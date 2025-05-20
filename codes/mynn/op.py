from abc import abstractmethod
import numpy as np


class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """

    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False,
                 weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None  # Record the input for backward process.

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay  # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda  # control the intensity of weight decay

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad: np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        # Compute gradients for W and b
        self.grads['W'] = np.dot(self.input.T, grad) / batch_size
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size

        # Apply weight decay
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        # Compute gradient for input
        input_grad = np.dot(grad, self.W.T)
        return input_grad

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        # Initialize weights and biases
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels,))
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out_channels, in_channels, k, k]
        """
        batch_size, in_channels, H, W = X.shape
        pad = self.padding
        k = self.kernel_size
        s = self.stride

        # Apply padding
        X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        self.input = X_padded

        # Calculate output dimensions
        H_out = (H + 2 * pad - k) // s + 1
        W_out = (W + 2 * pad - k) // s + 1

        output = np.zeros((batch_size, self.out_channels, H_out, W_out))

        # Perform convolution
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * s
                h_end = h_start + k
                w_start = w * s
                w_end = w_start + k
                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]
                for c in range(self.out_channels):
                    output[:, c, h, w] = np.sum(X_slice * self.W[c, :, :, :], axis=(1, 2, 3)) + self.b[c]
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X_padded = self.input
        batch_size, out_channels, H_out, W_out = grads.shape
        k = self.kernel_size
        s = self.stride
        pad = self.padding

        # Initialize gradients
        dW = np.zeros_like(self.W)
        db = np.sum(grads, axis=(0, 2, 3))
        dX_padded = np.zeros_like(X_padded)

        # Compute gradients
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * s
                h_end = h_start + k
                w_start = w * s
                w_end = w_start + k
                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]

                # Update dW
                for c in range(out_channels):
                    dW[c] += np.sum(X_slice * grads[:, c, h, w][:, None, None, None], axis=0)

                # Update dX_padded
                for c in range(out_channels):
                    dX_padded[:, :, h_start:h_end, w_start:w_end] += self.W[c][None, :, :, :] * grads[:, c, h, w][:,
                                                                                                None, None, None]

        # Apply weight decay
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        # Average gradients over batch
        self.grads['W'] = dW / batch_size
        self.grads['b'] = db / batch_size

        # Remove padding from input gradient
        if pad == 0:
            dX = dX_padded
        else:
            dX = dX_padded[:, :, pad:-pad, pad:-pad]

        return dX / batch_size

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class ReLU(Layer):
    """
    An activation layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X < 0, 0, X)
        return output

    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """

    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.max_classes = max_classes
        self.probs = None
        self.labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        batch_size = predicts.shape[0]
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts

        # One-hot encode labels
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch_size), labels] = 1

        # Compute loss
        loss = -np.sum(one_hot * np.log(probs + 1e-8)) / batch_size
        self.probs = probs
        self.labels = labels
        return loss

    def backward(self):
        batch_size = self.probs.shape[0]
        one_hot = np.zeros_like(self.probs)
        one_hot[np.arange(batch_size), self.labels] = 1

        if self.has_softmax:
            grad = (self.probs - one_hot) / batch_size
        else:
            grad = (self.probs - one_hot) / batch_size

        self.model.backward(grad)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """

    def __init__(self, model, lambda_reg=1e-4):
        super().__init__()
        self.model = model
        self.lambda_reg = lambda_reg

    def forward(self):
        l2_loss = 0
        for layer in self.model.layers:
            if isinstance(layer, (Linear, conv2D)) and layer.optimizable:
                l2_loss += np.sum(layer.W ** 2)
        return self.lambda_reg * l2_loss

    def backward(self):
        # Gradients are handled in each layer's backward pass
        pass


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition