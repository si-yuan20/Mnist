import copy

from .op import *
import pickle
import math


class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


def determine_padding(filter_shape, output_shape="same"):

    # No padding
    if output_shape == "valid":
        return (0, 0), (0, 0)
    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case output_height = height and stride = 1. This gives the
        # expression for the padding below.
        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))

        return (pad_h1, pad_h2), (pad_w1, pad_w2)


def image_to_column(images, filter_shape, stride, output_shape='same'):
    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)  # Add padding to the image
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')  # Calculate the indices where the dot products are to be applied between weights
    # and the image
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

    # Get content from image at those indices
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols


def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
    # First figure out what the size of the output should be
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)
    return (k, i, j)


class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def layer_name(self):
        """ The name of the layer. Used in model summary. """
        return self.__class__.__name__

    def parameters(self):
        """ The number of trainable parameters used by the layer """
        return 0

    def forward_pass(self, X, training):
        """ Propogates the signal forward in the network """
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        """ Propogates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer. """
        raise NotImplementedError()

    def output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()


def image_to_column(images, filter_shape, stride, padding):
    batch_size, channels, in_h, in_w = images.shape
    filter_h, filter_w = filter_shape

    # 计算输出尺寸
    if padding == 'same':
        out_h = in_h
        out_w = in_w
        pad_h = ((in_h - 1)*stride + filter_h - in_h) // 2
        pad_w = ((in_w - 1)*stride + filter_w - in_w) // 2
    else:
        pad_h, pad_w = 0, 0
        out_h = (in_h - filter_h) // stride + 1
        out_w = (in_w - filter_w) // stride + 1

    # 添加填充
    images_padded = np.pad(images,
                         ((0,0), (0,0),
                         (pad_h, pad_h),
                         (pad_w, pad_w)),
                         mode='constant')

    # 初始化列矩阵
    cols = np.zeros((channels * filter_h * filter_w, batch_size * out_h * out_w))

    # 填充列矩阵
    for y in range(filter_h):
        y_start = y
        y_end = y_start + out_h * stride
        for x in range(filter_w):
            x_start = x
            x_end = x_start + out_w * stride
            patch = images_padded[:, :, y_start:y_end:stride, x_start:x_end:stride]
            cols[y*filter_w + x, :] = patch.reshape(-1, batch_size * out_h * out_w)

    return cols


class Conv2D(Layer):
    """A 2D Convolution Layer.
    Parameters:
    -----------
    n_filters: int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    filter_shape: tuple
        A tuple (filter_height, filter_width).
    input_shape: tuple
        The shape of the expected input of the layer. (batch_size, channels, height, width)
        Only needs to be specified for first layer in the network.
    padding: string
        Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
        matches the input height and width. For 'valid' no padding is added.
    stride: int
        The stride length of the filters during the convolution over the input.
    """
    def __init__(self, n_filters, filter_shape, input_shape=None, padding='same', stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.trainable = True

    def initialize(self, optimizer):
        # Initialize the weights
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / math.sqrt(np.prod(self.filter_shape))
        self.W  = np.random.uniform(-limit, limit, size=(self.n_filters, channels, filter_height, filter_width))
        self.w0 = np.zeros((self.n_filters, 1))
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        batch_size, channels, height, width = X.shape
        self.layer_input = X
        # Turn image shape into column shape
        # (enables dot product between input and weights)
        self.X_col = image_to_column(X, self.filter_shape, stride=self.stride, output_shape=self.padding)
        # Turn weights into column shape
        self.W_col = self.W.reshape((self.n_filters, -1))
        # Calculate output
        output = self.W_col.dot(self.X_col) + self.w0
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(self.output_shape() + (batch_size, ))
        # Redistribute axises so that batch size comes first
        return output.transpose(3,0,1,2)

    def backward_pass(self, accum_grad):
        # Reshape accumulated gradient into column shape
        accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)

        if self.trainable:
            # Take dot product between column shaped accum. gradient and column shape
            # layer input to determine the gradient at the layer with respect to layer weights
            grad_w = accum_grad.dot(self.X_col.T).reshape(self.W.shape)
            # The gradient with respect to bias terms is the sum similarly to in Dense layer
            grad_w0 = np.sum(accum_grad, axis=1, keepdims=True)

            # Update the layers weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Recalculate the gradient which will be propogated back to prev. layer
        accum_grad = self.W_col.T.dot(accum_grad)
        # Reshape from column shape to image shape
        accum_grad = column_to_image(accum_grad,
                                self.layer_input.shape,
                                self.filter_shape,
                                stride=self.stride,
                                output_shape=self.padding)

        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)


class Model_CNN(Layer):
    """
    A model with conv2D layers. Implemented with manual conv2D operations.
    """

    def __init__(self, conv_configs=None, act_func=None, lambda_list=None):
        self.conv_configs = conv_configs  # 格式示例：[{"in_c":1, "out_c":6, "k":5, "stride":1, "pad":0}, ...]
        self.act_func = act_func  # 激活函数类型（如'ReLU'）
        self.lambda_list = lambda_list  # 各层权重衰减强度列表
        self.layers = []  # 层序列（conv2D + 激活）

        # 若提供配置则初始化网络
        if conv_configs and act_func:
            self._build_layers()

    def _build_layers(self):
        """根据配置构建网络层"""
        for i, config in enumerate(self.conv_configs):
            # 创建卷积层
            conv_layer = conv2D(
                in_channels=config["in_c"],
                out_channels=config["out_c"],
                kernel_size=config["k"],
                stride=config.get("stride", 1),
                padding=config.get("pad", 0),
                initialize_method=config.get("init", np.random.normal)
            )

            # 设置权重衰减
            if self.lambda_list and i < len(self.lambda_list):
                conv_layer.weight_decay = True
                conv_layer.weight_decay_lambda = self.lambda_list[i]

            self.layers.append(conv_layer)

            # 非最后一层添加激活函数
            if i < len(self.conv_configs) - 1:
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                # 可扩展其他激活函数

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # 输入格式检查（batch, channels, H, W）
        assert X.ndim == 4, f"Input must be 4D tensor, got {X.ndim}D"

        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_path):
        """加载保存的模型参数"""
        with open(param_path, 'rb') as f:
            saved_data = pickle.load(f)

        # 解析保存数据
        self.conv_configs = saved_data[0]
        self.act_func = saved_data[1]
        self.lambda_list = saved_data[2]

        # 重建网络结构
        self._build_layers()

        # 加载层参数
        param_idx = 0
        for layer in self.layers:
            if isinstance(layer, conv2D):
                layer_params = saved_data[3 + param_idx]
                layer.W = layer_params["W"]
                layer.b = layer_params["b"]
                layer.weight_decay = layer_params["weight_decay"]
                layer.weight_decay_lambda = layer_params["lambda"]
                param_idx += 1

    def save_model(self, save_path):
        """保存模型结构和参数"""
        save_data = [
            self.conv_configs,  # 卷积层配置
            self.act_func,  # 激活类型
            self.lambda_list  # 衰减参数列表
        ]

        # 添加各层参数
        for layer in self.layers:
            if isinstance(layer, conv2D):
                save_data.append({
                    "W": layer.W,
                    "b": layer.b,
                    "weight_decay": layer.weight_decay,
                    "lambda": layer.weight_decay_lambda
                })

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)