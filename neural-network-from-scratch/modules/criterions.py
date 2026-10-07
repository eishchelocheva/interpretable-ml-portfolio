import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        diff = input - target
        diff_squared = diff ** 2
        diff_squared_size = diff_squared.size
        summ = np.sum(diff_squared)
        mse = summ / diff_squared_size
        return mse

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        diff = input - target
        diff_size = diff.size
        return 2 * diff / diff_size


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        L = self.log_softmax(input)
        batch_size = input.shape[0]
        num_classes = input.shape[1]
        rows = np.arange(batch_size)
        correct = L[rows, target]
        summ = np.sum(correct)
        hard_loss = (-1) * summ / batch_size
        summ_L = np.sum(L)
        smooth_loss = (-1) * summ_L / (batch_size * num_classes)
        eps = self.label_smoothing
        loss = (1 - eps) * hard_loss + eps * smooth_loss
        return loss

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        batch_size = input.shape[0]
        num_classes = input.shape[1]
        rows = np.arange(batch_size)
        eps = self.label_smoothing
        grad_L = np.full_like(input, (-1) * eps / (batch_size * num_classes))
        grad_L[rows, target] -= (1 - eps) / batch_size
        grad_input = self.log_softmax.backward(input, grad_L)
        return grad_input
