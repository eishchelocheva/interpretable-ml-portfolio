import numpy as np
from .base import Module
from scipy.special import expit
from scipy.special import ndtr
from scipy.special import logsumexp


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(0, input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        mask = input > 0
        grad_input = grad_output * mask
        return grad_input

class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return expit(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input

class GELU(Module):
    """
    Applies element-wise GELU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return input * ndtr(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        grad_input = grad_output * (ndtr(input) + input * np.exp((-input**2) / 2) / np.sqrt(2 * np.pi))
        return grad_input

class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        max_val = np.max(input, axis=-1, keepdims=True)
        shifted_input = input - max_val
        exp_val = np.exp(shifted_input)
        exp_sum = np.sum(exp_val, axis=-1, keepdims=True)
        return exp_val / exp_sum


    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        gs = grad_output * self.output
        gs_sum = np.sum(gs, axis=-1, keepdims=True)
        grad_input = self.output * (grad_output - gs_sum)
        return grad_input


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return input - logsumexp(input, axis=-1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        S = np.exp(self.output)
        grad_output_sum = np.sum(grad_output, axis=-1, keepdims=True)
        grad_input = grad_output - S * grad_output_sum
        return grad_input
