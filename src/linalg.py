import numpy as np
import torch
from torch.autograd import Function


class LogDet(Function):
    def forward(self, L):
        self.save_for_backward(L)

        # Avoids determinant overflow/underflow
        sign, logdet = np.linalg.slogdet(L.numpy())

        return torch.from_numpy(np.asarray([logdet]))

    def backward(self, grad_output):
        (L, ) = self.saved_tensors
        grad_L = None

        if self.needs_input_grad[0]:
            # The gradient of log(L) is defined as (L^-1)^T
            # Find the pseudoinverse to avoid problems with singular matrices
            grad_L = torch.Tensor(np.linalg.pinv(L.numpy()).T)

        return grad_L

    def __str__(self):
        return "LogDet"


def logdet(L):
    return LogDet()(L)
