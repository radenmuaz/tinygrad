from tinygrad.tensor import Tensor
import math

def fft(x):
    if x.size(-1) != 2:
        x = Tensor.stack([x, x.zeros_like()], dim=-1)
    n_orig: int = x.size(-2)
    n_fft: int = 1 << (n_orig - 1).bit_length()
    x_padded = x.pad((0, n_fft - n_orig))

    # Create butterfly twiddle factors
    k = Tensor.arange(n_fft // 2).view(1, -1)
    theta = -2 * math.pi * k / n_fft
    twiddle_real = theta.cos()
    twiddle_imag = theta.sin()

    # Perform FFT
    y = x_padded.view(*x_padded.size()[:-1], 2)
    for _ in range(int(math.log2(n_fft))):
        y = y.view(*y.size()[:-2], -1, 2, 2)
        y = Tensor.stack([y[..., 0] + twiddle_real * y[..., 1],
                        y[..., 0] + twiddle_imag * y[..., 1]], dim=-1)
        y = y.view(*y.size()[:-2], -1, 2)

    # Reshape and crop the output
    y = y.view(x_padded.size())
    y = y[..., :n_orig]

    return y

def ifft(X_real, X_imag):
    # Get the input size
    n_orig = X_real.size(-1)
    n_fft = X_real.size(-1)

    # Create butterfly twiddle factors
    k = Tensor.arange(n_fft // 2).view(1, -1)
    theta = 2 * math.pi * k / n_fft
    twiddle_real = theta.cos()
    twiddle_imag = theta.sin()

    # Perform IFFT
    X_real, X_imag = X_real.view(-1, 2), X_imag.view(-1, 2)
    for _ in range(int(math.log2(n_fft))):
        X_real, X_imag = X_real.view(-1, 2, 2), X_imag.view(-1, 2, 2)
        X_real, X_imag = X_real[..., 0] + twiddle_real * X_real[..., 1], X_imag[..., 0] + twiddle_imag * X_imag[..., 1]
        X_real, X_imag = X_real.view(-1, 2), X_imag.view(-1, 2)

    # Normalize and reshape the output
    X_real /= n_fft
    X_imag /= n_fft
    X_real = X_real.view(X_real.size()[:-1] + (n_orig,))
    X_imag = X_imag.view(X_imag.size()[:-1] + (n_orig,))

    return X_real, X_imag

# import torch
# import math

# def ifft(X):
#     # Get the input size
#     n_orig = X.size(-2)
#     n_fft = X.size(-2)

#     # Create butterfly twiddle factors
#     k = torch.arange(n_fft // 2).view(1, -1)
#     theta = 2 * math.pi * k / n_fft
#     twiddle_real = torch.cos(theta)
#     twiddle_imag = torch.sin(theta)

#     # Perform IFFT
#     X = X.view(*X.size()[:-1], 2)
#     for _ in range(int(math.log2(n_fft))):
#         X = X.view(*X.size()[:-2], -1, 2, 2)
#         X = torch.stack([X[..., 0] + twiddle_real * X[..., 1],
#                         X[..., 0] + twiddle_imag * X[..., 1]], dim=-1)
#         X = X.view(*X.size()[:-2], -1, 2)

#     # Normalize and reshape the output
#     X /= n_fft
#     X = X.view(X.size()[:-1] + (n_orig,))

#     return X
