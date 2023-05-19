import torch
import math

def fft(x_real, x_imag):
    # Pad input to the nearest power of 2
    n_orig = x_real.size(-1)
    n_fft = 1 << (n_orig - 1).bit_length()
    x_real_padded = torch.nn.functional.pad(x_real, pad=(0, n_fft - n_orig))
    x_imag_padded = torch.nn.functional.pad(x_imag, pad=(0, n_fft - n_orig))
    X = torch.stack((x_real_padded, x_imag_padded), dim=-1)

    # Create butterfly twiddle factors
    k = torch.arange(n_fft // 2).view(1, -1)
    theta = -2 * math.pi * k / n_fft
    twiddle_real = torch.cos(theta)
    twiddle_imag = torch.sin(theta)
    twiddle = torch.stack((twiddle_real, twiddle_imag), dim=-1).view(1, -1, 1)

    # Perform FFT
    for _ in range(int(math.log2(n_fft))):
        X = X.view(-1, 2, 2)
        X_real = 
        X_imag = 
        X = torch.cat((X[..., 0] + twiddle[..., 0] * X[..., 1],
                       X[..., 1] + twiddle[..., 1] * X[..., 0]), dim=-1)
        X = X.view(-1, 2)

    # Reshape and crop the output
    X_real = X[..., 0].view(x_real_padded.size())
    X_imag = X[..., 1].view(x_imag_padded.size())
    X_real = X_real[..., :n_orig]
    X_imag = X_imag[..., :n_orig]

    return X_real, X_imag