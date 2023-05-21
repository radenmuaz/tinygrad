from tinygrad.tensor import Tensor, dtypes
from scipy.signal import check_COLA, get_window
import numpy as np
class STFT:
    def __init__(self, win_len=256, win_hop=100, fft_len=256,
                 win_type='hann', pad_center=False, dtype=dtypes.float32):
        super(STFT, self).__init__()
        assert fft_len >= win_len
        self.win_len = win_len
        self.win_hop = win_hop
        self.fft_len = fft_len
        self.win_type = win_type
        self.pad_center = pad_center
        self.pad_amount = self.fft_len // 2
        self.dtype = dtype
        self.fft_kernel, self.ifft_kernel = self.init_kernel()
    
    def init_kernel(self):
        fft_kernel = np.fft.fft(np.eye(self.fft_len))
        fft_kernel = fft_kernel[:self.win_len]
        fft_kernel = np.concatenate([fft_kernel.real, fft_kernel.imag], axis=1).T[:, None, :]
        ifft_kernel = np.linalg.pinv(fft_kernel)[:, None, :]

        window = get_window(self.win_type, self.win_len)
        left_pad = (self.fft_len - self.win_len)//2
        right_pad = left_pad + (self.fft_len - self.win_len) % 2
        window = np.pad(window,(left_pad, right_pad))
        self.ifft_window = window**2

        fft_kernel = (fft_kernel * window).astype(self.dtype.np)
        ifft_kernel = (ifft_kernel * window).astype(self.dtype.np)

        fft_kernel = Tensor(fft_kernel)
        ifft_kernel =  Tensor(ifft_kernel)
        return fft_kernel, ifft_kernel

    def transform(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        self.num_samples = x.shape[-1]
        if self.pad_center:
            x = x.pad(self.pad_amount, self.pad_amount)
        y = x.conv2d(self.fft_kernel, stride=self.win_hop)
        y = Tensor.stack([y[:,:self.fft_len,:], y[:,self.fft_len:,:]], dim=-1)
        return y

    def inverse(self, y):
        y = z[:0,:].stack([y[:,1,:]], dim=1)
        outputs = F.conv_transpose1d(inputs, self.ifft_k, stride=self.win_hop)
        t = (self.padded_window[None, :, None]).repeat(1, 1, inputs.size(-1))
        t = t.to(inputs.device)
        coff = F.conv_transpose1d(t, self.ola_k, stride=self.win_hop)
        rm_start, rm_end = self.pad_amount, self.pad_amount+self.num_samples
        outputs = outputs[..., rm_start:rm_end]
        coff = coff[..., rm_start:rm_end]
        coffidx = th.where(coff > 1e-8)
        outputs[coffidx] = outputs[coffidx]/(coff[coffidx])
        return outputs.squeeze(dim=1)