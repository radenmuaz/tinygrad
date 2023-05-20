from tinygrad.tensor import Tensor
from scipy.signal import check_COLA, get_window
import numpy as np
class STFT(th.nn.Module):
    def __init__(self, win_len=1024, win_hop=512, fft_len=1024,
                 win_type='hann',
                 win_sqrt=False, pad_center=False):
        super(STFT, self).__init__()
        assert fft_len >= win_len
        self.win_len = win_len
        self.win_hop = win_hop
        self.fft_len = fft_len
        self.win_type = win_type
        self.win_sqrt = win_sqrt
        self.pad_center = pad_center
        self.pad_amount = self.fft_len // 2
        self.strider = Tensor.eye(self.win_len)[:, None, :]

        self.fft_k, self.ifft_k = self.init_kernel()
    
    def init_kernel(self):
        fft_kernel = np.fft.fft(np.eye(self.fft_len), 1)
        fft_kernel = fft_kernel[:self.win_len]
        fft_kernel = np.concatenate(
            (fft_kernel[:, :, 0], fft_kernel[:, :, 1]), dim=1)
        ifft_kernel = np.linalg.pinverse(fft_kernel)[:, None, :]

        window = get_window(self.win_type, self.win_len)
        window = Tensor(window)
        left_pad = (self.fft_len - self.win_len)//2
        right_pad = left_pad + (self.fft_len - self.win_len) % 2
        window = window.pad(left_pad, right_pad)
        if self.win_sqrt:
            self.padded_window = window
            window = window.sqrt()
        else:
            self.padded_window = window**2

        fft_kernel = fft_kernel.T * window
        ifft_kernel = ifft_kernel * window
        
        return fft_kernel, ifft_kernel

    def transform(self, x):
        self.num_samples = x.size(-1) # (B,1,T)
        if self.pad_center:
            x = x.pad(self.pad_amount, self.pad_amount)
        x = x.conv(self.en_k, stride=self.win_hop)
        outputs = x.transpose(1, 2)
        outputs = x.linear(self.fft_k)
        outputs = outputs.transpose(1, 2)
        dim = self.fft_len//2+1
        real = outputs[:, :dim, :]
        imag = outputs[:, dim:, :]
        return real.stack([imag])

    def inverse(self, z):

        inputs = z[:0,:].stack([z[:,1,:]], dim=1)
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