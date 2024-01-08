import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU,
    'elu': nn.ELU,
    'none': nn.Identity,
}


class ObsEncoder(nn.Module):
    def __init__(self, input_shape, embedding_size, depth, kernels, activation):
        """
        :param input_shape: tuple containing shape of input
        :param embedding_size: Supposed length of encoded vector
        """
        super(ObsEncoder, self).__init__()
        self.shape = input_shape
        activation = ACTIVATIONS[activation]
        d = depth

        self.convolutions = []
        for i, k in enumerate(kernels):
            if i == 0:
                self.convolutions += [
                    nn.Conv2d(input_shape[0], d, k, stride=2),
                    activation(),
                ]
            else:
                self.convolutions += [
                    nn.Conv2d(d, 2*d, k, stride=2),
                    activation(),
                ]
                d *= 2
        self.convolutions = nn.Sequential(*self.convolutions)
        
        conv_shape = conv_out_shape(self.shape[1:], 0, kernels[0], 2)
        for k in kernels[1:]:
            conv_shape = conv_out_shape(conv_shape, 0, k, 2)
        self.embed_size = int(d*np.prod(conv_shape).item())

        if embedding_size == self.embed_size:
            self.fc_1 = nn.Identity()
        else:
            self.fc_1 = nn.Linear(self.embed_size, embedding_size)

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        embed = self.fc_1(embed)
        return embed


class ObsDecoder(nn.Module):
    def __init__(self, output_shape, embed_size, depth, kernels, activation):
        """
        :param output_shape: tuple containing shape of output obs
        :param embed_size: the size of input vector, for dreamerv2 : modelstate 
        """
        super(ObsDecoder, self).__init__()
        c, h, w = output_shape
        activation = ACTIVATIONS[activation]

        self.output_shape = output_shape
        self.conv_shape = (32*depth, 1, 1)
        self.linear = nn.Linear(embed_size, 32*depth)

        self.decoder = []
        for i, k in enumerate(kernels):
            d = int(2 ** (len(kernels) - i - 2) * depth)
            if i == len(kernels) - 1:
                self.decoder += [nn.ConvTranspose2d(d*2, c, k, 2)]
            elif i == 0:
                self.decoder += [
                    nn.ConvTranspose2d(32*depth, d, k, 2),
                    activation(),
                ]
            else:
                self.decoder += [
                    nn.ConvTranspose2d(d*2, d, k, 2),
                    activation(),
                ]
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)
        mean = torch.reshape(x, (*batch_shape, *self.output_shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.output_shape))
        return obs_dist
    
def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))



if __name__ == '__main__':
    encoder = ObsEncoder((3, 64, 64), 1024, 48, [4, 4, 4, 4], 'elu')
    obs = torch.randn(4, 3, 64, 64)
    embed = encoder(obs)
    print(embed.shape)
    decoder = ObsDecoder((3, 64, 64), 1024, 48, [5, 5, 6, 6], 'elu')
    obs = decoder(embed)
    print(obs)