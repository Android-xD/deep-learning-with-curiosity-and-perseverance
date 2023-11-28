import torch.nn as nn
import torch

# Define the encoder architecture
class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=12, hidden_channels=[16, 32, 32]):
        """
        Initialize the Encoder module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_channels (list): List of integers representing hidden layer channels.
        """
        super(Encoder, self).__init__()
        self.channels = [in_channels] + hidden_channels + [out_channels]
        layers = []
        for i in range(len(self.channels) - 1):
            layers.extend(
                [
                    nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                ]
            )
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


# Define the decoder architecture
class Decoder(nn.Module):
    def __init__(self, in_channels=12, out_channels=3, hidden_channels=[16, 32, 32]):
        """
        Initialize the Encoder module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_channels (list): List of integers representing hidden layer channels.
        """
        super(Decoder, self).__init__()
        self.channels = [in_channels] + hidden_channels + [out_channels]
        layers = []
        for i in range(len(self.channels) - 1):
            layers.extend(
                [
                    nn.ConvTranspose2d(self.channels[i], self.channels[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.ReLU()
                ]
            )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class AE(nn.Module):
    def __init__(self,in_channels=3, out_channels=12, encoder_channels=[16, 32, 32], decoder_channels=[32, 32, 16]):
        super(AE, self).__init__()
        self.encoder = Encoder(in_channels, out_channels, encoder_channels)
        self.decoder = Decoder(out_channels, in_channels, decoder_channels)


    def forward(self, x):
        # Encoder
        hidden = self.encoder(x)

        # Decoder
        reconstruction = self.decoder(hidden)
        return reconstruction


class Autoencoder(nn.Module):
    """ Adopted from
        Deep Clustering for Mars Rover image datasets
        Vikas Ramachandra
        Adjunct Professor, Data Science
        University of California, Berkeley
        Berkeley, CA
        virama@berkeley.edu

        Works for input size 32x32 """
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), return_indices=True)
        self.unconv1 = nn.ConvTranspose2d(6, 3, kernel_size=(5, 5))
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=(2, 2))
        self.unmaxunpool2 = nn.MaxUnpool2d(kernel_size=(2, 2))

        self.encoder1 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(6, 12, kernel_size=(5, 5)),
        )

        self.encoder2 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(12, 16, kernel_size=(5, 5)),
            nn.Tanh()
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 12, kernel_size=(5, 5)),
            nn.Tanh()
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(12, 6, kernel_size=(5, 5)),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.conv1(x)
        x, indices1 = self.maxpool1(x)
        x = self.encoder1(x)
        x, indices2 = self.maxpool2(x)
        x = self.encoder2(x)
        return x, (indices1, indices2)

    def decode(self, x, indices1, indices2):
        x = self.decoder2(x)
        x = self.unmaxunpool2(x, indices2)
        x = self.decoder1(x)
        x = self.maxunpool1(x, indices1)
        x = self.unconv1(x)
        x = nn.ReLU()(x)
        return x

    def forward(self, x):
        x, info = self.encode(x)
        x = self.decode(x, *info)

        return x



class Autoencoder_small(nn.Module):
    """ Adapted from:
        Deep Clustering for Mars Rover image datasets
        Vikas Ramachandra
        Adjunct Professor, Data Science
        University of California, Berkeley
        Berkeley, CA
        virama@berkeley.edu

        Works for input size 18x18"""
    def __init__(self):
        super(Autoencoder_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), return_indices=True)
        self.unconv1 = nn.ConvTranspose2d(6, 3, kernel_size=(3, 3))
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=(2, 2))
        self.unmaxunpool2 = nn.MaxUnpool2d(kernel_size=(2, 2))

        self.encoder1 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(6, 12, kernel_size=(3, 3)),
        )

        self.encoder2 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(12, 16, kernel_size=(3, 3)),
            nn.Tanh()
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 12, kernel_size=(3, 3)),
            nn.Tanh()
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(12, 6, kernel_size=(3, 3)),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.conv1(x)
        x, indices1 = self.maxpool1(x)
        x = self.encoder1(x)
        x, indices2 = self.maxpool2(x)
        x = self.encoder2(x)
        return x, (indices1, indices2)

    def decode(self, x, indices1, indices2):
        x = self.decoder2(x)
        x = self.unmaxunpool2(x, indices2)
        x = self.decoder1(x)
        x = self.maxunpool1(x, indices1)
        x = self.unconv1(x)
        x = nn.ReLU()(x)
        return x

    def forward(self, x):
        x, info = self.encode(x)
        x = self.decode(x, *info)

        return x


class Autoencoder_large(nn.Module):
    """ Adapted from
        Deep Clustering for Mars Rover image datasets
        Vikas Ramachandra
        Adjunct Professor, Data Science
        University of California, Berkeley
        Berkeley, CA
        virama@berkeley.edu

        Works for input size 32x32 """
    def __init__(self):
        super(Autoencoder_large, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=(5, 5))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), return_indices=True)
        self.unconv1 = nn.ConvTranspose2d(12, 3, kernel_size=(5, 5))
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=(2, 2))
        self.unmaxunpool2 = nn.MaxUnpool2d(kernel_size=(2, 2))

        self.encoder1 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(12, 24, kernel_size=(5, 5)),
        )

        self.encoder2 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(24, 32, kernel_size=(5, 5)),
            nn.Tanh()
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=(5, 5)),
            nn.Tanh()
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=(5, 5)),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.conv1(x)
        x, indices1 = self.maxpool1(x)
        x = self.encoder1(x)
        x, indices2 = self.maxpool2(x)
        x = self.encoder2(x)
        return x, (indices1, indices2)

    def decode(self, x, indices1, indices2):
        x = self.decoder2(x)
        x = self.unmaxunpool2(x, indices2)
        x = self.decoder1(x)
        x = self.maxunpool1(x, indices1)
        x = self.unconv1(x)
        x = nn.ReLU()(x)
        return x

    def forward(self, x):
        x, info = self.encode(x)
        x = self.decode(x, *info)

        return x

