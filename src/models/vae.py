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


class VAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, latent_channels=32, encoder_channels=[32, 64, 128], decoder_channels=[128,64,32]):
        """
        Variational Autoencoder (VAE) module for unsupervised representation learning.

        This module consists of an encoder, a decoder, and linear layers for
        modeling the mean (mu) and log variance (logvar) of the latent space.

        Args:
            in_channels (int): Number of input channels.
            latent_channels (int): Number of channels in the latent space.
            encoder_channels (list): List of integers representing encoder hidden layer channels.
            decoder_channels (list): List of integers representing decoder hidden layer channels.
        """
        super(VAE, self).__init__()
        latent_dim = latent_channels

        # assert that output will have the same size
        assert len(encoder_channels) == len(decoder_channels)

        self.encoder = Encoder(in_channels, out_channels, encoder_channels)
        self.decoder = Decoder(out_channels, in_channels, decoder_channels)

        self.fc_mu = nn.Linear(out_channels, latent_dim)
        self.fc_logvar = nn.Linear(out_channels, latent_dim)
        self.latent_dim = latent_dim


    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        # sample epsilon from standard normal
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        # Encoder
        hidden = self.encoder(x)
        b, c, _, _ = hidden.size()
        hidden = hidden.view(b, -1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)

        # Decoder
        reconstruction = self.decoder(z.view(b, c, 1, -1))
        return reconstruction, mu, logvar

    def encode(self, x):
        # Encoder
        hidden = self.encoder(x)
        b, c, _, _ = hidden.size()
        hidden = hidden.view(b, -1)
        return self.fc_mu(hidden)

    def decode(self, z):
        self.decoder(z.view(z.size(0), -1, 1, 1))
