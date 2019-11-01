import logging

import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

_logger = logging.getLogger(__name__)


class DPWGAN(object):
    """Class to store, train, and generate from a
    differentially-private Wasserstein GAN

    Parameters
    ----------
    generator : torch.nn.Module
        torch Module mapping from random input to synthetic data

    discriminator : torch.nn.Module
        torch Module mapping from data to a real value

    noise_function : function
        Mapping from number of samples to a tensor with n samples of random
        data for input to the generator. The dimensions of the output noise
        must match the input dimensions of the generator.
    """
    def __init__(self, generator, discriminator, noise_function):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_function = noise_function

    def train(self, data, epochs=100, n_critics=5, batch_size=128,
              learning_rate=1e-4, sigma=None, weight_clip=0.1):
        """Train the model

        Parameters
        ----------
        data : torch.Tensor
            Data for training
        epochs : int
            Number of iterations over the full data set for training
        n_critics : int
            Number of discriminator training iterations
        batch_size : int
            Number of training examples per inner iteration
        learning_rate : float
            Learning rate for training
        sigma : float or None
            Amount of noise to add (for differential privacy)
        weight_clip : float
            Maximum range of weights (for differential privacy)
        """
        generator_solver = optim.RMSprop(
            self.generator.parameters(), lr=learning_rate
        )
        discriminator_solver = optim.RMSprop(
            self.discriminator.parameters(), lr=learning_rate
        )

        # add hooks to introduce noise to gradient for differential privacy
        if sigma is not None:
            for parameter in self.discriminator.parameters():
                parameter.register_hook(
                    lambda grad: grad + (1 / batch_size) * sigma
                    * torch.randn(parameter.shape)
                )

        # There is a batch for each critic (discriminator training iteration),
        # so each epoch is epoch_length iterations, and the total number of
        # iterations is the number of epochs times the length of each epoch.
        epoch_length = len(data) / (n_critics * batch_size)
        n_iters = int(epochs * epoch_length)
        for iteration in range(n_iters):
            for _ in range(n_critics):
                # Sample real data
                rand_perm = torch.randperm(data.size(0))
                samples = data[rand_perm[:batch_size]]
                real_sample = Variable(samples)

                # Sample fake data
                fake_sample = self.generate(batch_size)

                # Score data
                discriminator_real = self.discriminator(real_sample)
                discriminator_fake = self.discriminator(fake_sample)

                # Calculate discriminator loss
                # Discriminator wants to assign a high score to real data
                # and a low score to fake data
                discriminator_loss = -(
                    torch.mean(discriminator_real) -
                    torch.mean(discriminator_fake)
                )

                discriminator_loss.backward()
                discriminator_solver.step()

                # Weight clipping for privacy guarantee
                for param in self.discriminator.parameters():
                    param.data.clamp_(-weight_clip, weight_clip)

                # Reset gradient
                self.generator.zero_grad()
                self.discriminator.zero_grad()

            # Sample and score fake data
            fake_sample = self.generate(batch_size)
            discriminator_fake = self.discriminator(fake_sample)

            # Calculate generator loss
            # Generator wants discriminator to assign a high score to fake data
            generator_loss = -torch.mean(discriminator_fake)

            generator_loss.backward()
            generator_solver.step()

            # Reset gradient
            self.generator.zero_grad()
            self.discriminator.zero_grad()

            # Print training losses
            if int(iteration % epoch_length) == 0:
                epoch = int(iteration / epoch_length)
                _logger.info('Epoch {}\n'
                             'Discriminator loss: {}; '
                             'Generator loss: {}'
                             .format(epoch,
                                     discriminator_loss.data.numpy(),
                                     generator_loss.data.numpy()))

    def generate(self, n):
        """Generate a synthetic data set using the trained model

        Parameters
        ----------
        n : int
            Number of data points to generate

        Returns
        -------
        torch.Tensor
        """
        noise = self.noise_function(n)
        fake_sample = self.generator(noise)
        return fake_sample
