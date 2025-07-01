import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            #hidden_dims = [32, 64, 128, 256, 512]
            hidden_dims = [32, 64, 128, 256, 512, 1024]


        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        with torch.no_grad():
            dummy_tensor = torch.randn(1, 3, 320, 320)  # Match input size
            conv_output = self.encoder(dummy_tensor)
            print(conv_output.shape)  # Find correct size
            flattened_size = torch.flatten(conv_output, start_dim=1).shape[1]

        # Update FC layers to match computed flattened size
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_var = nn.Linear(flattened_size, latent_dim)
        ##########################################################################################################
                                            # MODIFIED CODE ENDS HERE #
        ##########################################################################################################

        # Update fully connected layers
        """
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        """        
        #self.fc_mu = nn.Linear(flattened_size, latent_dim)
        #self.fc_var = nn.Linear(flattened_size, latent_dim)
        
        """
        self.fc_mu = nn.Linear(51200, latent_dim)  # Use correct size
        self.fc_var = nn.Linear(51200, latent_dim)  # Use correct size

        """

        # Build Decoder
        modules = [] 

        #self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 5 * 5)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],  
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            
            # Extra Upsample to 320x320
            nn.Upsample(size=(320, 320), mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                    kernel_size=3, padding=1),
            nn.Tanh()
)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        print(f"Encoder Output Shape Before Flattening: {result.shape}")

        result = torch.flatten(result, start_dim=1)

        print(f"Flattened Encoder Output Shape: {result.shape}")

        ##########################################################################################################
                                            # MODIFIED CODE STARTS HERE #
        ##########################################################################################################
        ##########################################################################################################
                                            # MODIFIED CODE ENDS HERE #
        ##########################################################################################################

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        print(f"Decoder Input Shape: {z.shape}")  # Check before reshaping
        result = self.decoder_input(z)

        ##########################################################################################################
                                            # MODIFIED CODE STARTS HERE #
        ##########################################################################################################
        expected_shape = (-1, 1024, 5, 5)  # Expected shape
        print(f"Reshaping to: {expected_shape}")

        result = result.view(expected_shape)  # Reshape correctly
        print(f"After Reshape: {result.shape}")
         ##########################################################################################################
                                            # MODIFIED CODE ENDS HERE #
        ##########################################################################################################

        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:

        #raise ValueError("This is a test error!") #Error test
        
        print("Forward Pass Started!")
        print(f"Input Shape: {input.shape}")  # Check input shape (should be [B, C, 320, 320])

        mu, log_var = self.encode(input)

        print("Encoding Done!") 
        print(f"After Encoding: mu={mu.shape}, log_var={log_var.shape}")

        z = self.reparameterize(mu, log_var)
        
        print("Latent Space Sampled!")
        print(f"Latent Vector Shape: {z.shape}")  # Should be [16, 128]
        print(f"Decoder Input Shape Before Reshape: {self.decoder_input(z).shape}")  


        output = self.decode(z)
        print(f"Final Output Shape Before Loss: {output.shape}")

        # This line was added by me to check for output size was actually down sampled
        if output.shape[-1] != 320:
            output = F.interpolate(output, size=(320, 320), mode='bilinear', align_corners=False)
        ############################################################################################

        return [output, input, mu, log_var]
        #return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

###################################################
        if recons.shape != input.shape:
            recons = F.interpolate(recons, size=input.shape[2:], mode='bilinear', align_corners=False)
####################################################

        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]