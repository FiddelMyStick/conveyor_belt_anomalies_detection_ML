# --- models.resnet.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List # Optional: for type hinting if desired

# Define the ResNetVAE_V2 class (with increased decoder capacity)
class ResNetVAE_V2(nn.Module):
    def __init__(self, latent_dim: int = 512, input_height: int = 320):
        super().__init__()
        self.latent_dim = latent_dim
        print(f"Initializing ResNetVAE_V2 with latent_dim = {self.latent_dim}")

        # --- Encoder ---
        try:
            resnet_weights = models.ResNet34_Weights.IMAGENET1K_V1
        except AttributeError:
            # Fallback for older torchvision versions
            print("Using older torchvision ResNet weights loading method.")
            resnet_weights = True # Loads default pretrained weights
        resnet = models.resnet34(weights=resnet_weights)

        encoder_modules = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*encoder_modules)

        # --- Flattened Size Calculation ---
        self.encoder_output_size = input_height // (2**5)
        self.encoder_output_channels = 512 # ResNet34 layer4 output channels
        flattened_size = self.encoder_output_channels * self.encoder_output_size * self.encoder_output_size
        # print(f"Encoder output feature map size: {self.encoder_output_channels} x {self.encoder_output_size} x {self.encoder_output_size}")
        # print(f"Flattened encoder output size: {flattened_size}") # Keep prints minimal in library code

        # --- Latent Space Layers ---
        self.fc_mu = nn.Linear(flattened_size, self.latent_dim)
        self.fc_var = nn.Linear(flattened_size, self.latent_dim)

        # --- Build Decoder (Increased Capacity) ---
        self.decoder_input = nn.Linear(self.latent_dim, flattened_size)

        decoder_modules = []
        hidden_dims = [self.encoder_output_channels, 512, 256, 128, 64] # Example: 512 -> 512 -> 256 -> 128 -> 64
        current_channels = self.encoder_output_channels
        for h_dim in hidden_dims[1:]:
             decoder_modules.append(
                 nn.Sequential(
                     nn.ConvTranspose2d(current_channels, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                     nn.BatchNorm2d(h_dim),
                     nn.LeakyReLU()
                 )
             )
             current_channels = h_dim
        self.decoder_conv = nn.Sequential(*decoder_modules) # Output: [batch, 64, 160, 160]

        # Final layer
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
                            nn.Tanh())

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder_output_channels, self.encoder_output_size, self.encoder_output_size)
        result = self.decoder_conv(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return [reconstruction, input, mu, log_var]

    # Loss Function (L1+KLD+LPIPS compatible)
    def loss_function(self, *args, **kwargs) -> dict:
        recons, input, mu, log_var = args[0], args[1], args[2], args[3]
        kld_weight = kwargs.get('M_N', 0.0001)
        lpips_weight = kwargs.get('lpips_weight', 0.0)
        lpips_model = kwargs.get('lpips_model', None) # Expect lpips model passed in

        if recons.shape != input.shape: recons = F.interpolate(recons, size=input.shape[2:], mode='bilinear', align_corners=False)

        recon_loss_l1 = F.l1_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        perceptual_loss = torch.tensor(0.0).to(input.device)
        # Calculate LPIPS only if model and weight are provided and weight > 0
        if lpips_model is not None and lpips_weight > 0:
             # Ensure LPIPS model is on the same device and in eval mode
             lpips_model.to(input.device)
             lpips_model.eval()
             with torch.no_grad(): # Don't track gradients for LPIPS model itself if pre-frozen
                 perceptual_loss = lpips_model(recons.detach(), input.detach()).mean() # Use detach if LPIPS isn't part of backward pass goal
                 # OR if you want LPIPS gradients to flow back (more complex optimization):
                 # perceptual_loss = lpips_model(recons, input).mean()

        loss = recon_loss_l1 + (kld_weight * kld_loss) + (lpips_weight * perceptual_loss)

        return {
            'loss': loss,
            'Reconstruction_Loss_L1': recon_loss_l1.detach(),
            'KLD': -kld_loss.detach(),
            'Perceptual_Loss': perceptual_loss.detach()
        }

# You can add other model definitions here later if needed