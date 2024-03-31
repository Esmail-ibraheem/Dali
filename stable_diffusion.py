import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import numpy as np 
import math 

class GaussingDistribution:
    def __init__(self, parameters: torch.Tensor) -> None:
        self.mean, log_variance = torch.chunk(parameters, 2, dim = 1)
        self.log_variance = torch.clamp(log_variance, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.log_variance)
    
    def sample(self):
        return self.mean + self.std * torch.rand_like(self.std)

class DDPMSampler:
    def __init__(self, generator: torch.Generator, number_training_steps = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120) -> None:
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, number_training_steps, dtype=torch.float32) ** 2 
        self.alphas = 1.0 - self.betas
        self.alphas_cumlative_product = torch.cumprod(self.alphas, d_model = 0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.number_training_timesteps = number_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, number_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, number_inference_steps = 50):
        self.number_inference_steps = number_inference_steps
        ratio = self.number_training_timesteps // self.number_inference_steps
        timesteps = (np.arange(0, number_inference_steps) * ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def _get_previous_timestep(self, timestep: int) -> int:
        previous_step = timestep - self.number_training_timesteps // self.number_inference_steps
        return previous_step
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        previous_step = self._get_previous_timestep(timestep)
        alphas_product_t = self.alphas_cumlative_product[timestep]
        alphas_product_t_previous = self.alphas_cumlative_product[previous_step] if previous_step >= 0 else self.one
        current_beta_t = 1 - alphas_product_t / alphas_product_t_previous
        variance = (1 - alphas_product_t_previous) / (1 - alphas_product_t) * current_beta_t
        variance = torch.clamp(variance, 1e-20)
        return variance 
    
    def set_strength(self, strength = 1):
        """
        Set how much noise to add to the input image. 
        More noise (strength ~ 1) means that the output will be further from the input image.
        Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        startstep = self.number_inference_steps - int(self.number_inference_steps * strength)
        self.timesteps = self.timesteps[startstep:]
        self.startstep = startstep
    
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        previous_step = self._get_previous_timestep(t)
        alphas_product_t = self.alphas_cumlative_product[t]
        alphas_product_t_previous = self.alphas_cumlative_product[previous_step] if previous_step >=0 else self.one
        current_alphas_product_t = alphas_product_t / alphas_product_t_previous
        beta_t = 1 - alphas_product_t
        beta_t_previous = 1 - alphas_product_t_previous
        current_beta_t = 1 - current_alphas_product_t
        predicate_original_samples = (latents - beta_t ** (0.5) * model_output) / alphas_product_t
        predicate_original_samples_coeff = (alphas_product_t_previous ** (0.5) * current_beta_t) / beta_t
        predicate_current_samples_coeff = current_alphas_product_t ** (0.5) * beta_t_previous / beta_t
        predicate_previous_samples = predicate_original_samples_coeff * predicate_original_samples + predicate_current_samples_coeff * latents
        variance = 0
        if t > 0:
            device = model_output.device 
            noise = torch.randn(model_output.shape, generator=self.generator, device = device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise 
        predicate_previous_samples = predicate_previous_samples + variance
        return predicate_previous_samples

    def add_noise(self, original_samples: torch.FloatTensor, timestep: torch.IntTensor):
        alphas_cumlative_product_t = self.alphas_cumlative_product.to(device = original_samples.device, dtype = original_samples.dtype) 
        timestep = timestep.to(original_samples.device)
        alphas_cumlative_product_t_squaroot = alphas_cumlative_product_t[timestep] ** 0.5 
        alphas_cumlative_product_t_squaroot = alphas_cumlative_product_t_squaroot.flatten()
        while len(alphas_cumlative_product_t_squaroot.shape) < len(original_samples.shape):
            alphas_cumlative_product_t_squaroot = alphas_cumlative_product_t_squaroot.unsqueeze(-1)

        alphas_cumlative_product_t_squaroot_mins_one = (1 - alphas_cumlative_product_t[timestep]) ** 0.5 
        alphas_cumlative_product_t_squaroot_mins_one = alphas_cumlative_product_t_squaroot_mins_one.flatten()
        while len(alphas_cumlative_product_t_squaroot_mins_one.shape) < len(original_samples.shape):
            alphas_cumlative_product_t_squaroot_mins_one = alphas_cumlative_product_t_squaroot_mins_one.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator=self.generator, device = original_samples.device, dtype = original_samples.dtype)
        noisy_samples = alphas_cumlative_product_t_squaroot * original_samples + alphas_cumlative_product_t_squaroot_mins_one * noise 
        return noisy_samples