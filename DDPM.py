import torch 
import torch.nn as nn
from torch.nn import functional as F 
import math
import numpy as np

class DenoisingDiffusionProbabilisticModelSampler:
    def __init__(self, generator: torch.Generator, number_training_steps = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120) -> None:
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, number_training_steps, dtype= torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumlativeproduct = torch.cumprod(self.alphas, d_model = 0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.number_training_timesteps = number_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, number_training_steps)[::-1].copy())
    
    def set_number_inference_steps(self, number_inference_steps = 50):
        self.number_inference_steps = number_inference_steps
        ratio = self.number_training_timesteps // self.number_inference_steps
        timesteps = (np.arange(0, self.number_inference_steps) * ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def _get_previous_step(self, timestep: int) -> int:
        previous_step = timestep - self.number_training_timesteps // self.number_inference_steps
        return previous_step
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        previous_step_t = self._get_previous_step(timestep)
        alpha_product_t = self.alphas_cumlativeproduct[timestep]
        alpha_product_t_previous = self.alphas_cumlativeproduct[previous_step_t] if previous_step_t >= 0 else self.one
        current_betas_t = 1 - alpha_product_t / alpha_product_t_previous
        variance = (1 - alpha_product_t_previous) /  (1- alpha_product_t) * current_betas_t
        variance = torch.clamp(variance, min=1e-20)
        return variance
    
    def set_strength(self, strength = 1):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        start_step = self.number_inference_steps - int(self.number_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
        


