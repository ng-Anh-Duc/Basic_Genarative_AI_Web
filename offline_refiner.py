from diffusers import DiffusionPipeline
import torch

device = torch.device('mps')

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                                             torch_dtype=torch.float16,
                                             variant='fp16',
                                             use_safetensors=True).to(device)