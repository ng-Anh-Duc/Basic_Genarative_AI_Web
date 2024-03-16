from diffusers import DiffusionPipeline
import torch

device = torch.device('mps')
model_path = 'Models/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b'
pipeline = DiffusionPipeline.from_pretrained(model_path,
                                             torch_dtype=torch.float16,
                                             variant='fp16',
                                             use_safetensors=True).to(device)

prompt = 'A guy wear traditional Chinese costume'
image = pipeline(prompt=prompt, num_inference_steps=20, guidance_scale=4).images[0]
image.show()