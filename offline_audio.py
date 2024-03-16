from transformers import pipeline
import torch
device = torch.device('mps')

model_path = 'Models/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e'
pipe = pipeline("automatic-speech-recognition", model=model_path, chunk_length_s=30, device=device)

text_out = pipe('Resources/Stellar_Love.mp3')['text']
print(text_out)
