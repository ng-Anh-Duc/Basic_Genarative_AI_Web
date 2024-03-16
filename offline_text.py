from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device('mps')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

prompt = 'Which is the capital of Vietnam?'

message = [{'role': 'user', 'content': prompt}]
encoded_input = tokenizer.apply_chat_template(message, return_tensors='pt')
input_ids = encoded_input.to(device)

output = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0], skip_special_tokens=True))