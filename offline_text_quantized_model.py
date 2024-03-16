from ctransformers import AutoModelForCausalLM

model_path = 'Models/models--TheBloke--Mistral-7B-v0.1-GGUF/snapshots/d4ae605152c8de0d6570cf624c083fa57dd0d551'
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(model_path, model_file="mistral-7b-v0.1.Q4_K_M.gguf",
                                           model_type="mistral", gpu_layers=50)

prompt = 'Which is the capital of Vietnam?'
print(llm(f'Question: {prompt} Answer:'))
