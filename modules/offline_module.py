from transformers import pipeline
import torch
from ctransformers import AutoModelForCausalLM
from diffusers import DiffusionPipeline
import streamlit as st
import os
from diffusers import DiffusionPipeline
import torch

device = torch.device('mps')

############################
#######   AUDIO   #########
##########################
# @st.cache_resource()
# def load_audio_model(model_path = '../Models/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e'):
#     model = pipeline("automatic-speech-recognition", model=model_path, chunk_length_s=30)
#     model.enable_model_cpu_offload()
#     return model

# def save_uploaded_audio_file(audio_file):
#     try:
#         os.makedirs('tempDir', exist_ok=True)
#         # file_name = os.path.basename(audio_file)
#         file_path = os.path.join('tempDir', audio_file.name)
#         print(file_path)
#         with open(file_path, 'wb') as f:
#             f.write(audio_file.getbuffer())
#         # audio_file.save(file_path)
#         return file_path

#     except Exception as e:
#         print(f"Error saving audio file: {e}")
#         return None

# def generate_text_from_audio(model, audio_file):
#     uploaded_file = save_uploaded_audio_file(audio_file=audio_file)

#     text_out = model(uploaded_file)['text']

#     return text_out

####### TEXT ##########
# def generate_text_from_text(prompt='Which is the capital of Vietnam?', gpu_layers=50):
#     model_path = 'Models/models--TheBloke--Mistral-7B-v0.1-GGUF/snapshots/d4ae605152c8de0d6570cf624c083fa57dd0d551'
#     llm = AutoModelForCausalLM.from_pretrained(model_path, model_file="mistral-7b-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=gpu_layers).to(device)

#     return llm(f'Question: {prompt} Answer:')

############################
#######   IMAGE   #########
##########################
@st.cache_resource()
def load_image_model(model_path = '../Models/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b',
                     torch_dtype=torch.float16, variant='fp16', use_safetensors=True,
                     refiner_model='../Models/models--stabilityai--stable-diffusion-xl-refiner-1.0/snapshots/5d4cfe854c9a9a87939ff3653551c2b3c99a4356'):
    model = DiffusionPipeline.from_pretrained(model_path,
                                                torch_dtype=torch.float16,
                                                variant=variant,
                                                use_safetensors=use_safetensors)

    if refiner_model:
        refiner = DiffusionPipeline.from_pretrained(
            refiner_model,
            text_encoder_2=model.text_encoder_2,
            vae=model.vae,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
            variant=variant
        )
        return model.to(device), refiner.to(device)

    return model.to(device)

def generate_image_from_text(model, prompt, num_inference_steps, guidance_scale, refiner=None,
                             noise_frac=0.6, output_type='latent', verbose='False', temperature=0.7):
    if refiner:
        image = model(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                      denoising_end=noise_frac, output_type=output_type, verbose=verbose, temperature=temperature).images
        image = refiner(prompt=prompt, num_inference_steps=num_inference_steps,
                      denoising_start=noise_frac, image=image, verbose=verbose).images[0]
    else:
        image = model(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

def main():
    # ###### AUDIO ########
    # st.title('Audio transcription')
    # model = load_audio_model().to(device)
    # audio_file = st.file_uploader("Choose an audio file...", type=['mp3', 'wav'])

    # if audio_file:
    #     if st.button('Transcribe'):
    #         st.audio(audio_file, format='audio/wav')
    #         with st.spinner('audio transcriping...'):
    #             result = generate_text_from_audio(model, audio_file)
    #             st.write(result)

    ####### IMAGE ########
    st. title('IMAGE GENERATION')
    model, refiner = load_image_model()
    prompt = st.text_input('Enter your prompt:', value='A cute cat')
    denoising_steps = st.number_input("Denoising steps:", step=1, value=50, format='%d')
    guidance_scale = st.number_input('Guidance scale:', step=0.1, value=7.5)
    if st.button('Generate'):
        with st.spinner('Generating image...'):
            image = generate_image_from_text(model=model, prompt=prompt,
                                             num_inference_steps=denoising_steps, guidance_scale=guidance_scale,
                                             refiner=refiner)
            st.image(image)

if __name__ == '__main__':
    main()