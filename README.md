# Privacy-Preserving-Multimodal-Medical-Question-Answer-LLM
 *A powerful framework designed to handle sensitive medical data with advanced privacy techniques.* 

This project extends LLaVA-Med with state-of-the-art privacy mechanisms for both text and images, ensuring the protection of sensitive information in real-time.

# Contents
- [Install](#Install)
- [Serving](#Serving)
- [Model Description](#ModelDescription)
- [Acknowledgement](#Acknowledgement)
- [Base Model](#Base_Model)

# Install
<a id="Install"></a>
1. Clone this repository and navigate to LLaVA-Med folder
     ```sh
     https://github.com/microsoft/LLaVA-Med.git
     cd LLaVA-Med
     ```
2. Install Package: Create conda environment
    ```sh
     conda create -n llava-med python=3.10 -y
     conda activate llava-med
     pip install --upgrade pip  # enable PEP 660 support
     pip install -e .
     ```
3. Install openai
   ```sh
     import openai
     ```
     If you are running a Python script and want to access the API key
      ```sh
     import os
     api_key = os.getenv("OPENAI_API_KEY")

     ```
# Serving
<a id="Serving"></a>
### Web UI

**Launch a Controller**
   ```sh
     python -m llava.serve.controller --host 0.0.0.0 --port 10000
   ```
**Launch a model worker**
   ```sh
     python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path microsoft/llava-med-v1.5-mistral-7b --multi-modal
   ```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".

**Launch a model worker (Multiple GPUs, when GPU VRAM <= 24GB)**
If your the VRAM of your GPU is less than 24GB (e.g., RTX 3090, RTX 4090, etc.), you may try running it with multiple GPUs.
   ```sh
     python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path microsoft/llava-med-v1.5-mistral-7b --multi-modal --num-gpus 2
   ```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".

**Send a test message**
   ```sh
     python -m llava.serve.test_message --model-name llava-med-v1.5-mistral-7b --controller http://localhost:10000
   ```
**Launch a gradio server**
   ```sh
     python -m llava.serve.gradio_web_server --controller http://localhost:10000
   ```
**You can open your browser and chat with a model now.**

# Model Description
<a id="ModelDescription"></a>

The Privacy-Preserving LLaVA-Med model enhances the original LLaVA-Med biomedical assistant by incorporating additional privacy protections for text and image data. Utilizing advanced natural language processing and computer vision techniques, this model preprocesses user data in real-time to obscure personally identifiable information and adds controlled noise to sensitive image regions. While it functions as a preprocessing module, it does not alter the core architecture of LLaVA-Med, ensuring that user privacy is maintained without compromising the accuracy of biomedical assistance.

**Limitations**

While the Privacy-Preserving LLaVA-Med model enhances privacy for both text and image data, it does have certain limitations. Although it is beneficial for protecting privacy, the added noise in images and the obfuscation of text can sometimes slightly impact the model's performance, particularly in situations where fine-grained details are essential for accurate medical interpretation. Furthermore, the use of the OpenAI API, which addresses text obfuscation, limits the number of simultaneous conversations and also slows down response times.

# Acknowledgement
<a id="Acknowledgement"></a>

# Base Model
<a id="Base_Model"></a>
[LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
