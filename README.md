# Modern NLP with Hugging Face
 - This is a 4-week long course that helps you understand the rich hugging face ecosystem.   
## Week 1: Datasets
 - [Slides](https://iitm-pod.slides.com/arunprakash_ai/dlp-lecture-1)
 - [Notebook](https://github.com/Arunprakash-A/Modern-NLP-with-Hugging-Face/blob/main/Notebooks/Datasets.ipynb)
 - Loading datasets in different formats
 - Understand the Structure of the loaded dataset
 - Access and manipulate samples in the dataset
   - Concatenate
   - Interleave
   - Map
   - Filter
## Week 2: Tokenizers
  - [Slides](https://iitm-pod.slides.com/arunprakash_ai/dlp-lecture-2)
  - [Notebook](https://github.com/Arunprakash-A/Modern-NLP-with-Hugging-Face/blob/main/Notebooks/Tokenizer.ipynb)
  - Set up a tokenization pipeline
  - Train the tokenizer
  - Encode the input samples (single or batch)
  - Test the implementation
  - Save and load the tokenizer
  - Decode the token_ids
  - Wrap the tokenizer with the `PreTrainedTokenizer` class
  - Save the pre-trained tokenizer
## Week 3: Pre-training GPT-2 
 - [Slides](https://iitm-pod.slides.com/arunprakash_ai/dlp_nlp_w3?token=5976JyVI)
 - [Notebook](https://github.com/Arunprakash-A/Modern-NLP-with-Hugging-Face/blob/main/Notebooks/PreTraining_GPT.ipynb)
 - Download the model checkpoints: [Here](https://drive.google.com/drive/folders/1DQwmmq9hLtIwfXVMKlYtNu_k9ZS7Nekx?usp=drive_link)
 - Set up the training pipeline
    - Dataset: BookCorpus
    - Number of tokens: 1.08 billion
    - Tokenizer: gpt-2 tokenizer
    - Model: gpt-2 with CLM head
    - Optimizer: AdamW
    - Parallelism: DDP (with L4 GPUs)
 - Train the model on
    - A100 80 GB single GPU
    - L4 48 GB single node Multiple GPUs
    - V100 32 GB single GPU
 - Training Report at [wandb](https://wandb.ai/a-arun283-iit-madras/DLP-GPT2-Node-1/reports/DLP-PreTraining-GPT-2--Vmlldzo5NTgxMTUx)
 - Text Generation
## Week 4: Fine-tuning Llama 3.2 1B
 - [Slides](https://iitm-pod.slides.com/arunprakash_ai/dlp-nlp-w4/fullscreen?token=vmr9TM_Z)
 - [Notebook](https://github.com/Arunprakash-A/Modern-NLP-with-Hugging-Face/blob/main/Notebooks/ContPreTrain-Peft-quantize.ipynb)
 - Concepts:
    - PEFT and Quantization
    - Continued Pretraining of Llama 3.2 1B on 48 GB L4 GPU
    - Task-specific fine-tuning (classification) [Notebook](https://github.com/Arunprakash-A/Modern-NLP-with-Hugging-Face/blob/main/Notebooks/Task-Specific-FineTuning.ipynb)
    - Instruction tuning [Notebook from Unsloth](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)
    -  Preference tuning 
   
     
