Multi-Domain LoRA Router

Training and loading a separate language model for every subject can quickly become expensive. I built this project to explore a more practical approach: use one shared base model, give it a small LoRA adapter for each domain, and automatically choose the right adapter based on the user's prompt.

The project currently supports three domains:

Code generation

Medical question answering

General instruction following

Everything is included in the multi_domain_lora_router_colab.ipynb notebook and is designed to run in Google Colab.


Instead of keeping three complete models in memory, the system loads GPT-2once. Each domain is represented by a much smaller LoRA adapter. A TF-IDF and Logistic Regression classifier reads the prompt, predicts its domain, and activates the matching adapter before generation.


Datasets

I use 1,000 valid examples from each dataset by default so the full pipeline issmall enough for a Colab experiment.

Domain

Dataset

Code

CodeAlpaca-20k

Medical

MedQuAD

General

Alpaca

The datasets have different column names, so the notebook converts them into ashared instruction-response format before tokenization.

### Instruction:
{prompt}

### Response:
{answer}

Running the notebook

Open the notebook in Google Colab.

Select Runtime → Change runtime type → GPU.

Start with a clean runtime and run the dependency cell.

Restart the runtime if Colab says an already-imported package was changed.

Run the remaining cells in order.

The default setup uses one epoch and 1,000 samples per domain. For a quick test,change SAMPLES_PER_DOMAIN to 100 before running the dataset cell.

Main settings:

MODEL_ID = "gpt2"
SAMPLES_PER_DOMAIN = 1_000
MAX_LENGTH = 384

Using the trained system

After the adapters and router have been trained, the system can be used throughone class:

engine = DynamicLoRARouter("/content/multidomain_lora")

result = engine.generate(
    "Write a Python function that removes duplicates from a list.",
    max_new_tokens=128,
)

print("Selected domain:", result["domain"])
print("Confidence:", result["confidence"])
print("Response:", result["response"])

The returned result includes the chosen domain, routing confidence, probabilityfor each domain, and generated response.

What I learned

This project helped me understand how several parts of an applied machine learning system fit together. The LoRA training itself is only one part of the work. Preparing datasets with different schemas, controlling GPU memory, saving adapters correctly, routing prompts, and building a usable inference class were equally important.

It also showed me why system evaluation needs more than one accuracy score. The router can select the correct domain while the language model still produces a weak answer, so routing quality and response quality need to be evaluated separately.

Tools used

Python and PyTorch

Hugging Face Transformers and Datasets

PEFT and LoRA

Scikit-learn

Google Colab
