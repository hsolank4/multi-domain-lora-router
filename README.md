# Multi-Domain LoRA Router

Training and loading a separate language model for every subject can quickly become expensive. I built this project to explore a more practical approach: use one shared base model, give it a small LoRA adapter for each domain, and automatically choose the right adapter based on the user's prompt.


## Architecture & Methodology

Instead of keeping three complete models in memory, the system loads GPT-2 once. Each domain is represented by a much smaller LoRA adapter. A TF-IDF and Logistic Regression classifier reads the prompt, predicts its domain, and activates the matching adapter before generation.

The project currently supports three domains:
* Code generation
* Medical question answering
* General instruction following

## Datasets

I use 1,000 valid examples from each dataset by default so the full pipeline is small enough for a Colab experiment.

| Domain | Dataset |
| :--- | :--- |
| **Code** | CodeAlpaca-20k |
| **Medical** | MedQuAD |
| **General** | Alpaca |

The datasets have different column names, so the notebook converts them into a shared instruction-response format before tokenization:

```text
### Instruction:
{prompt}

### Response:
{answer}
