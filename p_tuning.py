from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import Parameter
from datasets import load_dataset
from torch.optim import AdamW


model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt_embeddings = Parameter(torch.randn(10, model.config.hidden_size))


def add_prompt_to_input(input_ids, prompt_embeddings):
    prompt_tokens = prompt_embeddings.unsqueeze(0).repeat(input_ids.size(0), 1, 1)
    return torch.cat((prompt_tokens, input_ids), dim=1)


for param in model.parameters():
    param.requires_grad = False

optimizer = AdamW([prompt_embeddings], lr=5e-4)

dataset = load_dataset(
    "json",
    data_files={
        "train": "/home/jovyan/mistral_codex_hack/data/train_data.jsonl",
        "validate": "/home/jovyan/mistral_codex_hack/data/eval_data.jsonl"
    }
)
train_data = dataset['train']
test_data = dataset['validate']
num_epochs = 3


for epoch in range(num_epochs):
    for batch in train_data:
        inputs = tokenizer(
            batch['text'], return_tensors="pt", padding=True, truncation=True,
        )
        input_ids = inputs['input_ids']

        modified_inputs = add_prompt_to_input(input_ids, prompt_embeddings)

        outputs = model(input_ids=modified_inputs, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
