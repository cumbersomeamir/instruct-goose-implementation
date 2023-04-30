from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader, random_split
from torch import optim

from instruct_goose import Agent, RewardModel, RLHFTrainer, RLHFConfig, create_reference_model


dataset = load_dataset("imdb", split="train")
dataset, _ = random_split(dataset, lengths=[10, len(dataset) - 10]) # for demenstration purposes
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


model_base = AutoModelForCausalLM.from_pretrained("gpt2") # for demonstration purposes
reward_model = RewardModel("gpt2")

tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
eos_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model = Agent(model_base)
ref_model = create_reference_model(model)


max_new_tokens = 20
generation_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": max_new_tokens
}

config = RLHFConfig()
N_EPOCH = 1 # for demonstration purposes
trainer = RLHFTrainer(model, ref_model, config)
optimizer = optim.SGD(model.parameters(), lr=1e-3)


for epoch in range(N_EPOCH):
    for batch in train_dataloader:
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
        response_ids = model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"],
            **generation_kwargs
        )
        
        # extract the generated text
        response_ids = response_ids[:, -max_new_tokens:]
        response_attention_mask = torch.ones_like(response_ids)
        
        # evaluate from the reward model
        with torch.no_grad():
            text_input_ids = torch.stack([torch.concat([q, r]) for q, r in zip(inputs["input_ids"], response_ids)], dim=0)
            rewards = reward_model(text_input_ids)
        
        # calculate PPO loss
        loss = trainer.compute_loss(
            query_ids=inputs["input_ids"],
            query_attention_mask=inputs["attention_mask"],
            response_ids=response_ids,
            response_attention_mask=response_attention_mask,
            rewards=rewards
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss={loss}")
