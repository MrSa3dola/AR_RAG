import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()
torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
messages = [
    {"role": "system", "content": '''extract the furniture attributes only
     Query:I want to recommend to me an item suitable for room with light blue color let's say I want a black armchair for example
     Response: Black armchair
     '''},
    {
        "role": "user",
        "content": "recommend me a white sofa for my wide room",
    },
]
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
generation_args = {
    "max_new_tokens": 100,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
output = pipe(messages, **generation_args)
print(output[0]["generated_text"])
