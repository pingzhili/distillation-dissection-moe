from ddmoe.models import DeepseekV3ForCausalLM
from transformers import AutoTokenizer

model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
model = DeepseekV3ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
    {"role": "user", "content": "Is 123 a prime?"}
]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
attention_mask = input_ids != tokenizer.pad_token_id
generated_ids = model.generate(inputs=input_ids, attention_mask=attention_mask, max_new_tokens=500)
# response = tokenizer.batch_decode(generated_ids)[0]
