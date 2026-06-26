from transformers import AutoTokenizer, AutoModelForCausalLM
from config_loader import load_config
config = load_config()

def load_model():
    try:
        model_id = config.get("MODEL_ID")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        print(f"Erro ao carregar o modelo '{model_id}'.")

    return model, tokenizer

def send_message(model,tokenizer,usr_message):
    messages = [
        {"role": "user", "content": usr_message},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


def main():
    model, tokenizer = load_model()
    send_message(model,tokenizer,"Hi, how can you help me?")

if __name__ == "__main__":
    main()