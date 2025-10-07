from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading

# Prompt template
PROMPT_TEMPLATE = '''You are a helpful, factual assistant. Use ONLY the information in the CONTEXT to answer the question.
If the answer is not contained in the context, reply: "I don't know based on the provided documents."

CONTEXT:
{context}

USER QUESTION:
{question}

ASSISTANT:'''


def build_prompt(retrieved_chunks, question: str, max_context_chars: int = 3500):
    # concatenate retrieved chunks. Avoid exceeding token/char budget
    parts = []
    total = 0
    for c in retrieved_chunks:
        text = c.get("text", "")
        if total + len(text) > max_context_chars:
            break
        parts.append(f"(source: {c.get('source')}, score: {c.get('score'):.3f})\n{text}")
        total += len(text)
    context = "\n\n---\n\n".join(parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def stream_generate(model_name: str, prompt: str, max_new_tokens: int = 256):

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    tokenizer.pad_token = tokenizer.eos_token

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Tokenize full prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Enforce max sequence length
    max_input_tokens = 1024 - max_new_tokens
    if input_ids.shape[1] > max_input_tokens:
        input_ids = input_ids[:, -max_input_tokens:]
        attention_mask = attention_mask[:, -max_input_tokens:]

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text
