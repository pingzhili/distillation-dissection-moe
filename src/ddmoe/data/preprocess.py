from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizerBase, AutoTokenizer
from functools import partial
import torch

__all__ = ["batch_preprocess_fn"]


def batch_preprocess_fn(
        examples: Dict[str, List[Any]], task: str, tokenizer: PreTrainedTokenizerBase = None
) -> Dict[str, List[Any]]:
    task_to_fn = {
        "chat-eval": partial(chat_eval_batch_preprocess_fn, tokenizer=tokenizer),
        "sft-olmoe-train": partial(sft_olmoe_train_batch_preprocess_fn, tokenizer=tokenizer),
    }
    return task_to_fn[task](examples)


def chat_eval_batch_preprocess_fn(
        examples: Dict[str, List[Any]], tokenizer: Optional[PreTrainedTokenizerBase] = None
) -> Dict[str, List[Any]]:
    """
    Parameters
    ----------
    examples: Dict[str, List[Any]]
        examples to preprocess
    tokenizer: PreTrainedTokenizerBase, optional
        tokenizer to use

    Returns
    -------
    Dict[str, List[Any]]
        preprocessed examples

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("moonshotai/Moonlight-16B-A3B-Instruct", trust_remote_code=True)
    >>> raw_examples = {"messages": [[{"content": "Hello, how are you?", "role": "user"}, {"content": "I am good, how can I help you?", "role": "system"}]]}
    >>> preprocessed_examples = chat_eval_batch_preprocess_fn(raw_examples, tokenizer)
    >>> preprocessed_examples.keys()
    dict_keys(['input_ids'])
    """
    messages_list = [messeges[0]["content"] for messeges in examples["messages"]]
    chat_list = [
        [{"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
         {"role": "user", "content": messages}] for messages in messages_list
    ]
    if tokenizer is None:
        return {"content": chat_list}
    else:
        input_ids_list = tokenizer.apply_chat_template(chat_list, add_generation_prompt=True)
        return {"input_ids": input_ids_list, "content": messages_list}


def apply_general_chat_template(
        question: str,
        tokenizer: PreTrainedTokenizerBase,
        response: Optional[str] = None,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    if response is None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        messages.append({"role": "assistant", "content": response})
        return tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)


def sft_olmoe_train_batch_preprocess_fn(
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizerBase,
):
    if tokenizer is None:
        raise ValueError("Tokenizer is required for SFT training.")

    # 1. apply general chat template to each example

    all_chat_texts = []

    for question, response in zip(examples["question"], examples["response"]):
        chat_text = apply_general_chat_template(question, response=response, tokenizer=tokenizer)
        all_chat_texts.append(chat_text)

    # 2. Tokenize the chat
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for chat_text in all_chat_texts:
        encoded = tokenizer(chat_text, padding=False, truncation=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # 3. Only apply LM loss on the assistant's response & "<|im_end|>"
        labels = [-100] * len(input_ids)

        user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
        assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
        end_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        pos_assistant = -1
        pos_end_after_response = -1

        for i, token_id in enumerate(input_ids):
            if token_id == assistant_token_id:
                pos_assistant = i
            elif token_id == end_token_id and pos_assistant != -1:
                pos_end_after_response = i
                break

        if pos_assistant != -1 and pos_end_after_response != -1:
            for i in range(pos_assistant + 1, pos_end_after_response):
                labels[i] = input_ids[i]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels
    }