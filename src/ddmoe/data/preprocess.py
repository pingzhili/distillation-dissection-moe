from typing import Any, Dict, List

from transformers import PreTrainedTokenizerBase
from functools import partial

__all__ = ["batch_preprocess_fn"]


def batch_preprocess_fn(
        examples: Dict[str, List[Any]], task: str, tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List[Any]]:
    task_to_fn = {
        "chat-eval": partial(chat_eval_batch_preprocess_fn, tokenizer=tokenizer),
    }
    return task_to_fn[task](examples)


def chat_eval_batch_preprocess_fn(
        examples: Dict[str, List[Any]], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List[Any]]:
    """
    Parameters
    ----------
    examples: Dict[str, List[Any]]
        examples to preprocess
    tokenizer: PreTrainedTokenizerBase
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
    messages_list = [
        [{"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
         {"role": "user", "content": messages}] for messages in messages_list
    ]
    input_ids_list = tokenizer.apply_chat_template(
        messages_list, add_generation_prompt=True, return_tensors="pt"
    )
    return {"input_ids": input_ids_list}
