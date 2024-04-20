from transformers import AutoTokenizer
from datasets import load_dataset
from transformers.utils import logging as t_logging
import logging

t_logging.set_verbosity_info()
logger = t_logging.get_logger("transformers")
formatter = logging.Formatter('[%(levelname)s] %(message)s') 
file_handler = logging.FileHandler(filename='training_new_Tokenizer.log', mode='w')
file_handler.setFormatter(formatter)    
logger.addHandler(file_handler)

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

### Load the dataset ###
logger.info("Load dataset: code_search_net-python")
raw_datasets = load_dataset("code_search_net", "python")

### generator of batch of 1000 texts ###
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"] for i in range(0, len(raw_datasets["train"]), 1000)
    )

### Load GPT-2 Tokenizer ###
logger.info("Load gpt2 Tokenizer")
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
logger.info("Training the tokenizer")
tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), 52000)

### Tokenize example ###
old_tokens = old_tokenizer.tokenize(example)
tokens = tokenizer.tokenize(example)
logger.info("tokenize example")
logger.info(f"raw sentence: {example}")
logger.info(f"old tokenizer result: {old_tokens}")
logger.info(f"new tokenizer result: {tokens}")

### Save the new tokenizer ###
checkpoint = "code-search-net-tokenizer"
tokenizer.save_pretrained(checkpoint)