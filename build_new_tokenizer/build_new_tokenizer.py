from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from transformers.utils import logging as t_logging
import logging
from tokenizers import Regex

t_logging.set_verbosity_info()
logger = t_logging.get_logger("transformers")
formatter = logging.Formatter('[%(levelname)s] %(message)s') 
file_handler = logging.FileHandler(filename='build_Tokenizer.log', mode='w')
file_handler.setFormatter(formatter)    
logger.addHandler(file_handler)

sentence1 = "Let's test this tokenizer..."
sentence2 = "on a pair of sentences."

### Load the dataset ###
logger.info("load_dataset: wikitext-2-raw-v1")
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

### generator of batch of 1000 texts ###
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

### create WordPiece Tokenizer ###
logger.info("Create WordPiece Tokenizer")
wordpiece_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

### Compose Normalizer ### 
wordpiece_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)

### Compose PreTokenizer ###
wordpiece_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

### Trainer ###
#All special tokens to use
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"] 
wordpiece_trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
wordpiece_tokenizer.train_from_iterator(get_training_corpus(), trainer=wordpiece_trainer)

### Post Processing ###
cls_token_id = wordpiece_tokenizer.token_to_id("[CLS]")
sep_token_id = wordpiece_tokenizer.token_to_id("[SEP]")
wordpiece_tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)]
)

### include Decoder ###
wordpiece_tokenizer.decoder = decoders.WordPiece(prefix="##")

### Encoding & Decoding Result ### 
logger.info(f"encode({sentence1, sentence2})")
encoding = wordpiece_tokenizer.encode(sentence1, sentence2)
logger.info(f"encode result: {encoding.tokens}")
decoding = wordpiece_tokenizer.decode(encoding.ids)
logger.info(f"decode result: {decoding}")

### Save Tokenizer ###
file_name = "wordpiece_tokenizer.json"
logger.info(f"Save Wordpiece Tokenizer: {file_name}")
wordpiece_tokenizer.save(file_name)

### Create BPE Tokenizer ###
logger.info("Create BPE Tokenizer")
bpe_tokenizer = Tokenizer(models.BPE())

### Compose PreTokenizer ###
bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

### Training ###
bpe_trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
bpe_tokenizer.train_from_iterator(get_training_corpus(), trainer=bpe_trainer)

### Post Processing ###
bpe_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

### Decoder ###
bpe_tokenizer.decoder = decoders.ByteLevel()

### Result ###
logger.info(f"encode({sentence1, sentence2})")
encoding = bpe_tokenizer.encode(sentence1, sentence2)
logger.info(f"encode result: {encoding.tokens}")
result = bpe_tokenizer.decode(encoding.ids)
logger.info(f"decode result: {result}")

### Save BPE Tokenizer ### 
file_name = "bpe_tokenizer.json"
logger.info(f"Save BPE Tokenizer: {file_name}")
bpe_tokenizer.save(file_name)

### Create Unigram Tokenizer ###
unigram_tokenizer = Tokenizer(models.Unigram())

### Compose Normalizer ###
unigram_tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)

### Compose PreTokenizer ###
unigram_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

### Training ### 
special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
unigram_trainer = trainers.UnigramTrainer(
    vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
)
unigram_tokenizer.train_from_iterator(get_training_corpus(), trainer=unigram_trainer)

### Post Processing ###
cls_token_id = unigram_tokenizer.token_to_id("<cls>")
sep_token_id = unigram_tokenizer.token_to_id("<sep>")

unigram_tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
)

### Compose Decoder ###
unigram_tokenizer.decoder = decoders.Metaspace()

### Result ###
logger.info(f"encode({sentence1, sentence2})")
encoding = unigram_tokenizer.encode(sentence1, sentence2)
logger.info(f"encode result: {encoding.tokens}")
result = unigram_tokenizer.decode(encoding.ids)
logger.info(f"decode result: {result}")

### Save Unigram Tokenizer ### 
file_name = "unigram_tokenizer.json"
logger.info(f"Save Unigram Tokenizer: {file_name}")
bpe_tokenizer.save(file_name)