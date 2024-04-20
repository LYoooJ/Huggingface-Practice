## Build a New Tokenizer
이번 섹션에서는 처음부터 Tokenizer을 빌드해보도록 한다.</br>
토큰화는 다음과 같은 과정을 통해 이루어진다.
</br>
```
1. 정규화(Unicode 정규화, 악센트 제거 등)
2. 사전 토큰화(입력 텍스트를 단어들로 분리)  
3. 모델에서 입력 실행(사전 토큰화된 단어들로 토큰 시퀀스 생성)
4. 후처리(특수 토큰 추가, attention mask, 토큰 유형 ID 생성)
```
customized된 Tokenizer를 build하기 위해 다양한 타입의 components를 Tokenizer에 사용할 수 있으며, 제공되는 components는 아래와 같다.
- `normalizers`</br>
입력 문자열에 대한 전처리로, Unicode 정규화, lowercasing 등을 수행할 수 있는 Normalizer들이 존재한다. </br>
Examples> 
    * <strong>normalizers.NFD()</strong>: NFD 정규화
    * <strong>normalizers.Lowercase()</strong>: 모든 대문자를 소문자로 변경
    * <strong>normalizers.StripAccents()</strong>: unicode에서 모든 악센트를 제거(NFD와 함께 사용)
    * <strong>normalizers.Sequence()</strong>: 여러 normalizers들을 제공된 순서대로 실행되도록 결합 </br> </br>
- `pre_tokenizers`</br>
PreTokenizer는 규칙에 따라 입력 문자열을 분할하는 역할을 한다.  </br> 
Examples> 
    * <strong>pre_tokenizers.ByteLevel()</strong>: 공백으로 입력 문자열을 분리하며, 모든 바이트를 visibel characters의 집합으로 remapping한다.
    * <strong>pre_tokenizers.Whitespace()</strong>: 공백과 문장부호를 기준으로 분할한다.
    * <strong>pre_tokenizers.CharDelimiterSplit()</strong>: 주어진 문자를 기준으로 분할한다.
    * <strong>pre_tokenizers.Sequence</strong>: 여러 PreTokenizer들이 제공된 순서대로 실행되도록 결합 </br> </br>
- `models`</br>
tokenize를 위해 사용되는 핵심 알고리즘  </br>
Examples> 
    * <strong>models.WordLevel()</strong>: 단순히 입력으로 받은 tokens들을 ID로 매핑한다.
    * <strong>models.BPE()</strong>: 가장 빈번하게 발생하는 토큰 쌍을 병합하여 새로운 토큰을 만든다.
    * <strong>models.WordPiece()</strong>: BERT에서 사용되는 subword tokenization 알고리즘으로 긴 단어를 만들고, 전체 단어가 vocabulary에 존재하지 않으면 여러 개의 Token으로 분할하며, 단어의 시작 부분이 아닌 Token에 대해 prefix로 ##을 사용한다.
    * <strong>models.Unigram()</strong>: subword tokenization 알고리즘으로 주어진 문장에 대한 확률을 최대화하는 최고의 subword token 집합을 찾는 방식으로 동작한다.  </br> </br>
- `trainers`</br>
말뭉치로 모델을 훈련한다. </br>
Examples> 
    * <strong>trainers.BpeTrainer()</strong>: 
    BPE 모델을 훈련할 수 있는 Trainer
    * <strong>trainers.UnigramTrainer()</strong>: 
    Unigram 모델을 훈련할 수 있는 Trainer
    * <strong>trainers.WordLevelTrainer()</strong>: WordLevel 모델을 훈련할 수 있는 Trainer
    * <strong>trainers.WorldPieceTrainer()</strong>: WordPiece 모델을 훈련할 수 있는 Trainer  </br> </br>
- `processors`</br>
tokenized된 문자열에 입력하기 전, 특수한 token을 삽입해준다. </br>
Examples> 
    * <strong>processors.TemplateProcessing()</strong>: 특수한 token을 삽입할 수 있으며, 단일 문장과 문장 쌍에 대해 토큰을 삽입하는 방식을 지정해주어야 하고, 콜론 뒤에 type_id(두 개의 문장이 주어질 때 구분하기 위한 별도의 id)를 지정해준다.  </br> </br>
- `decoders`</br>
토큰화된 출력을 디코딩한다. </br>
Examples> 
    * <strong>decoders.BPEDecoder()</strong>: 
    BPE 디코더
    * <strong>decoders.ByteLevel()</strong>:
    ByteLevel 디코더
    * <strong>decoders.WordPiece()</strong>: 
    WordPiece 디코더 </br> </br>
### <Example> WordPiece Tokenizer 빌드
WikiText-2 데이터셋을 사용해, WordPiece Tokenizer를 빌드해본다. 

```python
from datasets import load_dataset

### Load the dataset ###
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

### generator of batch of 1000 texts ###
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
```
우선 load_dataset() 함수를 이용해 WikiText-2 데이터셋을 로드하고, Tokenizer의 학습을 위해 1000개의 텍스트를 포함하는 batch를 생성하는 generator을 반환하는 get_training_corpus() 함수를 정의한다.</br></br>
```python
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)

### create WordPiece Tokenizer ###
wordpiece_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
```
다음으로 WordPiece 모델을 사용하여 Tokenizer를 생성하며, 이때 Tokenizer가 unknown word에 사용할 토큰인 `unk_token`을 지정해준다.</br></br>
```python
### Compose Normalizer ### 
wordpiece_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
```
토큰화의 첫번째 단계인 정규화를 위해 Normalizer를 구성해준다. normalizer.Sequence()를 통해 여러 Normalizer가 순차적으로 진행되도록 결합할 수 있으며,이 예제에서는 `NFD Unicode Normalizer`(유니코드 정규화), `Lowercase Normalizer`(모든 대문자를 소문자로 변경), `StripAccent Normalizer`(악센트 제거)를 통해 Normalizer를 구성하였다. (StripAccents Normalizer가 정상적으로 악센트를 제거하기 위해서는 NFD Unicode Normalizer의 사용이 필요하다.)</br></br>
```python
### Compose Pre-Tokenizer ###
wordpiece_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
```
사전 토큰화를 위해 Pre-tokenizer를 빌드해주며, `Whitespace Pre-tokenizer`는 공백과 문장부호를 기준으로 문장을 분할한다. </br></br>
```python
### Trainer ###
#All special tokens to use
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"] 
wordpiece_trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
wordpiece_tokenizer.train_from_iterator(get_training_corpus(), trainer=wordpiece_trainer)
```
앞서 생성헀던 WordPiece Tokenizer의 학습을 진행하여야 하며, 이를 위해 `WordPieceTrainer`을 인스턴스화해주어야 한다. 이때, 사용하고자 하는 모든 special token을 trainer에 전달해주어야 모든 special token들이 vocabulary에 추가되며, 위 예제에서는 `vocab_size`를 통해 vocabulary의 사이즈를 25000으로 지정해주었다. </br>
이제 앞서 정의했던 `get_training_corpus()`를 활용해, `train_from_iterator()`를 통해 모델을 학습시킨다.</br></br>
```python
### Post Processing ###
cls_token_id = wordpiece_tokenizer.token_to_id("[CLS]")
sep_token_id = wordpiece_tokenizer.token_to_id("[SEP]")
wordpiece_tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)]
)
```
토큰화의 마지막 단계인 Post processing을 통해 문장의 시작에 `[CLS]` 토큰을, 문장의 끝에 `[SEP]` 토큰을 추가해준다. 먼저 `token_to_id`를 통해 학습된 Tokenizer의 vocabulary에서 [CLS], [SEP] 토큰의 ID를 알아내고, `TemplateProcessor`을 통해 단일 문장과 문장쌍을 처리하는 방법을 각각 지정해준다.</br> 추가하고자 하는 special token을 나타내어주며, 첫번째 문장은 $A로 나타내고 두번째 문장이 존재하는 경우 두번째 문장은 $B로 나타낸다. 이때, 토큰과 문장에 대해 `:` 뒤에 `Token type ID`(입력 문자열이 문장쌍인 경우 두 문장을 구별하는 마스크 역할을 함)를 지정해준다.
```python
### include Decoder ###
wordpiece_tokenizer.decoder = decoders.WordPiece(prefix="##")
```
마지막으로 Tokenizer에 decoder를 포함시켜, Tokenizer가 encoding 결과를 다시 문장으로 변환시킬 수 있게 한다. 해당 예제에서는 단어의 시작 부분이 아닌 Token에 대해 prefix로 `##`을 사용하므로 위와 같이 지정해준다.</br></br>
```python
### Encoding & Decoding Result ### 
sentence1 = "Let's test this tokenizer..."
sentence2 = "on a pair of sentences."

encoding = wordpiece_tokenizer.encode(sentence1, sentence2)
decoding = wordpiece_tokenizer.decode(encoding.ids)
print(f"Encoding: {encoding.tokens}")
print(f"Decoding: {decoding}")
```
위의 문장쌍을 입력으로 주었을 때, 지금까지 빌드한 Tokenizer의 encoding과 decoding 결과는 아래와 같다.
```
Encoding: ['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##en', '##izer', '...', '[SEP]', 'on', 'a', 'pair', 'of', 'sentences', '.', '[SEP]']
Decoding: let ' s test this tokenizer... on a pair of sentences.
``` 
</br>

```python
### Save Tokenizer ###
file_name = "wordpiece_tokenizer.json"
wordpiece_tokenizer.save(file_name)
```
이번 예제를 통해 빌드한 Tokenizer을 JSON 파일로 저장할 수 있다. </br></br>
```python
from transformers import PreTrainedTokenizerFast

tokenizer = Tokenizer.from_file('wordpiece_tokenizer.json')

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
```
또한, `from_file()`을 이용해 저장한 Tokenizer을 다시 로드할 수 있으며, 이를 사용하기 위해서는 `PreTrainedTokenizerFast`에 `tokenizer_object` 또는 `tokenizer_file`로 Tokenizer을 전달하고, special token을 설정해주어야 한다.

## WordPiece Tokenizer 빌드 전체 코드
```python
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

### Load the dataset ###
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

### generator of batch of 1000 texts ###
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

### create WordPiece Tokenizer ###
wordpiece_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

### Compose Normalizer ### 
wordpiece_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)

### Compose Pre-Tokenizer ###
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
```
</br></br>
## 참고 문서
- https://huggingface.co/docs/tokenizers/python/latest/components.html

- https://wikidocs.net/166828

- https://huggingface.co/docs/tokenizers/python/latest/api/reference.html