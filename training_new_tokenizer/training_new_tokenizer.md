## Training a New Tokenizer
이번 섹션에서는 기존 Tokenizer을 이용해 새로운 Tokenizer를 학습시켜보도록 한다. 기존의 Tokenizer을 이용하면 토큰화 알고리즘, special tokens들에 대해 따로 지정해주지 않아도 되며, 기존의 Tokenizer와 동일하지만 다른 말뭉치를 이용한 학습으로 인해 vocabulary만 변경되는 결과를 얻을 수 있다.</br></br>
이번 예제에서는 gpt2 Tokenizer을 python 소스코드 말뭉치를 이용해 새롭게 학습시켜보고자 한다. 그렇다면, 우선 기존의 gpt2 Tokenizer에서 python 함수 정의 소스코드를 어떻게 처리하는지 살펴보자.
```python
from transformers import AutoTokenizer

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
old_tokens = old_tokenizer.tokenize(example)
print(old_tokens)
```
</br>

```
['def', 'Ġadd', '_', 'n', 'umbers', '(', 'a', ',', 'Ġb', '):', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`', 'a', '`', 'Ġand', 'Ġ`', 'b', '`', '."', '""', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```
gpt2 Tokenizer에서 `Ċ`는 줄바꿈을 나타내는 특수기호이며, `Ġ`는 공백을 나타내는 특수기호이다. 기존의 gpt2 Tokenizer가 위의 python 소스코드를 처리한 결과를 살펴보면, 소스코드에서 공백이 그룹으로 나타나는 것이 일반적임에도 여러 개의 공백을 그룹화하지 않고 하나의 공백을 하나의 token으로 표현하고, 'n', 'umbers'와 같이 함수명이 이상하게 분할되고 있는 것을 확인할 수 있다. 이처럼 기존의 gpt2 Tokenizer는 python 소스코드를 잘 처리하지 못하며, 이번 섹션에서는 python 소스코드 말뭉치를 이용해 새롭게 학습한 gpt2 Tokenizer가 이러한 문제를 해결할 수 있을지 확인해보고자 한다.</br> </br>

### 말뭉치 수집
이번 예제에서는 기존의 Tokenizer를 python 소스코드 말뭉치를 이용해 새롭게 학습시켜보고자 하며, 따라서 우선 CodeSearchNet 데이터셋의 python 부분을 로드한다.
```python
from datasets import load_dataset

### Load dataset ###
raw_datasets = load_dataset("code_search_net", "python")

print(raw_datasets)
```
```
DatasetDict({
    train: Dataset({
        features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 'func_code_url'],
        num_rows: 412178
    })
    test: Dataset({
        features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 'func_code_url'],
        num_rows: 22176
    })
    validation: Dataset({
        features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 'func_code_url'],
        num_rows: 23107
    })
})
```
데이터셋이 위와 같은 구조를 가졌음을 확인할 수 있다.</br>
<details>
  <summary>whole_func_string</summary>
</br> 

  ```python
  print(raw_datasets["train"][10]["whole_func_string"])
  ```

  ```python 
    def __create_nlinks(self, data, inds=None, boundary_penalties_fcn=None):
            """
            Compute nlinks grid from data shape information. For boundary penalties
            are data (intensities) values are used.

            ins: Default is None. Used for multiscale GC. This are indexes of
            multiscale pixels. Next example shows one superpixel witn index 2.
            inds = [
                [1 2 2],
                [3 2 2],
                [4 5 6]]

            boundary_penalties_fcn: is function with one argument - axis. It can
                it can be used for setting penalty weights between neighbooring
                pixels.

            """
            # use the gerneral graph algorithm
            # first, we construct the grid graph
            start = time.time()
            if inds is None:
                inds = np.arange(data.size).reshape(data.shape)
            # if not self.segparams['use_boundary_penalties'] and \
            #         boundary_penalties_fcn is None :
            if boundary_penalties_fcn is None:
                # This is faster for some specific format
                edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
                edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
                edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]

            else:
                logger.info("use_boundary_penalties")

                bpw = self.segparams["boundary_penalties_weight"]

                bpa = boundary_penalties_fcn(2)
                # id1=inds[:, :, :-1].ravel()
                edgx = np.c_[
                    inds[:, :, :-1].ravel(),
                    inds[:, :, 1:].ravel(),
                    # cc * np.ones(id1.shape)
                    bpw * bpa[:, :, 1:].ravel(),
                ]

                bpa = boundary_penalties_fcn(1)
                # id1 =inds[:, 1:, :].ravel()
                edgy = np.c_[
                    inds[:, :-1, :].ravel(),
                    inds[:, 1:, :].ravel(),
                    # cc * np.ones(id1.shape)]
                    bpw * bpa[:, 1:, :].ravel(),
                ]

                bpa = boundary_penalties_fcn(0)
                # id1 = inds[1:, :, :].ravel()
                edgz = np.c_[
                    inds[:-1, :, :].ravel(),
                    inds[1:, :, :].ravel(),
                    # cc * np.ones(id1.shape)]
                    bpw * bpa[1:, :, :].ravel(),
                ]

            # import pdb; pdb.set_trace()
            edges = np.vstack([edgx, edgy, edgz]).astype(np.int32)
            # edges - seznam indexu hran, kteres spolu sousedi\
            elapsed = time.time() - start
            self.stats["_create_nlinks time"] = elapsed
            logger.info("__create nlinks time " + str(elapsed))
            return edges 
  ```
  위의 코드를 통해 데이터셋의 `whole_func_string`열의 예시를 하나 확인해보면 위와 같은 결과를 얻을 수 있으며, python function definition을 값으로 가지고 있음을 확인할 수 있다.
</details>
</br>

```python
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"] for i in range(0, len(raw_datasets["train"]), 1000)
    )
```
이번 예제에서는 해당 데이터셋의 train 영역의 whole_func_string을 말뭉치로 하여 기존의 gpt2 Tokenizer의 학습을 진행할 것이다. 이를 위해 1000개의 텍스트를 가지는 batch를 생성하는 generator를 반환하는 `get_training_corpus()` 함수를 정의해준다.

## Tokenizer 학습
```python
from transformers import AutoTokenizer

### Load GPT-2 Tokenizer ###
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```
우선 gpt2 checkpoint를 이용해 GPT-2 Tokenizer을 로드한다.
```python
tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), 52000)
```
다음으로 `train_new_from_iterator()` 메소드를 이용해 python 소스코드 말뭉치로 새로운 Tokenizer를 학습한다.</br>
```python
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = tokenizer.tokenize(example)
print(tokens)
```
```
['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`', 'a', '`', 'Ġand', 'Ġ`', 'b', '`."""', 'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```
학습이 끝난 새로운 Tokenizer에 아까의 python 함수 정의 소스코드를 입력하면, 기존의 gpt2 Tokenizer에 비해 들여쓰기를 나타내는 `ĊĠĠĠ` 토큰, 세 개의 따옴표를 나타내는 `Ġ"""` 토큰 등 python 함수에 특화적인 토큰을 새롭게 학습했음을 확인할 수 있고, 기존과는 달리 함수명 또한 잘 분할하고 있음을 확인할 수 있다.</br></br>
```python
### Save the new tokenizer ###
tokenizer.save_pretrained("code-search-net-tokenizer")
```
이처럼 새롭게 학습한 Tokenizer를 위와 같이 저장하여 나중에 로드하여 사용할 수 있게 할 수 있다.</br></br>

## Training a new tokenizer 전체 코드
```python
from transformers import AutoTokenizer
from datasets import load_dataset

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

### Load the dataset ###
raw_datasets = load_dataset("code_search_net", "python")

### generator of batch of 1000 texts ###
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"] for i in range(0, len(raw_datasets["train"]), 1000)
    )

### Load GPT-2 Tokenizer ###
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), 52000)

### Tokenize example ###
tokens = tokenizer.tokenize(example)

### Save the new tokenizer ###
tokenizer.save_pretrained("code-search-net-tokenizer")
```
</br></br>
## 참고 문헌
- https://wikidocs.net/166821