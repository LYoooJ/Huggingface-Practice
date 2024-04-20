# Behind the pipeline

앞서 사용해본 transformers 라이브러리의 pipeline()는 내부적으로 전처리, 모델로 입력 전달, 후처리의 3단계를 실행한다고 하였다. 이번에는 각각의 단계를 직접 실행해 아래와 같은 task에 대해 pipeline()과 같은 실행 결과를 얻어보고자 한다.

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier(["I've been waiting for a HuggingFace course my whole life.",
                     "I hate this so much!"])
print(result)
```
#### 실행 결과
```
[{'label': 'POSITIVE', 'score': 0.9598049521446228}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]
```
</br>

## 1. Processing with Tokenizer
모델은 사람의 텍스트를 직접 처리할 수 없으며, 따라서 모델에 입력을 전달하기 전 텍스트를 모델이 이해할 수 있는 형태로 변환해야 한다. 이러한 전처리를 위해 Tokenizer가 사용되며, Tokenizer는 입력된 텍스트를 token으로 분할하고, 각 token을 하나의 정수로 매핑하며, 모델에 추가적인 입력을 추가한다. 

이때, 입력 텍스트는 모델이 사전 학습될 때와 동일한 방식으로 전처리되어야 하기 때문에, 모델이 사전 학습될 때와 동일한 Tokenizer를 사용해야 한다. `AutoTokenizer` class에서 `from_pretrained()` 메서드에 사용하는 모델의 checkpoint를 넘겨주면, 해당 모델의 Tokenizer와 연결된 데이터를 Model Hub에서 가져와 캐시한다.

위 예제에서는 `sentiment-analysis task`를 위해 default 모델이 사용되었으며, 해당 task의 default checkpoint는 `distilbert-base-uncased-finetuned-sst-2-english`이다. 

### Code
```python
from transformers import AutoTokenizer

### sentiment-analysis pipeline의 default check point ###
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

### Preprocessing using tokenizer ###
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print("inputs: {0}".format(inputs))
```
sentiment-analysis pipeline의 defalut checkpoint인 `distilbert-base-uncased-finetuned-sst-2-english`의 Tokenizer를 로드하고, `return_tensors` 인자를 `pt`로 설정하여 Tokenizer가 반환하는 텐서의 유형이 `PyTorch`가 되도록하였다.
### 실행 결과
```
inputs: {'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
             0,     0,     0,     0,     0,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```
Tokenizer의 출력 유형은 위와 같이 `input_ids`, `attention_mask`를 key로 하고, `tensor`를 value로 하는 dictionary이다. 

</br>

## 2. Going through the Model
Tokenizer와 동일한 방식으로, `AutoModel` 클래스와 `from_pretrained()` 메서드에 checkpoint를 넘겨주어 사전 학습된 모델을 로드할 수 있다.
</br>하지만, Transformers에서는 모델이 지원하는 task에 따라 다른 모델 헤드를 가지고 있고, 해당 예제에서는 입력 받은 문장을 긍정문 또는 부정문으로 분류하기 위해 시퀀스 분류 헤드가 포함되어 있는 모델이 필요하기 때문에 `AutoModelForSequenceClassification` 클래스를 활용한다.
</br>(모델 헤드는 은닉 상태(hidden state)의 고차원 벡터를 입력으로 받아 다른 차원으로 투영하여 최종 출력을 생성하는 부분)
</br>

### Code
```python
from transformers import AutoModelForSequenceClassification

### Passing the inputs through the model ###
model = AutoModelForSequenceClassification.from_pretrained(checkpoint) 
outputs = model(**inputs)

### logits: raw, unnormalized scores outputted by the last layer of the Model ###
print("logits: {0}".format(outputs.logits))
```

### 실행 결과
```
logits: tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>)
```
이 예제에서 우리는 두 개의 문장을 입력했고, 부정문 또는 긍정문이라는 2개의 레이블이 존재하기 때문에 모델에서 얻은 결과는 2x2가 된다. 이때, 출력된 값은 모델의 마지막 계층에서 출력된 원시 점수이기 때문에, 이를 확률로 변환해주어야 한다.
</br>
## 3. Postprocessing the output
2단계에서 얻었던 logits 값을 확률로 변환하려면 SoftMax 계층을 통과해야 한다. 
### SoftMax
SoftMax는 입력 받은 벡터의 각 요소을 0~1 사이의 값으로 정규화하여 이산 확률 분포의 형태로 출력을 반환하며, 출력 벡터의 요소들의 합은 1이 된다.

$$ SoftMax_{i}(x) = \frac{e^{x_{i}}}{\sum\limits_{j=1}^{n}e^{x_{j}}} (x \in \mathbb{R^{n}}, i \in [1, n])$$
### Code
```python
import torch

### postprocessing ###
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

### probability ### 
print("probability: {0}".format(predictions))

### prediction label ### 
print("prediction label: {0}".format(model.config.id2label))
```

### 실행 결과
```
probability: tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward0>)
prediction label: {0: 'NEGATIVE', 1: 'POSITIVE'}
```

logits을 SoftMax 계층에 통과시킨 후 출력해보면, 모델이 첫번째 문장에 대해 [0.040195, 0.95980], 두번째 문장에 대해 [0.99946, 0.00054418]로 예측했음을 알 수 있다. 또한 `id2label` 속성 값을 통해 각 위치에 해당하는 레이블(`POSITIVE` 또는  `NEGATIVE`)을 가져올 수 있으며, 따라서 0번째의 값이 부정문일 확률을, 1번째의 값이 긍정문일 확률을 나타냄을 확인할 수 있다.
따라서 모델이 예측한 결과는 아래와 같다.
</br>
* 첫번째 문장 -> NEGATIVE: 0.040195, POSITIVE: 0.95980

* 두번째 문장 -> NEGATIVE: 0.99946, POSITIVE: 0.00054418