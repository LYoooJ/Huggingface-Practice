## Pipeline

<strong>pipeline()</strong>: Transformers 라이브러리의 가장 기본적인 객체

pipeline에 텍스트를 입력하면 내부적으로 3단계를 거쳐 결과를 출력한다. 

1. 입력된 텍스트를 모델이 학습할 수 있는 형식으로 전처리
2. 전처리한 입력 텍스트를 모델에 전달
3. 모델이 예측한 결과를 사람이 이해할 수 있는 형태로 후처리

## 주요 parameters
<strong>task</strong>: pipeline을 통해 수행할 task

example:  "sentiment-analysis", "text-generation", "summarization", etc.

<strong>model</strong>: pipeline이 prediction을 만들어내기 위해 사용할 모델, 입력하지 않는 경우 task에 대해 지정된 default model이 사용된다.

<strong> config</strong>: pipeline이 모델을 초기화하는데 사용할 configuration, 입력하지 않는 경우 사용되는 모델의 default configuration이 사용된다.

<strong>tokenizer</strong>: pipeline이 텍스트를 인코딩하는데 사용할 tokenizer, 입력하지 않는 경우 사용되는 모델의 default tokenizer가 사용된다. 

## Practice
### - default model 사용
#### Code
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier(["I've been waiting for a HuggingFace course my whole life.",
                     "I hate this so much!"])
print(result)
```
위 코드처럼 List를 이용해 여러 개의 문장을 동시에 입력할 수 있다.
#### 실행 결과
```
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
[{'label': 'POSITIVE', 'score': 0.9598049521446228}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]
```
model을 지정하지 않아, distilbert-base-uncased-finetuned-sst-2-english 모델이 default로 사용되었음을 확인할 수 있다. 

### - model 지정
#### Code
```python
from transformers import pipeline
generator = pipeline("text-generation", model="distilgpt2")
result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(result)
```
model parameter에 특정 모델을 지정해주어 원하는 모델로 task를 수행할 수 있다. 위 코드에서는 distilgpt2 모델을 로드해 text 생성 task를 수행한다. 또한 위 코드에서는 generator 객체에 max_length 인자를 30으로 설정하여 출력 텍스트의 길이를 제어하고, num_return_sequences 인자를 2로 지정하여 실행 결과 2개의 문장이 생성된다.
#### 실행 결과
```
[{'generated_text': 'In this course, we will teach you how to be better at your learning. We will talk about many things, and why you need to consider all'}, {'generated_text': 'In this course, we will teach you how to integrate our new curriculum into our classrooms, including what makes new content relevant to the classroom.\n\n'}]
```