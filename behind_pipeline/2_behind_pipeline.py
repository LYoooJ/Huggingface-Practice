from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import logging

logging.basicConfig(filename='behind_pipeline.log', level=logging.INFO, filemode='w')

#sentiment-analysis pipeline의 default check point
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
logging.info('checkpoint for sentiment-analysis: {0}'.format(checkpoint))

### Preprocessing using tokenizer ###
logging.info('Preprocessing Using Tokenizer')
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
logging.info("inputs: {0}".format(inputs))

### Passing the inputs through the model ###
logging.info('Passing the inputs throught the model')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint) 
outputs = model(**inputs)

### logits: raw, unnormalized scores outputted by the last layer of the Model ###
logging.info("logits: {0}".format(outputs.logits))

### postprocessing ###
logging.info('Postprocessing')
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

### probability ### 
logging.info("probability: {0}".format(predictions))

### prediction label ### 
logging.info("prediction label: {0}".format(model.config.id2label))