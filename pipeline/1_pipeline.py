from transformers import pipeline
import logging

logging.basicConfig(filename='pipeline.log', level=logging.INFO, filemode='w')

###Sentiment-analysis###
logging.info('Sentiment-analysis')
classifier = pipeline("sentiment-analysis")
result = classifier(["I've been waiting for a HuggingFace course my whole life.",
                     "I hate this so much!"])
logging.info(result)

###Zero-shot classification###
logging.info('Zero-shot Classification')
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"]
)
logging.info(result)

###Text Generation###
logging.info('Text Generation')
generator = pipeline("text-generation", model="distilgpt2")
result = generator("In this course, we will teach you how to",
                    max_length=30,
                    num_return_sequences=2,
                    pad_token_id=generator.tokenizer.eos_token_id
                    )
logging.info(result)

###Mask filling###
logging.info('Mask Filling')
unmasker = pipeline("fill-mask")
result = unmasker("This course will teach you all about <mask> models.", top_k=2)
logging.info(result)

###Named entity recognition###
logging.info('Named Entity Recognition')
ner = pipeline("ner", grouped_entities=True)
result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
logging.info(result)

###Qestion Answering###
logging.info('Question Answering')
question_answer = pipeline("question-answering")
result = question_answer(
         question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
logging.info(result)

###Summarization###
logging.info('Summarization')
summarizer = pipeline("summarization")
result = summarizer("""
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
""")
logging.info(result)

###Translation###
logging.info('Translation')
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result = translator("Ce cours est produit par Hugging Face")
logging.info(result)