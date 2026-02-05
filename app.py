import transformers as tf 
import gradio as gr
from transformers import pipeline 


model = pipeline ("sentiment-analysis",model =
"cardiffnlp/twitter-xlm-sentiment-multilingual")

def analyze (text) :
  result = model(text)[0]
return f"{result['label']} , {result['score']:.0% }"

gr.Interface(fn=analyze,inputs="text",outputs="text").launch()
