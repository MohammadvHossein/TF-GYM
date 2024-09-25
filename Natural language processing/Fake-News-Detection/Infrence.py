import pandas as pd
import string
import re
import pickle
import tensorflow as tf

model = tf.keras.models.load_model('best_model_FackNews.keras')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorization = pickle.load(f)

def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def predict_message(message):
    preprocessed_message = preprocess(message)
    
    message_vectorized = vectorization.transform([preprocessed_message])
    
    prediction = model.predict(message_vectorized)
    
    prediction_label = (prediction > 0.5).astype(int)[0][0]
    
    return "True" if prediction_label == 1 else "Fake"

massage = '''
A recent report highlights the increasing challenge of fake news, particularly in the context of social media and political discourse. As misinformation spreads rapidly, it complicates public understanding of events and erodes trust in credible information sources. For instance, a sniper incident involving Donald Trump led to rampant speculation on social media, showcasing how quickly misinformation can proliferate and influence public perception
 ''' # True
print(predict_message(massage))

massage = ''''
One of the most notorious examples of fake news is the "Pizzagate" conspiracy theory, which falsely claimed that a Washington D.C. pizza restaurant was involved in a child sex trafficking ring linked to political figures. This misinformation led to a shooting incident at the restaurant, where a man entered with a rifle, believing he was rescuing children. The conspiracy was fueled by false tweets and social media posts, illustrating the dangerous consequences of fake news spreading unchecked'''
# Fack
print(predict_message(massage))

massage = '''
A notable example of fake news during the COVID-19 pandemic includes false claims about the virus's origins and misinformation regarding its treatment. For instance, some stories suggested that the virus was created by the U.S. or Chinese governments, while others promoted ineffective remedies like sipping water every 15 minutes. These types of misinformation can have dangerous implications, as they mislead the public and undermine trust in legitimate health information '''
# Fack
print(predict_message(massage))

massage = '''
Trump now is president of united states
'''
# Fack
print(predict_message(massage))

