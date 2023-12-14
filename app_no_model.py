import requests
import streamlit as st
st.set_page_config(page_title="Fake News Detector", page_icon=":rolled_up_newspaper:", layout="wide")

try:
  from beautifulsoup import BeautifulSoup as bs
except:
  from bs4 import BeautifulSoup as bs

import pandas as pd
import numpy as np


import nltk
from nltk.corpus import stopwords


import re



from functools import reduce

#nltk packages
nltk.download('stopwords')
nltk.download('wordnet')
nltk.data.path.append('/root/nltk_data/corpora/')
stop_words = stopwords.words('english')
print(stop_words) # some words I like to remove are not included
new_words = ['said','like','year','would','house','also','sends']
stop_words.extend(new_words)
stop_words = set(stop_words)
nltk.download('punkt')

#functions

def scrape_url(url):
  article = requests.get(url)
  soup = bs(article.content, 'html.parser')
  text = [p.getText() for p in soup.find_all('p')]
  return text


# Create function to automatically lemmatization and remove stopwords
def lemmatization_and_stopwords(text):
    clean_text = []
    # Set all text into lowercase to match the stopwords
    #text = [x.lower() for x in text]
    #text = [x.strip() for x in text]
    text = text.lower()
    #regex = re.compile('[^a-zA-Z ]')
    #text = [regex.sub('', x) for x in text]
    # Tokenize the text before processing
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()

    for token in tokens:
        if token not in stop_words and len(token)>3:
            token = lemmatizer.lemmatize(token)
            clean_text.append(token)

    text = " ".join(clean_text)

    return clean_text


def text_cleaning(text):
  text_df = pd.DataFrame(text)
  text_df.rename(columns={0:'text'},inplace=True)
  text_df = text_df['text'].astype(str)
  text_df = text_df.apply(lemmatization_and_stopwords)
  t = text_df[0]
  clean_text = t
  return clean_text



def predicting(text):

  answer = 0.6
  if answer > 0.5:
    return f"I think it's fake news! Be skeptic! confidence: {answer}"
  else:
    return f"Seems legit to me! confidence: {answer}" 

def main_article_check(url):
  with st.status("Prepering prediction...",expanded=True) as status:
    st.write("Scraping the web...")
    text = scrape_url(url)
    st.write("Found text")
    st.write("Cleaning text")
    text = text_cleaning(text)
    st.write("Text is cleaned")
    st.write("Asking the imp what he thinks")
    answer = predicting(text)
    status.update(label="Process complete!", state="complete",expanded=False)
  return answer

#images
troll1 = "https://github.com/Shacham-R/fake_news_detector/blob/main/streamlit_app_data/troll1.png?raw=true"
imp = "https://github.com/Shacham-R/fake_news_detector/blob/main/streamlit_app_data/box_imp.png?raw=true"
RF_FI = "https://github.com/Shacham-R/fake_news_detector/blob/main/streamlit_app_data/Random_forest_feature_importance.png?raw=true"
DT_EM = "https://github.com/Shacham-R/fake_news_detector/blob/main/streamlit_app_data/Decision_tree_error_metrics.png?raw=true"
decision_tree_diagramm = "https://github.com/Shacham-R/fake_news_detector/blob/main/streamlit_app_data/decistion_tree.png?raw=true"
word_cloud_fake = "https://github.com/Shacham-R/fake_news_detector/blob/main/streamlit_app_data/word_cloud_fake.png?raw=true"
word_cloud_real = "https://github.com/Shacham-R/fake_news_detector/blob/main/streamlit_app_data/word_cloud_real.png?raw=true"


tab1, tab2, tab3 = st.tabs(["Overview", "Process", "Demo"])


#----------------------

with tab1:
    col11, col12 = st.columns([0.3,0.7])
    with col11:
            st.image(troll1,"A troll with his laptop, spreading misinformation")
    with col12:
            st.header("Overview")
            st.markdown('''
    - *Business question:* in our turbulent time, how do    you moderate the content on your site? Trolls are all around… Protect yourself with our fake news detector!

- *MVP:* model that classifies text as real or fake news (Binary Classification) using NLP methods with tensorflow - keras.

- *Data source:* For the MVP- Kaggle, Bonus - news sites
        ''')
    if st.button("click me!"):
        st.snow()

    
#----------------------
 
with tab2:
    st.header("EDA")
    st.write('''- Data cleaning:
    - Removing text with RE
    - Statistical analysis:
            ''')
    st.write("Word frequency")
    st.image(word_cloud_real, "most frequent words labeled 'real'.")
    st.image(word_cloud_fake, "most frequent words labeled 'fake'.")
    "---"
    st.header("Modeling")
    st.write("Decision Tree")
    st.image(decision_tree_diagramm, "No metter how deep it goes...")
    st.image(DT_EM, "A lone decision tree doesn't give the best results.")
    st.write("Random Forest")
    st.write("feature importance")
    st.image(RF_FI, "Even a whole forest can't decide what's important factor.",width=250)
    st.write("SNN")
    "---"
    st.header("Streamlit app")
    st.header("Dealing with the Cloud")
        
    
    
        



        
        
        

#----------------------

with tab3:
    col31,col32 = st.columns([0.3,0.7])
    with col31:
            st.image(imp, "The imp that provides the answers lived all i'ts life in a box, and is not very smart.")
    with col32:
        st.header("Demo")
        st.warning("This is a prototype, plaese do not take seriously")
        st.text("Example URLs: \nhttps://www.bbc.com/news/world-us-canada-67710761\nhttps://www.bbc.com/news/world-asia-china-67689072")
        url = st.text_input('Please enter a url to test')  
        if url:
          article_text = scrape_url(url)
          text_area = st.text_area("Article Text",article_text)
        

        if st.session_state.get('predict'):
            answer = main_article_check(url)
            st.info(answer)
        st.button("Predict",key="predict")
