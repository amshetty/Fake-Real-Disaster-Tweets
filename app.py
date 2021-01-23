import streamlit as st
import joblib
import pandas as pd
import nltk
nltk.download('punkt')
import re
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

page_bg_img = '''
<style>
body {
background-image: url("https://images.pexels.com/photos/1939485/pexels-photo-1939485.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
pstem = PorterStemmer()
def clean_text(text):
    text= text.lower()
    text= re.sub('[0-9]', '', text)
    text  = "".join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens=[pstem.stem(word) for word in tokens]
    text = ' '.join(tokens)
    return text

def main():

    st.title('Fake/Real Disaster Tweet Detection')
    st.subheader('A Natural Language Processing Project')

    # Preprocessing
    tfidf = joblib.load('vectorizer.h5')
    message = st.text_area('Enter Test Tweet')
    message = clean_text(message)
    data = pd.Series(data = message)
    data = tfidf.transform(data)

    #model predict
    model = joblib.load('model.h5')
    pred = model.predict(data)


    if st.button('Verify Tweet'):
        if pred == 0:
            st.error('Result: Fake Disaster Tweet')

        else:
            st.success('Result: Real Disaster Tweet')



if __name__ == '__main__':
    main()
