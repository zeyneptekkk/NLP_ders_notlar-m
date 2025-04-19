
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords") 



#veri setinin içeriye aktarılması
df=pd.read_csv("C:/NLP dersleri/IMDB Dataset.csv")

#metin verilerini alalım
documents=df["review"]
lalbels=df["sentiment"] 

#metin temizleme
def clean_text(text):
    #buyuk kucuk harf cevirimi
    text=text.lower()
    
    #rakamları temizle
    text=re.sub(r"\d+","",text)
    
    #ozel karakterlerin kaldırılması
    text=re.sub("r[^\w\s]","",text)
    
    #kısa kelimelerin temizlenmesi
    text=" ".join([word for word in text.split() if len(word)>2])
    
    #stop words leri cıkartma
    stopW=set(stopwords.words("english"))
    textt=text.split()
    text=[w for w in textt if w not in stopW ]
    
    return text

#metinleri temizle
cleaned_doc=[clean_text(row) for row in documents]



#%%bow

#vectorizer tanimla
vectorizer=CountVectorizer()

#metin sayısal hale getir
x=vectorizer.fit_transform(cleaned_doc[:75])


#kelime kumesi goster
features_names=vectorizer.get_feature_names_out()

#vektor temsili
vector_temsili2=x.toarray()
df_bow=pd.DataFrame(vector_temsili2,columns=features_names)


#kelimelerin frekansını göster
word_caunts=x.sum(axis=0).A1
word_freq=dict(zip(features_names,word_caunts))

#ilk 5 kelimeyi print et
most_common_5_words=Counter(word_freq).most_common(5)
print(most_common_5_words
      )












