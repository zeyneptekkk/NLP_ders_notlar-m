import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


#ornek metin olustur
documents=["Köpek çok tatlı bir hayvandır"
           ,"Köpek ve kuşlar çok tatlı hayvanlardır.",
           "Inekler süt üretirler"]

#vektorizer tanimla
tfidf_vectorizer=TfidfVectorizer()

#metinleri sayisal hale getir
x=tfidf_vectorizer.fit_transform(documents)

#kelime kumesini incele
feature_names=tfidf_vectorizer.get_feature_names_out()

#vektor temsilini incele 
vector_temsili=x.toarray()
df_tfidf=pd.DataFrame(vector_temsili,columns=feature_names)

#ortalama tf idf degerlerine bakalım
tf_idf=df_tfidf.mean(axis=0)

