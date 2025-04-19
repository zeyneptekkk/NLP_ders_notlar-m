import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords") 
stop_words_eng=set(stopwords.words("english"))

#veri seti yükleme
df=pd.read_csv('C:/NLP dersleri/IMDB Dataset.csv')
documents=df["review"]

#metin temizlme işleme
def clean_text(text):
    text=text.lower()              #kücük harfe cevir
    text=re.sub(r"\d+","",text)    #sayiları temizle  
    text=re.sub(r"[^\w\s]","",text) # özel karakterleri temizle
    text=" ".join([word for word in text.split() if len(word)>2])
   
    text=[word for word in text.split() if word not in stop_words_eng ]

    return text

cleaned_documments=[clean_text(doc) for doc in documents]


#metin tokenization
tokenized_documents=[simple_preprocess(doc) for doc in cleaned_documments]

#%%

#Wor2Vec modeli tanımla
model=Word2Vec(sentences=tokenized_documents,vector_size=50,window=5,min_count=1,sg=0)
word_vectors=model.wv
words=list(word_vectors.index_to_key[:500])
vectors=[word_vectors[word] for word in words]


#Clustering KMeans K=2
kmeans=KMeans(n_clusters=2)   # n_clusters, oluşturulacak küme sayısıdır
# Modeli eğit
kmeans.fit(vectors)           # vectors, kümeleme yapılacak veri matrisidir
clusters=kmeans.labels_      # 0 ,1


#PCA 50->2
pca=PCA(n_components=2)
reduced_vectors=pca.fit_transform(vectors)


#2 boyutlu bir görselleştirme
plt.figure()
plt.scatter(reduced_vectors[:,0],reduced_vectors[:,1],c=clusters,cmap="viridis")

centers=pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],c="red",marker="x",s=130,label="Center")
plt.legend()


#figür üzerine kelimelerin eklenmesi
for i,word in enumerate(words):
    plt.text(reduced_vectors[i,0],reduced_vectors[i,1],word,fontsize=7)
plt.title("Word2Vc")














