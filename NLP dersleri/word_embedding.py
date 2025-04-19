
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #principle component analysis:dimension reduction
from gensim.models import Word2Vec,FastText         # Kelime vektörleştirm
from gensim.utils import simple_preprocess  # Metin ön işleme


#örnek veri setioluşturma
sentences=[
    "Köpek çok tatlı bir hayvandır",
    "Köpekler evcil hayvanlardır",
    "Kediler genellikle bağımsız hareket etmeyi severler",
    "Köpekler sadık ve dost canlısı hayvanlardır",
    "Hayvanlar insanlar için iyi arkadaşlardır"
    
    ]

tokenized_sentences=[simple_preprocess(sentence) for sentence in sentences]
 

# word2vec 
word2vec_model=Word2Vec(sentences=tokenized_sentences,vector_size=50,window=5,min_count=1,sg=0)


#fasttext  ( FastText modelini eğitmek için kullanılır. FastText, Word2Vec'in geliştirilmiş bir versiyonudur ve kelime gömme 
#(word embedding) işlemlerinde daha başarılı sonuçlar verir. FastText, özellikle morfolojik açıdan zengin diller (Türkçe gibi) için Word2Vec'ten daha iyi performans gösterir,
# çünkü kelimelerin alt kelime (subword) bilgilerini de kullanır.)
fastText_model=FastText(sentences=tokenized_sentences,vector_size=50,window=5,min_count=1,sg=0)



#görselleştirme:PCA
def plot_word_embedding(model,title):
    word_vectors=model.wv
    
    words=list(word_vectors.index_to_key)[:1000]
    vectors=[word_vectors[word] for word in words]
    
    #PCA
    pca=PCA(n_components=3)
    reduced_vectors=pca.fit_transform(vectors)
    
    #3d görselleştirme
    fig=plt.figure(figsize=(12,10))
    ax=fig.add_subplot(111,projection="3d")    
    
    #vectorelri çizdirelim
    ax.scatter(reduced_vectors[:,0],reduced_vectors[:,1],reduced_vectors[:,2])
    
    #kelimeleri etiketle
    for i,word in enumerate(words):
        ax.text(reduced_vectors[i,0],reduced_vectors[i,1],reduced_vectors[i,2],word,fontsize=12)
        
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()
 
    
 
plot_word_embedding(word2vec_model ,"Word2Vec")  
plot_word_embedding(fastText_model, "FastText")  












