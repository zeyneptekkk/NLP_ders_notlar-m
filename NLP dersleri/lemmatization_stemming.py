
import nltk

nltk.download("wordnet") #wordnet : lemmatization islemi için gerekli veri tabanı

from nltk.stem import PorterStemmer  #stemming için fonksiyon

#porter stemmer nesnesini oluştur
stemmer=PorterStemmer()

words=["running","runner","ran","runs","better","go","went"]

#kelimelerin stemlerini buluyoruz bunu yaparken dde porter stemmerin steam() fonksiyonunu kullanıyoruz
stems=[stemmer.stem(w) for w in words]
print(stems)


#%%lematization

from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()
words=["running","runner","ran","runs","better","go","went"]

lemmas=[lemmatizer.lemmatize(w,pos="v") for w in words]

print(lemmas)