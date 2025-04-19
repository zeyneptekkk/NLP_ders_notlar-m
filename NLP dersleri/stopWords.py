import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")  #farklı dillerde en cok kullanılan stopwordsleri içerir
 

#ingilizce stop words analizi(nltk)

stop_words_eng=set(stopwords.words("english"))

text="there are some examples of handling stop words from a texts."
text_list=text.split()
filtered_words_tr=[word for word in text_list if word.lower() not in stop_words_eng ]

#%%türkce stop words analizi(nltk)

stop_words_tr=set(stopwords.words("turkish"))

metin="merhaba arkadasslar cok guzel bir ders işliyoruz"
text_list_tr=metin.split()
filtered_words_tr=[word for word in text_list_tr if word.lower() not in stop_words_tr]

#%%kutuphanesiz stop wwords cıkarımı

tr_stop_words=["için","ile","mu","mi","özel"]

metin="bu bir denemedir amacımız bu listede bulunan özel karakterleri elemek mi acaba?"
filtred_words=[w for w in metin.split() if w.lower() not in tr_stop_words ]
print(filtred_words)