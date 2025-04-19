import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter

#ornek veri seti olustur
corpus=[
        "I love apple",
        "I love him",
        "I love NLP",
        "You love me",
        "He loves apple",
        "They love apple",
        "I love you and you love me"
        ]

"""

problem tanimi yapalım :
    dil modeli yapmak istiyoruz
    amac 1 kelimeden sonra gelecek kelimeyi tahmin etmek :metin turetmek/olusturmak
    bunun için n-gram dil modelini kullanıcaz
    
    ex:I ... (love)...(apple)
    
"""

#verileri token haline getir

tokens=[word_tokenize(sentence.lower()) for sentence in corpus]

#bigram 2 li kelime grupları olustur
bigrams=[]
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))


bigrams_freq=Counter(bigrams)


#trigram
trigrmas=[]
for token_list in tokens:
    trigrmas.extend(list(ngrams(token_list, 3)))

trigram_freq=Counter(trigrmas)


#model testing
#"ı love " bigramından sonra "you" veya "apple" kelimelerinin gelme olasılıklarını hesaplayalım

bigram=("i","love")  #heddef bigram

# " i love you" olma olasılığı 
prob_you=trigram_freq[("i","love","you")]/bigrams_freq[bigram]
print(f"you kelimesinin olma olasılığı : {prob_you}")   # ı love den sonra you gelme olasılığı 0,25


# ı love apple ola olasılıgı
prob_apple=trigram_freq[("i","love","apple")]/bigrams_freq[bigram]
print(f"apple kelimesinin olma olasılığı : {prob_apple}")   # ı love den sonra you gelme olasılığı 0,25








