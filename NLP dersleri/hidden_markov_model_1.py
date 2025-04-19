"""
Part Of Speech POS:kelimeleri uygun sözcuk türünü bulma çabası
HMM

I (zamir) am a teacher(isim)

"""

import nltk
from nltk.tag import hmm

#ornek training data tanımla
train_data=[
    [("I","PRP"),("am","VBP"),("a","DT"),("teacher","NN")],
    [("You","PRP"),("are","VBP"),("a","DT"),("student","NN")]
    ]

# train HMM 
trainer=hmm.HiddenMarkovModelTrainer()
hmm_tagger=trainer.train(train_data)

#yeni bir cümle olustur ve cumlenin icerisindde bulunan her bir sozcugun türünü etiketle
test_sentence="I am a student".split()
tags=hmm_tagger.tag(test_sentence)
print(f"Yeni Cumle: {tags}")
