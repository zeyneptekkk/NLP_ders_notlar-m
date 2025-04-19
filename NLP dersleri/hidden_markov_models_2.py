import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000 #conll2000 corpus'u, POS (Part of Speech) etiketleme ve chunking (kelime gruplarını belirleme) gibi doğal dil işleme görevleri için yaygın olarak kullanılan bir veri setidir. Bu corpus, eğitim ve test verileri içerir ve özellikle POS etiketleme modellerini eğitmek için kullanılabilir.

#gerkli veri setini içeriye aktar
nltk.download("conll2000")

train_data=conll2000.tagged_sents("train.txt")
test_data=conll2000.tagged_sents("test.txt")

#train hmm
trainer=hmm.HiddenMarkovModelTrainer()
hmm_tagger=trainer.train(train_data)

#yeni cumle ve test
test_sentence="I like going to school".split()
tags=hmm_tagger.tag(test_sentence)
