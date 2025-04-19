
import nltk #natural language toolkit

nltk.download("punkt") #metni kelime ve cümle bazında tokenlaraayırabilmek için gerekli

text="Hello ,World!  How are you? Hello,hii..."
#kelime tokenizasyounu: word_tokenize: metni kelimelere ayırır,noktalama isaretlei ve 
#bosluklar ayrı birer token olarak elde edilecektir

word_tokens=nltk.word_tokenize(text)

#cumle tokenizasyonu : sent_tokenize: metni cümlelere ayırır

sentence_tokens=nltk.sent_tokenize(text)