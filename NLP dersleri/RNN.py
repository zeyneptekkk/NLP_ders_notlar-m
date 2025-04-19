'''
solve Classification problem(sentiment analysis in nlp) with RNN(Deep learning based language model)
duygu analizi -> bir cümlenin etiketlenmesi(positive ve negative)

restorant yorumlarını değerlendirme

'''

#import libraries
import numpy as np
import pandas as pd
from gensim.models import Word2Vec # metin temsili
from keras_preprocessing.sequence import pad_sequences
from keras.layers import SimpleRNN ,Dense,Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#create dataset

data = {
    "text": [
        "Yemek çok güzeldi",
        "Yemek çok pişmişti",
        "Servis harikaydı",
        "Garson çok kaba davrandı",
        "Tatlılar muazzamdı",
        "Çorba tuzsuzdu",
        "Mekan çok şıktı",
        "Koltuklar rahatsızdı",
        "Yemekler tam zamanında geldi",
        "Çok bekledik, sinirlendik",
        "Tat olarak çok iyiydi",
        "Lezzetsizdi, yiyemedim",
        "Garsonlar güler yüzlüydü",
        "Hizmet çok yavaştı",
        "Sunum harikaydı",
        "Yemekler soğuktu",
        "Porsiyonlar doyurucuydu",
        "Çok küçük porsiyonlardı",
        "Fiyatlar makuldü",
        "Aşırı pahalıydı",
        "Mekan çok kalabalıktı",
        "Sessiz ve huzurluydu",
        "Garson siparişi karıştırdı",
        "Tam istediğimiz gibi geldi",
        "Yemekler taze ve sıcaktı",
        "Salata bayattı",
        "Tatlı enfesti",
        "İçecekler ılık geldi",
        "Masamız kirliydi",
        "Tertemiz bir ortam vardı",
        "Rezervasyonumuz olmasına rağmen bekledik",
        "Müşteriyle çok ilgililerdi",
        "Hiç kimse ilgilenmedi",
        "Et tam kıvamındaydı",
        "Et lastik gibiydi",
        "Tatlılar taze ve lezizdi",
        "Çatal bıçaklar temiz değildi",
        "Servis hızlıydı",
        "Yemek çok tuzluydu",
        "Çok lezzetliydi, tekrar geleceğim",
        "Kötüydü, asla bir daha gelmem",
        "Sunum çok özensizdi",
        "Şef gerçekten harika iş çıkarmış",
        "Makarnanın sosu mükemmeldi",
        "Pilav kuru ve tatsızdı",
        "Hemen servis edildi",
        "Siparişimiz eksik geldi",
        "Menü çok çeşitliydi",
        "Hiçbir şey bulamadım yiyecek",
        "Yemekler ev yapımı gibiydi",
        "Hazır yemek gibiydi",
        "Çalışanlar çok anlayışlıydı",
        "İletişim kurmak zordu",
        "Her şey çok özenliydi",
        "İlgi alaka sıfırdı",
        "Yemek sonrası ikram harikaydı",
        "İkram yoktu, sormadan hesap getirdiler",
        "Çocuk dostu bir yerdi",
        "Çocuklar için uygun değildi",
        "Fiyat/performans olarak çok iyiydi",
        "Pahalı ama değmedi",
        "Çok memnun kaldım",
        "Hayal kırıklığına uğradım",
        "Pizzası efsaneydi",
        "Tadı yoktu",
        "Tatlılar porsiyon olarak çok büyüktü",
        "Porsiyonlar çok küçüktü",
        "Hızlı servis bizi çok mutlu etti",
        "Kahvaltı menüsü çok zayıftı",
        "Yemeklerde sevgi vardı adeta",
        "Tuzsuz ve yağsızdı, hiç tat alamadım",
        "Tatlıların sunumu şahaneydi",
        "Görsel olarak hiç tatmin edici değildi",
        "Rezervasyonumuz kaybolmuştu",
        "Bizi çok güzel ağırladılar",
        "Hemen ilgilendiler",
        "Dakikalarca bekledik",
        "Sıcak karşılandık",
        "Soğuk bir ortam vardı",
        "Yemek sonrası çay ikram ettiler",
        "İkram yoktu, hesap kabarıktı",
        "Atmosfer çok keyifliydi",
        "İçerisi havasızdı",
        "Tatlı menüsü çok zengindi",
        "Tatlıların hepsi tükenmişti",
        "Yemek sıcacık geldi",
        "Soğuk geldi, beğenmedim",
        "Garson çok ilgiliydi",
        "Garson bizi görmezden geldi",
        "Baharat kullanımı çok yerindeydi",
        "Baharatlar boğazımı yaktı",
        "Mekan oldukça temizdi",
        "Tuvaletler bile çok kirliydi",
        "Yemek hızlı ve eksiksiz geldi",
        "Sipariş yanlış geldi",
        "Etler lokum gibiydi",
        "Et kokuyordu",
        "Kahvesi mükemmeldi",
        "Kahve çok acıydı",
        "Hemen masa verdiler",
        "Yer yoktu, çok bekledik",
        "Garsonlar çok bilgiliydi",
        "Ne söyledik ne geldi anlamadık",
        "Mekanı çok beğendik",
        "Bir daha asla gitmem"
    ],
    "label": [
        "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "negatif", "pozitif", "negatif", "pozitif", "pozitif", "negatif", "pozitif", "negatif", "negatif", "pozitif",
        "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif",
        "negatif", "negatif", "pozitif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif",
        "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif",
        "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif",
        "negatif", "pozitif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif"
    ]
}


# %%metin temizleme ve preprocessing:tokenization ,padding,label encoding,train test split




#%%metin temsili:word embedding:word2vec


#%% modelling :build train ve test rnn modeli

