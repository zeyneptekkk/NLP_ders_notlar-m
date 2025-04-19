from sklearn.feature_extraction.text import CountVectorizer

documents=["kedi bahçede","kedi evde"]

vectorizer=CountVectorizer()

# metni sayısal vektorlere cevir
x=vectorizer.fit_transform(documents)

feature_names=vectorizer.get_feature_names_out()

#vektör temsili
vektor_temsili=x.toarray()