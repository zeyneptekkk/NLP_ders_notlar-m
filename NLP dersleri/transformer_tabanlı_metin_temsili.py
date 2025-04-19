from transformers import AutoTokenizer,AutoModel
import torch
#model ve tokenizer yukle
model_name="bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModel(model_name)

#input text(metni) tanımla
text="Transformers can be used for natural language preccesing."

#metni tokenlara çevirmek
inputs=tokenizer(text,return_tensors="pt") #çıktı pytorch tensoru olarak return edilir

#modeli kullanarak metin temsili oluştur
with torch.no_grad():  #gradyanların hesaplanmaası durdurulur ,böylece belleği daha verimli kullanır
    outputs=model(**inputs)
    
    
#modelin çıkısından son gizli durumu alalım
last_hidden_state=outputs.last_hidden_state #tum token çıktılarını almak içinprint
first_token_embedding=last_hidden_state[0,0,:].numpy()
print(f"metin Temsili: {first_token_embedding}")