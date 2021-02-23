# Twitter-roBERTa-base for Emoji prediction

This is a roBERTa-base model trained on ~58M tweets and finetuned for emoji prediction with the TweetEval benchmark.

- Paper: [_TweetEval_ benchmark (Findings of EMNLP 2020)](https://arxiv.org/pdf/2010.12421.pdf). 
- Git Repo: [Tweeteval official repository](https://github.com/cardiffnlp/tweeteval).

## Example of classification

```python
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='emoji'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

text = "Looking forward to Christmas"
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

# # TF
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

# text = "Looking forward to Christmas"
# text = preprocess(text)
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# scores = output[0][0].numpy()
# scores = softmax(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")

```

Output: 

```
1) 🎄 0.5457
2) 😊 0.1417
3) 😁 0.0649
4) 😍 0.0395
5) ❤️ 0.03
6) 😜 0.028
7) ✨ 0.0263
8) 😉 0.0237
9) 😂 0.0177
10) 😎 0.0166
11) 😘 0.0143
12) 💕 0.014
13) 💙 0.0076
14) 💜 0.0068
15) 🔥 0.0065
16) 💯 0.004
17) 🇺🇸 0.0037
18) 📷 0.0034
19) ☀ 0.0033
20) 📸 0.0021
```
