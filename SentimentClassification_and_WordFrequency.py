# %%

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
import string
nltk.download("stopwords")

model = TFDistilBertForSequenceClassification.from_pretrained(
    "/saved_model")

# get the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# %%
# tokenize input text
input_ids = tokenizer(
    'This is my my dog. I have some cute adorable pictures of my dog. Dog has a cute smile. I love my dog', return_tensors='tf')

# get tokens as a list of words
tokens = tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0])

# predict the label
preds = model(input_ids)

# %%
# show the prediction result
sentiment = model.config.id2label[preds[0][0].numpy().argmax()]

# %%

# filter out the only words that are not stopwords, punctuation, or numbers, cls token, and pad token
stwrds = stopwords.words('english')

# remove cls and sep tokens
filtered_words = [word for word in tokens if word not in [
    '[CLS]', '[SEP]', stwrds, string.punctuation, string.digits, '.']]

# plot frequency distribution of words with frequency greater than 1
freq = nltk.FreqDist(filtered_words)

# filter set where value is more than 1
new_set = [(sub, val) for sub, val in freq.items() if val > 1]

top = freq.most_common(4)


# %%
# bold the words that are most common in the original text
for sub, val in new_set:
    tokens = [word if word != sub else '**' + word + '**' for word in tokens]

# final ouptut
new_text = tokenizer.convert_tokens_to_string(tokens)

# filter cls and sep tokens
new_text = new_text.replace('[CLS]', '').replace('[SEP]', '')

# %%
print("**********************************\n")
print("Sentiment of sentence is: ", sentiment)
print("\n")
print("The text with most frequent words bolded is: \n")
print(new_text)

print("\n")
print("The top 4 most frequent words are: \n")
print(top)
