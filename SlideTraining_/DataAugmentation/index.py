from gensim.models import Word2Vec
from nltk.corpus import stopwords
import bs4 as bs
import urllib.request
import nltk
from nltk.stem import PorterStemmer
import re
ps = PorterStemmer()


def train_with_url(url):
    # nltk.download('punkt') # run this command for only the first time
    scrapped_data = urllib.request.urlopen(url)
    article = scrapped_data .read()
    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    return [p.text for p in paragraphs]


def train_with_txt(filename):
    return open(filename, "r")


paragraphs = train_with_txt("LOTR.txt")
# paragraphs = train_with_url(
# 'https://en.wikipedia.org/wiki/Artificial_intelligence')
article_text = ""
for p in paragraphs:
    article_text += p

# Cleaing the text
processed_article = article_text.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
processed_article = re.sub(r'\s+', ' ', processed_article)

# Preparing the dataset
all_sentences = nltk.sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words
for i in range(len(all_words)):
    all_words[i] = [ps.stem(w) for w in all_words[i]
                    if w not in stopwords.words('english')]


word2vec = Word2Vec(all_words, min_count=2, size=100)

vocabulary = word2vec.wv.vocab
word = 'call'
print(word2vec.wv.most_similar(ps.stem(word)))
