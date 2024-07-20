from warnings import filterwarnings
filterwarnings('ignore')

# API Bağlantısının Yapılması
#!pip install tweepy

import tweepy, codecs


consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

api.update_status("Hello from Python for Udemy Data Science and Machine Learning Course")

# Twitter'dan Veri Çekmek
mvk = api.me()
mvk.screen_name
mvk.followers_count
mvk.friends
for friend in mvk.friends(count = 20):
    print(friend.screen_name)
dir(mvk)

#kullanıcı temel bilgileri
user = api.get_user(id = "murat_aksit")
dir(user)
user.screen_name
user.followers_count
dir(user)
user.profile_image_url

# hometimeline
public_tweets = api.home_timeline(count = 10)
for tweet in public_tweets:
    print(tweet.text)

#user timeline
name = "murat_aksit"
tweet_count = 10

user_timetime = api.user_timeline(id = name, count = tweet_count)

for i in user_timetime:
    print(i.text)

#retweet edilen tweetler
retweets = api.retweets_of_me(count = 3)
for retweet in retweets:
    print(retweet.text)

#hashtag
results = api.search(q = "#pazartesi", 
                     lang = "tr", 
                     result_type = "recent", 
                     count = 100)
for retweet in results:
    print(retweet.text)

#dataframe cevirmek
def tweets_df(results):
    id_list = [tweet.id for tweet  in results]
    import pandas as pd
    data_set = pd.DataFrame(id_list, columns = ["id"])
    
    data_set["text"] = [tweet.text for tweet in results]
    data_set["created_at"] = [tweet.created_at for tweet in results]
    data_set["retweet_count"] = [tweet.retweet_count for tweet in results]
    data_set["user_screen_name"] = [tweet.author.screen_name for tweet in results]
    data_set["user_followers_count"] = [tweet.author.followers_count for tweet in results]
    data_set["user_location"] = [tweet.author.location for tweet in results]
    data_set["Hashtags"] = [tweet.entities.get('hashtags') for tweet in results]
    
    return data_set

data = tweets_df(results)
data["text"]
data.to_csv("data_twitter.csv")

# Profil Analizi
#temel bilgiler
mvk = api.get_user("mvahitkeskin")
mvk.name
mvk.id
mvk.url
mvk.verified
mvk.screen_name
mvk.location
mvk.statuses_count
mvk.followers_count
mvk.favourites_count
mvk.friends_count

tweets = api.user_timeline(id = "mvahitkeskin")
for i in tweets:
    print(i.text)

def timeline_df(tweets):
    idler = [tweet.id for tweet  in tweets]
    import pandas as pd
    df = pd.DataFrame(idler, columns = ["id"])
    
    df["created_at"] = [tweet.created_at for tweet in tweets]
    df["text"] = [tweet.text for tweet in tweets]
    df["favorite_count"] = [tweet.favorite_count for tweet in tweets]
    df["retweet_count"] = [tweet.retweet_count for tweet in tweets]
    df["source"] = [tweet.source for kisi in tweets]
    
    return df


timeline_df(tweets)

def  timeline_df(tweets):
    import pandas as pd
    df = pd.DataFrame()
    df['id'] = list(map(lambda tweet: tweet.id, tweets))
    df['created_at'] = list(map(lambda tweet: tweet.created_at, tweets))
    df['text'] = list(map(lambda tweet: tweet.text, tweets))
    df['favorite_count'] = list(map(lambda tweet: tweet.favorite_count, tweets))
    df['retweet_count'] = list(map(lambda tweet: tweet.retweet_count, tweets))
    df['source'] = list(map(lambda tweet: tweet.source, tweets))
    return df

tweets = api.user_timeline(id = "fatihportakal", count = 200)
df = timeline_df(tweets)
df.shape
df.head()

#profil enleri
df.sort_values("favorite_count", ascending = False).head()
df.sort_values("retweet_count", ascending = False)[["text","retweet_count"]].iloc[0:3]
df.sort_values("favorite_count", ascending = False)[["text","favorite_count"]].iloc[0:3]

## Retweet & Favori Sayılarının Dağılımlarının Çıkarılması
df.head()
#%config InlineBackend.figure_format = 'retina'
import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(df.favorite_count, kde = False, color = "blue")
sns.distplot(df.retweet_count, color = "blue");
plt.xlim(-100, 5000)

## Tweet-Saat Dağılımı
df.head()
df["tweet_saati"] = df["created_at"].apply(lambda x: x.strftime("%H"))

import pandas as pd
df["tweet_saati"] = pd.to_numeric(df["tweet_saati"])
df.info()
sns.distplot(df["tweet_saati"], kde = False, color ="blue")
df["gunler"] = df["created_at"].dt.weekday_name
df.head()

df.groupby("gunler").count()["id"]
gun_freq = df.groupby("gunler").count()["id"]
gun_freq.plot.bar(x = "gunler", y = "id")


## Tweet Atma Kaynaklarının Betimlenmesi

kaynak_freq = df.groupby("source").count()["id"]
kaynak_freq.plot.bar(x = "source", y = "id")
df.groupby("source").count()["id"]
df.groupby(["source","tweet_saati","gunler"])[["tweet_saati"]].count()


## Takipçi ve Arkadaşların Analizi
user = api.get_user(id = "fatihportakal")
for friend in user.friends():
    print(friend.screen_name)

friends = user.friends()
followers = user.followers()

def followers_df(takipci):
    import pandas as pd
    idler = [kisi.id for kisi  in takipci]
    df = pd.DataFrame(idler, columns = ["id"])
    
    df["created_at"] = [kisi.created_at for kisi in takipci]
    df["screen_name"] = [kisi.screen_name for kisi in takipci]
    df["location"] = [kisi.location for kisi in takipci]
    df["followers_count"] = [kisi.followers_count for kisi in takipci]
    df["statuses_count"] = [kisi.statuses_count for kisi in takipci]
    df["friends_count"] = [kisi.friends_count for kisi in takipci]
    df["favourites_count"] = [kisi.favourites_count for kisi in takipci]
    
    return df


df = followers_df(followers)
df.head()

## Takipçi Segmentasyonu
df.index = df["screen_name"]
s_data = df[["followers_count", "statuses_count"]]
s_data.info()

s_data["followers_count"] = s_data["followers_count"] + 0.01
s_data["statuses_count"] = s_data["statuses_count"] + 0.01

s_data = s_data.apply(lambda x: (x-min(x))/(max(x)-min(x)))

s_data["followers_count"] = s_data["followers_count"] + 0.01
s_data["statuses_count"] = s_data["statuses_count"] + 0.01

s_data.head()

skor = s_data["followers_count"] * s_data["statuses_count"]
skor.sort_values(ascending = False)
skor[skor > skor.median() + skor.std()/len(skor)].sort_values(ascending = False)
skor.std()
skor.median()
s_data["skor"] = skor


import numpy as np
s_data["segment"] = np.where(s_data["skor"] >= skor.median() + 
                             skor.std()/len(skor), "A","B")

# Hashtag Analizi
api.trends_available()
def ulke_kodlari():
    places = api.trends_available()
    all_woeids = {place['name'].lower(): place['woeid'] for place in places}
    return all_woeids
ulke_kodlari()

def ulke_woeid(ulke_adi):
    ulke_adi = ulke_adi.lower()
    trends = api.trends_available()
    all_woeids = ulke_kodlari()
    return all_woeids[ulke_adi]

ulke_woeid("turkey")

trendler = api.trends_place(id = 23424969)

import json
print(json.dumps(trendler, indent = 4))

turkiye = api.trends_place(id = 23424969)
trendler = turkiye[0]["trends"]

for i in trendler:
    print(i["name"])

## Hashtag'den Veri Çekmek
tweetler = api.search(q = "#pazartesi", 
                      lang = "tr", 
                      result_type = "recent", 
                     count = 1000)

def hashtag_df(tweetler):
    import pandas as pd
    id_list = [tweet.id for tweet  in tweetler]
    df = pd.DataFrame(id_list, columns = ["id"])
    
    df["text"] = [tweet.text for tweet in tweetler]
    df["created_at"] = [tweet.created_at for tweet in tweetler]
    df["retweeted"] = [tweet.retweeted for tweet in tweetler]
    df["retweet_count"] = [tweet.retweet_count for tweet in tweetler]
    df["user_screen_name"] = [tweet.author.screen_name for tweet in tweetler]
    df["user_followers_count"] = [tweet.author.followers_count for tweet in tweetler]
    df["user_location"] = [tweet.author.location for tweet in tweetler]
    df["Hashtags"] = [tweet.entities.get('hashtags') for tweet in tweetler]
    
    return df

df = hashtag_df(tweetler)
df.shape
df.head()

## Hashtag Betimlemek
#essiz katılımcı
df.head()
df["user_screen_name"].unique().size
df.groupby("user_screen_name")["id"].count().sum()

#en cok katkı sağlayanlar
df.groupby("user_screen_name").count()["id"].sort_values(ascending = False)[0:5]

#ozgun katkı
df.head()
df["text"].str.startswith("RT").head(10)
df[~df["text"].str.startswith("RT")].count()["id"] / len(df)
df[~df["text"].str.startswith("RT")].count()["id"] / df[df["text"].str.startswith("RT")].count()["id"]


## en cok fav-retweet
df.sort_values("retweet_count", ascending = False).head()

## Tweet-Saat Dağılımı
## Tweet Atma Kaynaklarının Betimlenmesi
kaynak_freq = df.groupby("source").count()["id"]
kaynak_freq.plot.bar(x = "source", y = "id")
df.groupby("source").count()["id"]

# Twitter Text Mining
df = hashtag_df(tweetler)

#buyuk-kucuk donusumu
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#noktalama işaretleri
df['text'] = df['text'].str.replace('[^\w\s]','')

#sayılar
df['text'] = df['text'].str.replace('\d','')

#stopwords
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#lemmi
from textblob import Word
#nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

#noktalama işaretleri
df['text'] = df['text'].str.replace('rt','')
df["text"]
freq_df = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis =0).reset_index()
freq_df.columns = ["kelimeler","frekanslar"]
freq_df.shape
freq_df[freq_df.frekanslar > freq_df.frekanslar.mean() + 
        freq_df.frekanslar.std()]
a = freq_df[freq_df.frekanslar > freq_df.frekanslar.mean() + 
        freq_df.frekanslar.std()]
a.plot.bar(x = "kelimeler",y = "frekanslar")

## Word cloud
#!pip install wordcloud
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

text = " ".join(i for i in df.text)
wordcloud = WordCloud(background_color = "white").generate(text)
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

vbo_mask = np.array(Image.open("VBO.jpg"))
wc = WordCloud(background_color = "white", 
               max_words = 1000, 
               mask = vbo_mask,
              contour_width = 3, 
              contour_color = "firebrick")

wc.generate(text)
plt.figure(figsize=(10,10))
plt.imshow(wc, interpolation = "bilinear")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

# Twitter Sentiment Analizi
from textblob import TextBlob

def sentiment_skorla(df):

    text = df["text"]

    for i in range(0,len(text)):
        textB = TextBlob(text[i])
        sentiment_skoru = textB.sentiment.polarity
        df.set_value(i, 'sentiment_skoru', sentiment_skoru)
        
        if sentiment_skoru <0.00:
            duygu_sinifi = 'Negatif'
            df.set_value(i, 'duygu_sinifi', duygu_sinifi )

        elif sentiment_skoru >0.00:
            duygu_sinifi = 'Pozitif'
            df.set_value(i, 'duygu_sinifi', duygu_sinifi )

        else:
            duygu_sinifi = 'Notr'
            df.set_value(i, 'duygu_sinifi', duygu_sinifi )
            
    return df 

df.head()
sentiment_skorla(df)
df.groupby("duygu_sinifi").count()["id"]
duygu_freq = df.groupby("duygu_sinifi").count()["id"]
duygu_freq.plot.bar(x = "duygu_sinifi",y = "id")
tweetler = api.search(q = "#apple", lang = "en")
df = hashtag_df(tweetler)

#buyuk-kucuk donusumu
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#noktalama işaretleri
df['text'] = df['text'].str.replace('[^\w\s]','')

#sayılar
df['text'] = df['text'].str.replace('\d','')

#stopwords
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#lemmi
from textblob import Word
#nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

#noktalama işaretleri
df['text'] = df['text'].str.replace('rt','')

sentiment_skorla(df)

duygu_freq = df.groupby("duygu_sinifi").count()["id"]
duygu_freq.plot.bar(x = "duygu_sinifi",y = "id")




















