 
from tokenizer import split_into_sentences
 
import itertools
import re 
import string
import nltk
import xlsxwriter
from nltk.corpus import stopwords 
import pandas as pd
import csv  
import json
import GetOldTweets3 as got
import pandas as pd

from flask import Flask, render_template,request

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tag import pos_tag
import matplotlib.pyplot as plt

import twitter
class fetch_tweets: 
   def parseurl(url):
       data=url.split('/')
       return data 
   def get_length(data):
      length=len(data)
      return data[length-1]   
def SearchTweetThroughUrl(url1,url2,searchtag1,searchtag2): 
   
   print("wait while collecting the data set----------")
   tweetCriteria = got.manager.TweetCriteria().setQuerySearch(searchtag1).setUsername(url1).setTopTweets(True).setSince("2019-11-01")
   tweets = got.manager.TweetManager.getTweets(tweetCriteria)
   text_tweets = [tweet.text for tweet in tweets] 
   print("completed")
#   for t in text_tweets:
#     print(t)
   dfn = pd.DataFrame(text_tweets)
   dfn.to_csv('file1.csv')
   print("wait while collecting the data set2----------")
   tweetCriteria = got.manager.TweetCriteria().setQuerySearch(searchtag2).setUsername(url2).setTopTweets(True).setSince("2019-11-01")
   tweets = got.manager.TweetManager.getTweets(tweetCriteria)
   text_tweets = [tweet.text for tweet in tweets]
   print("completed")
#   for t in text_tweets:
#     print(t)
   dfp = pd.DataFrame(text_tweets)
   dfp.to_csv('file2.csv')
def normalized1(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
       

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens 
def tokenization(tweet):
    
    #TOKENIZATION
    #print("fatima")
    g = split_into_sentences(tweet)
# Loop through the sentences
    for sentence in g:
    # Obtain the individual token strings
        tokens = sentence.split()
        filtered_sentence = [w for w in tokens if not w in stopwords.words('english')]
       # print(tokens)
   #  Print the tokens, comma-separated
#        print(", ".join(tokens))
     # store the tokens in a list
        thisList = filtered_sentence
        yield thisList  
def wordlemitizer(x):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(x):
         if tag.startswith('NN'):
            pos = 'n'
         elif tag.startswith('VB'):
            pos = 'v'
         else:
            pos = 'a'
    lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence        
def preprocessing(tweet):
        #PREPROCESSING
    #tweet = "One thing https://twitter.com/pid_gov/status/1209148676741967874 helloooo shocked FARWA me in #IPLAuction #DelhiDaredevils buy! #ChrisMorris ABSOLUTELY forrrr Rs 1100 Lakh's And #SunrisersHyderabad takes #Yusufpathan for Rs 190 Lakh's @MazherArshad @bhogleharsha #ipl #IPL2018 #IPL2018Auction #IPLAuction2018"
    # remove old style retweet text "RT"
  #  stop_words = list(set(stopwords.words('english')))
  #  allowed_word_types = ["J","R","V"]
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    #remove numbers
    tweet = re.sub('\d', '', tweet)
    # convert text to lower-case
    tweet = tweet.lower()

    # remove repeating characters
   # tweet = ''.join(c[0] for c in itertools.groupby(tweet))
    #print(tweet)

    # remove URLs
    tweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', tweet)
    #tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', tweet)
    #tweet = re.sub(r':.*$', ":", tweet)
    
    # remove the # in #hashtag
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # remove usernames
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  
    #remove email addreses
    
    tweet = re.sub('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', tweet)

    # remove punctuations like quote, exclamation sign
    tweet = re.sub(r'['+string.punctuation+']+', ' ', tweet)

    yield tweet
    #print(tweet) 

def tweet_url(t):
    return "https://twitter.com/%s/status/%s" % (t.user.screen_name, t.id)

def get_tweets(filename):
    for line in open(filename):
        #print(twitter.Status.NewFromJsonDict(json.loads(line)))
        yield twitter.Status.NewFromJsonDict(json.loads(line))
def sentimentCount(t,n):
    c=0
    nether=0
    stps=""
    for item1 in t:
        if item1 == 'neither' or item1 == 'nor' :
           nether=-1
    for item1 in range (len(t)): 
  # print("word") 
   #print(item1) 
      for item2 in range (len(n)):
    #   print(item2)
        if t[item1] == n[item2] and t[item1]!='a' and t[item1]!='t':
            if t[item1-1] != 'not' and nether != -1 :
               c=c+1
               stps=stps+","+str(t[item1])
             #  print(t[item1])
    return c,stps  
app = Flask(__name__)
     
list1=[]  
df = pd.read_csv('file1.csv', delimiter=',')
df2= pd.read_csv('file2.csv', delimiter=',')
tw1=[]
tw2=[]
dataList = [] #empty list
for row in df.itertuples(): 
    mylist = [row[2]]
    tw1.append(mylist)
dataList2 = [] #empty list
for row in df2.itertuples(): 
    mylist = [row[2]]
    tw2.append(mylist)
#pdToList = list(df['reviews'])
#for row in tw1:
#   print(row)  
re1=[]
ourtweets1=[]
recsv1=[]
normal=[]
for i in range(len(tw1)):
    custom_tokens = normalized1(word_tokenize(str(tw1.pop())))
    normal.append(custom_tokens)
normal.reverse()

normal1=[]
for i in range(len(tw2)):
    custom_tokens = normalized1(word_tokenize(str(tw2.pop())))
    normal1.append(custom_tokens)
normal1.reverse()
for x, word in enumerate(normal):  
    
    word2=str(word)
    r=preprocessing(word2) 
     #print(r)
    
    #print(r) 
    r1=tokenization(r)
     
   # print(r1)
    re1.append(r1)
for index,words in enumerate(normal):
    word2=str(words)
    s=preprocessing(word2)
    r1=tokenization(s)
    recsv1.append(r1)    
    ourtweets1.append(r1)
 
       
re2=[]
recsv2=[]    
for x, word in enumerate(normal1):      
    word2=str(word)
    r=preprocessing(word2)
    r1=tokenization(r)
    re2.append(r1)
for index,words in enumerate(normal1):
    word2=str(words)
    s=preprocessing(word2)
    r1=tokenization(s)
    recsv2.append(r1) 
    
#for x in re1:
#    print(x)
#    for x in re1:
#        print(x)
neg=[]
file1post1=[]       
file1post2=[]
with open("negative-words.txt", "r") as f:
    negText = f.read()
neg = tokenization(negText)
with open("positive-words.txt", "r") as f:
    posText = f.read()
pos=tokenization(posText)


for group in pos:
    stp=", ".join(group)
for g in neg:
    stn=", ".join(g)
p=stp.split(', ')  
n=stn.split(', ')
tweetsentiment=[]       
        #sentiment tweet1      
g1=[]   
tweetPCount1=0
tweetNCount1=0
tweetCount1=0
negwordlist=[]
poswordlist=[]
tokens1=[]       
     
for x, word in enumerate(re1):
    #print("tokenized:",x)
    spt1=0
    snt1=0
    r6=re1[x]
  #  print(r6)
    for x in r6:
     #   print("tokenized:",x)
        tokens1.append(x) 
        st1=", ".join(x) 
        t1=st1.split(', ')  
        for group in t1:
            g1=group
      #      print("string:",g1)
        negC1,negative_word=sentimentCount(t1,n)
        posC1,positive_word=sentimentCount(t1,p)
      #  print("neg Words",negC1)
      #  print("pos Words",posC1)
        spt1=spt1+posC1
        snt1=snt1+negC1
        poswordlist.append(positive_word)
        negwordlist.append(negative_word)
        file1post1.append(spt1)
        file1post2.append(snt1)
       # print("total positive words of reply of tweet1:",spt1)
        #print("total negative words of reply of tweet1:",snt1)
        if(spt1>snt1):
                tweetcheck1="positive"
                
                result6=tweetcheck1
                tweetPCount1=tweetPCount1+1
            #    print("positive reply")
        if((snt1==0 and spt1==0) or snt1==spt1):
          #      print("neutral")
                tweetcheck1="neutral"
                
               
                tweetCount1=tweetCount1+1
                result6=tweetcheck1
                
        if(snt1>spt1):                   
         #       print("negative reply")                   
                tweetcheck1="negative"
                result6=tweetcheck1
                tweetNCount1=tweetNCount1+1
        tweetsentiment.append(tweetcheck1)
dfs1 = pd.DataFrame(recsv1) 
dfs1['tweets']=tokens1  
dfs1['positivewordscount']=file1post1
dfs1['negativewordscount']=file1post2
dfs1['positive_words']=poswordlist
dfs1['negative_words']=negwordlist 
dfs1['over all sentiment']=tweetsentiment
dfs1.to_csv('tweet repective to pos and neg.csv')
  
g=[]   
tweetPCount=0
tweetNCount=0
tweetCount=0
files2post1=[]
files2post2=[]
negwordslist2=[]
poswordslist2=[]
tweetsentiment1=[]
tokens2=[]
for x, word in enumerate(re2):
    sp1=0
    sn1=0
    r5=re2[x]
    for x in r5:
               #print("tokenized:",x)
        tokens2.append(x)       
        st2=", ".join(x) 
        t2=st2.split(', ')  
        for group in t2:
            g=group 
        negC,negativeword1=sentimentCount(t2,n)
        posC,positiveword1=sentimentCount(t2,p)
       # print("neg Words",negC)
       # print("pos Words",posC)
        sp1=sp1+posC
        sn1=sn1+negC
        files2post1.append(sp1)
        files2post2.append(sn1)
        negwordslist2.append(negativeword1)
        poswordslist2.append(positiveword1)
       # print("total positive words of reply tweet2:",sp1)
        #print("total negative words of reply tweet2:",sn1)
        if(sp1>sn1):
                 tweetcheck="positive"
        #         print("positive reply")
                 result9=tweetcheck
                 tweetPCount=tweetPCount+1
        if((sn1==0 and sp1==0)or sp1==sn1):
                 tweetcheck="neutral"
                 result9=tweetcheck
                 tweetCount=tweetCount+1
         #        print("neutral")
                       
        if(sn1>sp1):
                tweetcheck="negative"
                result9=tweetcheck
          #      print("negative reply")
                tweetNCount=tweetNCount+1
        tweetsentiment1.append(tweetcheck)
dfs2 = pd.DataFrame(recsv2) 
dfs2['tweets']=tokens2  
dfs2['positive_words_count']=files2post1
dfs2['negative_words_count']=files2post2
dfs2['positive_words']=poswordslist2
dfs2['negative_words']=negwordslist2

dfs2['over all sentiment']=tweetsentiment1

dfs2.to_csv('tweet repective to pos and neg2.csv')
       
print("number of positive of tweets news house1",tweetPCount1)  
print("number of negative  tweets of news house1",tweetNCount1)
print("number of neutral of news house 1",tweetCount1)
print("number of positive  tweet of news house 2",tweetPCount)  
print("number of negative tweet of news house 2",tweetNCount) 
print("number of neutral tweet of news house 2:",tweetCount) 
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(["number of positive  tweets of news house1 ",str(tweetPCount1)])
    writer.writerow(["number of negative  tweets of news house1",str(tweetNCount1)])
    writer.writerow(["number of neutral  tweets of news house1",str(tweetCount1)])
    writer.writerow(["number of positive  tweets of news house2",str(tweetPCount)])
    writer.writerow(["number of negative  tweets of news house2",str(tweetNCount)])
    
    writer.writerow(["number of neutral  tweets of news house2",str(tweetCount)]) 
print("Analysis for News House UNIHX:")
labels = 'Positive', 'Negative', 'Neutral'
sizes = [tweetPCount, tweetNCount,tweetCount]
colors = ['yellowgreen', 'lightskyblue', 'lightcoral']
explode = (0.5, 0, 0)  
                   
# Plot1
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()
# Data to plot2
print("Analysis for News House WHO:")
labels1 = 'Positive', 'Negative', 'Neutral'
sizes1 = [tweetPCount1, tweetNCount1,tweetCount1]
colors1 = ['gold', 'red', 'magenta']
explode1 = (0.5, 0, 0)  

# Plot2
plt.pie(sizes1, explode=explode1, labels=labels1, colors=colors1,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()
totalcount1=tweetPCount1+tweetNCount1+tweetCount1
percent_positive=(tweetPCount1*100)/totalcount1
#print(percent_positive)
percent_negative=(tweetNCount1*100)/totalcount1
#print(percent_negative)
percent_neutral=(tweetCount1*100)/totalcount1
#print(percent_neutral)
totalcount=tweetPCount+tweetNCount+tweetCount
percent_positive1=(tweetPCount*100)/totalcount
#print(percent_positive1)
percent_negative1=(tweetNCount*100)/totalcount
#print(percent_negative1)
percent_neutral1=(tweetCount1*100)/totalcount1
#print(percent_neutral1)
dif=percent_positive-percent_negative
#print(dif)
if(dif>=10 and percent_positive+percent_negative>=50):
    print("this news house WHO spread more  positivity towards corona virus")
elif(dif<0 and dif<=(-10) and percent_positive+percent_negative>=50):
    print("this news house WHO spread more thrill towards corona virus")
elif(percent_neutral>=50):
    print("this news house WHO spread nuetral news towards corona virus")
else:
    print("this news house WHO spread nuetral news towards corona virus")
dif1=percent_positive1-percent_negative1
#print(dif1)

if(dif1>=10):
    print("this news house XHNEWS spread more  positivity towards corona virus")
elif(dif1<0 and dif1<=(-10)):
    print("this news house XHNEWS spread more thrill towards corona virus")
else:
    print("this news house XHNEWS spread nuetral news towards corona virus")
workbook = xlsxwriter.Workbook('chart_pie4.xlsx') 
worksheet = workbook.add_worksheet() 
bold = workbook.add_format({'bold': 1}) 

# Data to plot1
headings1 = ['Category', 'Values'] 

data1 = [ 
	['Positive', 'Negative', 'Neutral'], 
	[tweetPCount, tweetNCount,tweetCount], 
] 

worksheet.write_row('A1', headings1, bold) 
worksheet.write_column('A2', data1[0]) 
worksheet.write_column('B2', data1[1]) 
chart1 = workbook.add_chart({'type': 'pie'}) 
chart1.add_series({ 
	'name':	 'News House XHNEWS', 
	'categories': ['Sheet1', 1, 0, 3, 0], 
	'values':	 ['Sheet1', 1, 1, 3, 1], 
}) 
chart1.set_title({'name': 'Analysis for News House XHNEWS:'}) 
chart1.set_style(10) 
worksheet.insert_chart('C2', chart1, {'x_offset': 25, 'y_offset': 10}) 
                   
# Data to plot2
headings2 = ['Category', 'Values'] 

data2 = [ 
	['Positive', 'Negative', 'Neutral'], 
	[tweetPCount1, tweetNCount1,tweetCount1], 
] 
worksheet.write_row('A18', headings2, bold) 
worksheet.write_column('A19', data2[0]) 
worksheet.write_column('B19', data2[1]) 

chart2 = workbook.add_chart({'type': 'pie'}) 
chart2.add_series({ 
	'name':	 'News House WHO', 
	'categories': ['Sheet1', 18, 0, 20, 0], 
	'values':	 ['Sheet1', 18, 1, 20, 1], 
}) 
chart2.set_title({'name': 'Analysis for News House WHO:'}) 
chart2.set_style(10) 
worksheet.insert_chart('C18', chart2, {'x_offset': 25, 'y_offset': 10}) 
workbook.close() 

    
      
    
       

                   
