#stop words
stop_words = ['a', 'an', 'the', 'for', 'in', 'it', 'its', 'on', 'at', 'they', 'them',
                  'their', 'theirs', 'that', 'what', 'which', 'has', 'have', 'had',
                  'having', 'thus', 'do', 'no', 'nor', 'not', 'is', 'are', 'was',
                  'were', 'be', 'been', 'being', 'did', 'in', 'out', 'both', 'each',
                  'few', 'he', 'she', 'him', 'her', 'these', 'those', 'but', 'if',
                  'of', 'after', 'before', 'up', 'down', 'over', 'under',
                  'again', 'then', 'there', 'some', 'such', 'other',
                  'only', 'same', 'so', 'than', 'and', 'to']

#polarity, subjectivity, len
from textblob import TextBlob
Polarity=[]
Subjectivity=[]
countLen=[]


for i in processed_final2['NERtext']:
    x=TextBlob(i).sentiment
    Polarity.append(x[0])
    Subjectivity.append(x[1])
    countLen.append(len(i.split()))



#date features
dates_int=[(df['date'][i]-df['date'].min()).components[0] for i in range(len(df))]
dates_days=df['date'].apply(lambda x: x.dayofweek)

