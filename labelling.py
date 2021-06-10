import pandas
import numpy as np
import re 

%cd /content/ds.csv/
tweets = pandas.read_csv('train.csv', header = None)
tweets['label'] = 1
print(tweets)
