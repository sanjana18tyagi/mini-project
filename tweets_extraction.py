import tweepy
	import csv
	import pandas as pd
	####input your credentials here
	  consumer_key ='249QYJU8UqoS364VqIOoWmpJQ'
	  consumer_secret = '2hPGevE7tzMsysrWqV2eWp5YNmzgybMSu8y07YIM2GI7FE21DR'
	  access_token = '969807986498600960-6lkHrqBOzbZxoO30jcM2yZqaMiWsoay'
	  access_token_secret = 'Vu3POhc2yRk3Gka2efVSbFYMTVzXhtuwHLREoZH4zHzcE'
	    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	    auth.set_access_token(access_token, access_token_secret)
	    api = tweepy.API(auth,wait_on_rate_limit=True)
	# Open/Create a file to append data
	csvFile = open('2.csv', 'a')
	#Use csv Writer
	      csvWriter = csv.writer(csvFile)
	      for tweet in tweepy.Cursor(api.search,q="#fuckyou",count=100,
	                           lang="en",
	                           since="2017-04-03").items():
	    print (tweet.created_at, tweet.text)
	    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

