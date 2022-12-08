from auth import *
import csv

#go through iSarcasm dataset, download all tweets referenced there, then write to file

sarcasm_list = []
non_sarcasm_list = []

with open("iSarcasm/isarcasm_train.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        tweet_id = row[0]
        classification = row[1]
        subclass = row[2]
        try:
            tweet = api.get_status(id=tweet_id)
        
            #remove line breaks
            text = tweet.text.replace('\n', ' ').replace('\r', '')
        
            #print(tweet_id)
            if classification == "sarcastic":
                sarcasm_list += [text]
            else:
                non_sarcasm_list += [text]
                
            #print(classification)
                
            #print(tweet.text)
            if len(sarcasm_list) + len(non_sarcasm_list) % 100 == 0:
                print(f"Number of fetched tweets: {len(sarcasm_list) + len(non_sarcasm_list)}")
            #print(sarcasm_list)
            #print(non_sarcasm_list)
        except:
            print(f"Error: Couldn't fetch tweet with ID {tweet_id} (tweet probably doesn't exist anymore)")
            continue
        

with open("iSarcasm/isarcasm_train_sarcasm.txt", "w") as file:
    for elem in sarcasm_list:
        try:
            file.write(elem)
        except:
            continue

with open("iSarcasm/isarcasm_train_non_sarcasm.txt", "w") as file:
    for elem in non_sarcasm_list:
        try:
            file.write(elem)
        except:
            continue


#tweet_id = 1090351571395899392
#tweet = api.get_status(id=tweet_id)
#print(tweet)