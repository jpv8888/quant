# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:36:53 2022

@author: jpv88
"""
import praw

reddit = praw.Reddit(
    client_id="Rgt3x9LD1OZAtdIDdymV1g",
    client_secret="-OjkcgS9mgAvZEwoBlKOjJGzpSltNQ",
    user_agent="penny stock enjoyer bot v1.0 by /u/pennystockenjoyer",
)

# %% 

test = []
for submission in reddit.subreddit("pennystocks").hot(limit=10):
    test.append(submission.title)
    

    
# %%

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

mail_content = "\n".join(test)

#The mail addresses and password
sender_address = 'pennystockenjoyer@gmail.com'
sender_pass = '3"8DPBuZV/Jb$f:*'
receiver_address = 'matt.wallace17@gmail.com'
#Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'top 10 posts in r/pennystocks hot right now'   #The subject line
#Create SMTP session for sending the mail

#The body and the attachments for the mail
message.attach(MIMEText(mail_content, 'plain'))
session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
session.starttls() #enable security
session.login(sender_address, sender_pass) #login with mail_id and password
text = message.as_string()
session.sendmail(sender_address, receiver_address, text)
session.quit()
print('Mail Sent')
    