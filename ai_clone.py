import os
import requests
import torch
import random
import pandas as pd
import time
from bs4 import BeautifulSoup
from textblob import TextBlob
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import tweepy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twitter API credentials
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET_KEY = os.getenv("TWITTER_API_SECRET_KEY")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Authenticate with Twitter
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET_KEY,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
)

# Step 1: Scrape Blog Content
def scrape_blog(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    posts = soup.find_all('article')  # Adjust the selector based on actual blog structure
    texts = [post.get_text(strip=True) for post in posts]
    return texts

jack_blog_url = "https://jackjay.io/"
jack_blog_content = scrape_blog(jack_blog_url)

# Save scraped content
df_blog = pd.DataFrame(jack_blog_content, columns=['text'])
df_blog.to_csv('jack_blog_content.csv', index=False)
print("Blog content saved to jack_blog_content.csv")

# Step 2: Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df_blog['sentiment'] = df_blog['text'].apply(get_sentiment)
df_blog.to_csv('jack_blog_with_sentiment.csv', index=False)
print("Sentiment analysis added and saved.")

# Step 3: Fine-tune GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

train_texts = df_blog['text'].tolist()
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

class BlogDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Split the dataset into training and evaluation sets
train_size = int(0.8 * len(train_texts))
eval_texts = train_texts[train_size:]
train_texts = train_texts[:train_size]

# Tokenize the evaluation texts
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=512)
eval_dataset = BlogDataset(eval_encodings)

# Update training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="steps",  # Evaluate during training
    eval_steps=500,  # Evaluation frequency
    logging_dir='./logs',
)

# Pass eval_dataset to Trainer
# Tokenize the training texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
train_dataset = BlogDataset(train_encodings)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Correct train_dataset
    eval_dataset=eval_dataset,   # Correct eval_dataset
)



trainer.train()
model.save_pretrained('./jack_gpt2')
print("Model fine-tuned and saved.")

# Step 4: Generate and Post Tweets
model = GPT2LMHeadModel.from_pretrained('./jack_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def preprocess_blog_content(file_path):
    df = pd.read_csv(file_path)
    sentences = []
    for text in df['text']:
        for sentence in text.split('.'):
            sentence = sentence.strip()
            if 10 < len(sentence) <= 120:
                sentences.append(sentence)
    return sentences

def generate_tweet(prompt, num_tweets=1):
    generated_tweets = []
    for _ in range(num_tweets):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        output = model.generate(
            input_ids,
            max_length=60,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
        )
        tweet = tokenizer.decode(output[0], skip_special_tokens=True).split('.')[0].strip() + '.'
        if 10 < len(tweet) <= 280:
            generated_tweets.append(tweet)
    return generated_tweets

blog_sentences = preprocess_blog_content('jack_blog_content.csv')

posted_tweets = set()
retry_attempts = 0

for prompt in blog_sentences:
    try:
        generated_tweets = generate_tweet(prompt, num_tweets=1)
        for tweet in generated_tweets:
            if tweet in posted_tweets:
                continue
            response = client.create_tweet(text=tweet)
            print(f"Posted tweet: {tweet}")
            posted_tweets.add(tweet)
            time.sleep(2)
    except tweepy.errors.TooManyRequests:
        retry_attempts += 1
        print(f"Rate limit exceeded. Waiting for 15 minutes (Attempt {retry_attempts})...")
        time.sleep(15 * 60)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
