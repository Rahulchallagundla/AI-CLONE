# 1. Prerequisites
- **Python Installed:** Ensure Python 3.8+ is installed.
  
- **Install Libraries:** Install the required Python libraries using pip:

  ```bash

  pip install torch transformers textblob tweepy requests beautifulsoup4 python-dotenv

  ```

- **API Credentials:** Obtain your **Twitter API** credentials by setting up a developer account at [Twitter Developer Portal](https://developer.twitter.com/).

---

### **2. Setting Up the `.env` File**
1. Create a `.env` file in your project directory. Example path:

   ```

   C:\Users\chall\OneDrive\Desktop\.env
   ```

2. Add your Twitter API keys like this:

   ```

   TWITTER_API_KEY=your_twitter_api_key
   TWITTER_API_SECRET_KEY=your_twitter_api_secret_key
   TWITTER_ACCESS_TOKEN=your_access_token
   TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
   TWITTER_BEARER_TOKEN=your_bearer_token
   ```

---

### **3. Project Folder Structure**
Your project directory should look like this:
```
project-folder/
│
├── ai_clone.py          # Your main script
├── .env                 # Your environment variables
├── jack_blog_content.csv (optional, auto-generated)
└── requirements.txt     # Optional list of dependencies
```

---

### **4. Running the Project**
1. Open a terminal or command prompt in your project directory.
2. Run the script:
   ```bash
   python ai_clone.py
   ```
3. Output files (`jack_blog_content.csv` and `jack_blog_with_sentiment.csv`) will be generated.
4. Fine-tuned model will be saved in the `./jack_gpt2` directory.
5. Generated tweets will be posted to Twitter if all credentials are valid.
