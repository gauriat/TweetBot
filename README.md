# 🤖 TweetBot – Sentiment Analysis, Tweet Generation, and Reply Suggestion

TweetBot is an AI-powered web application that analyzes tweet sentiment, generates new tweets, and suggests tweet replies based on the provided context.  
It leverages cutting-edge transformer models like **BERT**, **GPT-2**, and **BART**, all integrated into a simple Flask web interface.

---

## ✨ Features

### 1️⃣ Sentiment Analysis using BERT
- Input a tweet → get sentiment: **Positive**, **Negative**, or **Neutral**
- Model: Pretrained **BERT** fine-tuned for sentiment classification

### 2️⃣ Tweet Generation using GPT-2
- Enter a **topic** and desired **sentiment**
- Generates a tweet aligned with the given context and emotion
- Model: **GPT-2** fine-tuned on Twitter data

### 3️⃣ Tweet Reply Generation using BART
- Enter a tweet → generate a smart, context-aware reply
- Trained using a dataset of real tweets and replies
- Model: Custom fine-tuned **BART**

---

## 🧠 Models Used

| Feature               | Model    | Format           |
|-----------------------|----------|------------------|
| Sentiment Analysis    | BERT     | `.pt`            |
| Tweet Generation      | GPT-2    | Hugging Face JSON config & tokenizer |
| Reply Generation      | BART     | `.pt`, `.safetensors`, tokenizer config |


---



## ⚙️ Technologies Used

- **Python 3.x**
- **Flask** – for the web server
- **Hugging Face Transformers** – for all model implementations
- **Torch** – for deep learning backend
- **HTML / Jinja2** – for frontend

---

## 🚧 Future Enhancements

- ✅ Add Hugging Face Space version for deployment  
- ✅ Improve tweet quality with more recent datasets  
- ☑️ Add multilingual tweet generation  
- ☑️ Add sarcasm or hate speech detection filter  
- ☑️ UI/UX improvements

---

## 📫 Contact

- **Gauri Thambkar**  
  [LinkedIn](https://www.linkedin.com/in/gauri-thambkar-6b616126a/)
  [Email](mailto:gaurithambkar@gmail.com)  
- GitHub: [github.com/gauriat]((https://github.com/gauriat))

---

🚀 *AI that understands, writes, and responds like a human on Twitter.*
