import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer, BartTokenizer, BartForConditionalGeneration
from flask import Flask, request, jsonify, render_template
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the Sentiment Analysis Model class
class SentimentAnalysisModel(nn.Module):
    def __init__(self, num_classes):
        super(SentimentAnalysisModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# Load the label encoder
with open('label_encoder (1).pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the pre-trained sentiment analysis model's state dictionary
state_dict = torch.load('sentiment_analysis_model (1).pt', map_location='cpu')
num_classes = state_dict['fc.bias'].shape[0]

# Initialize the sentiment analysis model with the correct number of classes
sentiment_model = SentimentAnalysisModel(num_classes)
sentiment_model.load_state_dict(state_dict)
sentiment_model.eval()  # Set the model to evaluation mode

# Move model to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentiment_model.to(device)

# Initialize BERT tokenizer for sentiment analysis
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize GPT-2 model and tokenizer for tweet generation
gpt2_model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model.to(device)
gpt2_model.eval()

# Initialize BART tokenizer and model for reply generation
bart_model_name = './tweet_reply_model'
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)

# Ensure correct loading of the BART model
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
bart_model.load_state_dict(torch.load('tweet_reply_model.pt', map_location='cpu'))
bart_model.to(device)
bart_model.eval()

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    try:
        data = request.json
        text = data.get('text', '')

        inputs = bert_tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = sentiment_model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)

        sentiment_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        sentiment_label = 'positive' if sentiment_label == 1 else 'negative'
        
        return jsonify({'sentiment': sentiment_label})
    except Exception as e:
        logging.error(f"Error occurred in sentiment prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_tweet', methods=['POST'])
def generate_tweet():
    try:
        data = request.get_json()
        topic = data.get('topic', '')  # Safely get the 'topic' key

        inputs = gpt2_tokenizer.encode(topic, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = gpt2_model.generate(
                inputs,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=gpt2_tokenizer.eos_token_id,
                early_stopping=True
            )

        tweet = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'tweet': tweet})
    except Exception as e:
        logging.error(f"Error occurred in tweet generation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_reply', methods=['POST'])
def generate_reply():
    try:
        data = request.get_json()
        tweet = data.get('tweet', '')  # Safely get the 'tweet' key

        input_ids = bart_tokenizer.encode(tweet, return_tensors='pt').to(device)

        # Generate a reply using the BART model
        with torch.no_grad():
            reply_ids = bart_model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        reply = bart_tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        return jsonify({'reply': reply})
    except Exception as e:
        logging.error(f"Error occurred in reply generation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(port=5000, debug=True)
