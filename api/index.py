from flask import Flask, request, jsonify, render_template
import pandas as pd
from convokit import Corpus, Utterance, Speaker
from transformers import pipeline
import logging
import os

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set the Hugging Face token as an environment variable
os.environ['HF_TOKEN'] = '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'

# Helper function to split text into chunks
def split_text(text, max_length=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Load summarization pipeline with authentication
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", use_auth_token=os.environ['HF_TOKEN'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    logs_df = pd.read_csv(file)

    # Convert 'Date' column to datetime format
    logs_df['Date'] = pd.to_datetime(logs_df['Date'], format='%Y-%m-%d,%H:%M:%S')

    # Create a dictionary to hold Speaker objects
    speakers = {}

    # Convert logs into ConvoKit Utterances
    utterances = []
    for idx, row in logs_df.iterrows():
        user_tag = row['User tag']
        if user_tag not in speakers:
            speakers[user_tag] = Speaker(id=user_tag)
        utterances.append(Utterance(
            id=str(idx),
            speaker=speakers[user_tag],
            text=row['Content'],
            timestamp=row['Date']
        ))

    # Create a ConvoKit Corpus
    corpus = Corpus(utterances=utterances)

    # Concatenate all texts for summarization
    full_text = " ".join([utterance.text for utterance in corpus.iter_utterances()])

    # Split the text into smaller chunks
    text_chunks = split_text(full_text)

    # Summarize each chunk
    chunk_summaries = []
    for chunk in text_chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            chunk_summaries.append(summary[0]['summary_text'])
            logging.info("Chunk summarized successfully.")
        except Exception as e:
            logging.error(f"Error summarizing chunk: {e}")

    # Combine the chunk summaries into a final summary
    final_summary = " ".join(chunk_summaries)

    return jsonify({"summary": final_summary})

if __name__ == '__main__':
    app.run()
