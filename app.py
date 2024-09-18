import json
from flask import Flask, request, jsonify
import openai
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load product data from external JSON file
with open('infromation.json', 'r') as f:
    products_data = json.load(f)

# Initialize OpenAI GPT-4 API (set your API key here)
openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize sentiment analysis using Hugging Face pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyze user sentiment using Hugging Face's pre-trained model
def analyze_user_sentiment(user_input):
    result = sentiment_analyzer(user_input)[0]
    sentiment = result['label']
    score = result['score']
    return sentiment, score

# Find a similar product if user is not interested in the current one
def suggest_similar_products(category):
    similar_products = [product for product in products_data['products'] if product['category'] == category]
    return similar_products

# Handle the negotiation conversation with GPT-4 and sentiment analysis
@app.route('/neg', methods=['POST'])
def negotiate():
    try:
        user_input = request.json.get('user_input')
        product_name = request.json.get('product_name')

        # Get the product details
        product = next((p for p in products_data['products'] if p['name'].lower() == product_name.lower()), None)

        if not product:
            return jsonify({"message": "Product not found"}), 404

        # Call GPT-4 for conversation flow
        conversation = f"The user is trying to buy a {product_name}. The max price is {product['max_price']}, and the min price is {product['min_price']}. Negotiate the price."
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=conversation + "\nUser: " + user_input + "\nAI:",
            max_tokens=150
        )

        bot_reply = response.choices[0].text.strip()

        # Perform sentiment analysis on user's input
        sentiment, score = analyze_user_sentiment(user_input)

        # If sentiment is negative and user doesn't seem interested, suggest similar products
        if sentiment == 'NEGATIVE' and score > 0.75:
            similar_products = suggest_similar_products(product['category'])
            return jsonify({
                "bot_reply": bot_reply,
                "sentiment": "negative",
                "similar_products": similar_products
            })

        return jsonify({
            "bot_reply": bot_reply,
            "sentiment": sentiment
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

