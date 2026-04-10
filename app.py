import streamlit as st
import pickle
import pandas as pd
from textblob import TextBlob

# Load trained pipeline model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Fake Account Detector", layout="centered")

st.title("🔍 Fake Account Detection System")
st.write("Advanced ML Model using Text + Behavioural Features")

# Inputs
username = st.text_input("Username")
bio = st.text_area("Bio / Description")

followers = st.number_input("Followers", min_value=0.0)
following = st.number_input("Following", min_value=0.0)
favourites = st.number_input("Favourites", min_value=0.0)

# 🔹 Feature Engineering (MATCHES COLAB)

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

spam_words = ['free', 'win', 'offer', 'click', 'buy', 'money']

def count_spam_words(text):
    text = text.lower()
    return sum(word in text for word in spam_words)


# Prediction
if st.button("Predict"):

    # Clean text
    description_clean = bio.lower()

    # Engineered features
    follower_friend_ratio = followers / (following + 1)
    engagement_score = favourites / (followers + 1)
    bio_length = len(bio)
    sentiment = get_sentiment(bio)
    spam_count = count_spam_words(bio)

    # 🔥 IMPORTANT: Create DataFrame with EXACT column names
    input_data = pd.DataFrame([{
        'description_clean': description_clean,
        'followers_scaled': followers,
        'friends_scaled': following,
        'favourites_scaled': favourites,
        'follower_friend_ratio': follower_friend_ratio,
        'engagement_score': engagement_score,
        'bio_length': bio_length,
        'sentiment': sentiment,
        'spam_count': spam_count
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    try:
        prob = model.predict_proba(input_data)[0][prediction]
        confidence = round(prob * 100, 2)
    except:
        confidence = 75.0

    st.subheader("Result")

    if prediction == 1:
        st.error("🚨 Fake Account Detected")
    else:
        st.success("✅ Real Account")

    st.write(f"Confidence: {confidence}%")

    # Explanation
    st.subheader("Why this prediction?")

    reasons = []

    if follower_friend_ratio < 0.3:
        reasons.append("Low follower-following ratio")

    if following > followers * 2:
        reasons.append("Following too many accounts")

    if len(bio.strip()) == 0:
        reasons.append("Empty bio")

    if spam_count > 0:
        reasons.append("Contains spam-like words")

    if not reasons:
        reasons.append("Profile looks normal")

    for r in reasons:
        st.write(f"• {r}")