import streamlit as st 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
df= pd.read_csv(r"C:\Users\maren\INNOMATICS_327\ML_classes_327\PROJECTS\SPAM_OR _HAM_PREDICTION_Project_2\spam.csv") 
st.markdown(
    "<h1 style='text-align: center;'>📩EMAIL <span style='color: red;'>SPAM</span> or <span style='color: green;'>HAM</span> CLASSIFIER PREDICTON</h1>",
    unsafe_allow_html=True)
# About the Project
st.write("""
This project focuses on building a machine learning model to classify emails as **Spam (Unwanted) or Ham (Legitimate)** based on their textual content. The dataset used contains labeled email messages, and the project follows a structured approach, including **data preprocessing, exploratory data analysis (EDA), feature engineering, and model building**.  

The key steps in this project include:  
- **Data Understanding & Cleaning**: Loading the dataset, handling missing values, and preprocessing text data.  
- **Exploratory Data Analysis (EDA)**: Analyzing the distribution of spam vs. ham emails, visualizing common words, and detecting patterns.  
- **Feature Engineering**: Converting text into numerical features using techniques like **TF-IDF and Count Vectorization**.  
- **Model Training & Evaluation**: Implementing machine learning algorithms like **Naïve Bayes, Logistic Regression, or Random Forest** to classify emails effectively.  
- **Performance Metrics**: Evaluating accuracy, precision, recall, and F1-score to ensure reliable predictions.  

This project helps in automating spam detection, improving email filtering mechanisms, and enhancing cybersecurity by preventing phishing attacks.
""")
if "Category" in df.columns and "Message" in df.columns:

        # Encode labels ('ham' -> 0, 'spam' -> 1)
        df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

        # Extract features and labels
        X = df["Message"]
        Y = df["Category"]

        # Convert text to numerical representation

        bow = CountVectorizer(stop_words="english")
        final_X = bow.fit_transform(X).toarray()

        # Train-Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(final_X, Y, test_size=0.25, random_state=20)

        # Model Selection Dropdown
        model_choice = st.selectbox(
            "Select a Classification Model",
            ("Logistic Regression", "Naïve Bayes", "K-Nearest Neighbors", "Decision Tree", "Support Vector Machine"),
        )

        # Initialize and train the selected model
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Naïve Bayes":
            model = MultinomialNB()
        elif model_choice == "K-Nearest Neighbors":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "Support Vector Machine":
            model = SVC()

        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)



# Display Accuracy Score with Progress Bar
if st.button("📊 Compute Model Accuracy", key="accuracy_button"):
    accuracy = accuracy_score(Y_test, y_pred)
    st.success(f"✅ **{model_choice} Accuracy:** {accuracy:.2%}")
    
    # Add a progress bar visualization for accuracy
    st.progress(float(accuracy))

# Email Classification Input Section
email_input = st.text_area("📩 **Enter an Email Message for Classification:**", placeholder="Type your email here...")

if st.button("🚀 Classify Email Now!", key="predict_button"):
    if email_input.strip():
        # Transform the input text into a vectorized format
        input_vectorized = bow.transform([email_input]).toarray()
        prediction = model.predict(input_vectorized)[0]

        # Unique Display for Spam & Ham Classification
        if prediction == 1:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(to right, #ff6a6a, #ff2e2e); 
                    padding: 20px; 
                    border-radius: 10px; 
                    text-align: center; 
                    box-shadow: 3px 3px 10px rgba(255, 0, 0, 0.4);
                ">
                    <h2 style='color: white;'>🚨 ALERT: SPAM DETECTED! 🚨</h2>
                    
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(to right, #a8ff78, #78ffd6); 
                    padding: 20px; 
                    border-radius: 10px; 
                    text-align: center; 
                    box-shadow: 3px 3px 10px rgba(0, 255, 0, 0.4);
                ">
                    <h2 style='color: black;'>✅ HAM - SAFE EMAIL ✅</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # Animated warning for empty input
        st.warning("⚠️ **Oops!** You forgot to enter an email. Please type a message before predicting.")
