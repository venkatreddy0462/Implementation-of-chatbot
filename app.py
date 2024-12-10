import os
import json
import datetime
import csv
import nltk # type: ignore
import ssl
import streamlit as st
import random
import google.generativeai as genai 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')
genai.configure(api_key="AIzaSyDAckJOelAP5oX6JrKWyJmj0YAoJHlruIM") 

#def Ai(user_input):
#    try:
#        model = genai.GenerativeModel("gemini-1.5-flash")  # Assuming correct model identifier
#        response = model.generate_content(user_input)  # Updated method to generate content
#        response1 = extract_text(response) + "\n\n Satisfied with the response ?   Yes or No"  # Or use the correct property from response object
#        return response1.replace("*","")
#    except Exception as e:
#        return f"Error: {e}"
    
#def extract_text(response):
#    try:
#        # Navigate through the nested structure to extract the text field
#        text = response.candidates[0].content.parts[0].text
#        return text
#    except Exception as e:
#       return f"Error extracting text: {e}"


# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 5))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
def entered(user_input):
    return user_input is not None and user_input != ""
        
counter = 0
if "input_received" not in st.session_state:
    st.session_state.input_received = False
def main():
    global counter
    st.title("Chatbot using NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About", "Contact Developer"]
    choice = st.sidebar.selectbox("Menu", menu)

    

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        
# Counter to manage unique keys for input widgets
        counter = 0
        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")
        if user_input:
            st.session_state.input_received = True
        if st.session_state.input_received:
            user_input_str = str(user_input)

#            response = Ai(user_input_str)
            response2 = chatbot(user_input_str)
                    
        # Display chatbot response
#            """st.text_area("GenAI Chatbot:", value=response, height=120, max_chars=None, key=f"GenAi_chatbot_response_{counter}")"""
            st.text_area("Conventional Chatbot:", value=response2, height=120, max_chars=None, key=f"Conventional_chatbot_response_{counter}")

        # Log interaction in CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response2, timestamp])
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            rows = list(csv_reader)
            for row in rows[:0:-1]:
                st.text(f"User: {row[0]}")
#                """st.text(f"GenAi Chatbot: {row[1]}")"""
                st.text(f"Chatbot: {row[1]}")
    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
#                """st.text(f"GenAi Chatbot: {row[1]}")"""
                st.text(f"Conventional Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":

        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")

        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")
    elif choice == "Contact Developer":
        st.subheader(f"WhatsApp: {6304082260}")
        st.subheader("Email "+"rikhithreddy@gmail.com")
        st.write()

if __name__ == '__main__':
    main()
