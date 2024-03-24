import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from PIL import Image
import re
from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from keras.preprocessing.text import Tokenizer
import os
import PIL
import nltk
nltk.download('punkt')





# Function to extract word vectors for tokenized text
# Extracting word vectors using Word2Vec
def extract_word_vectors(tokenized_data, vector_size=300, window=5, min_count=1, sg=0):
    model = Word2Vec(sentences=tokenized_data, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    word_vectors = {word: model.wv[word] for word in model.wv.index_to_key}
    return word_vectors

#Cleaning Malayalam Text

special_characters_to_remove = r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\\]'

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ').replace('\u200d', '').replace('\xa0', ' ').replace(
        '\u200c', '').replace('“', ' ').replace('”', ' ').replace('"', ' ').replace('\u200b', '')
    x = re.sub(r'\([^)]*\)', '', x)  # Remove text within parentheses
    x = re.sub('<[^<]+?>', '', x)  # Remove HTML tags
    x = re.sub(r'\d+(\.\d+)?', 'NUM ', x)  # Replace numbers with 'NUM'
    x = re.sub(special_characters_to_remove, ' ', x)  
    x = re.sub(r'\s+', ' ', x)  # Remove extra spaces
    return x.strip()  # Strip leading/trailing spaces

def tokenize_malayalam_ptext(data):
    tokenized_headlines = []
    cleaned_headline = fixup(data)
    words = word_tokenize(cleaned_headline, language="malayalam")
    tokenized_headlines.append(words)
    return tokenized_headlines

MAX_SEQUENCE_LENGTH = 100

# Loading the pre-trained fusion model
fusion_model = load_model('fusion__model.h5')

# Defining a function to preprocess external text data
def preprocess_text(text_data):
    tokenized_text = tokenize_malayalam_ptext(text_data)
    word_vectors = extract_word_vectors(tokenized_text)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_text)
    word_vectors = tokenizer.texts_to_sequences(tokenized_text)
    padded_sequences = pad_sequences(word_vectors, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequences

# Load and preprocess images, ensuring they have the same length as text sequences
def load_and_preprocess_images(image_paths, max_sequence_length):
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

    images = []
    for path in image_paths:
        try:
            img = Image.open(path)  
            img = img.convert("RGB")  # Convert to RGB format
            img = img.resize((224, 224))  # Resize the image to VGG-16 input size
            img = np.array(img)  # Convert PIL Image to NumPy array
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            features = model.predict(img)
            # Flatten the features and ensure they have the same length as max_sequence_length
            flattened_features = features.flatten()[:max_sequence_length]
            if len(flattened_features) < max_sequence_length:
                # If features are shorter than max_sequence_length, pad with zeros
                pad_length = max_sequence_length - len(flattened_features)
                flattened_features = np.pad(flattened_features, (0, pad_length), 'constant')
            images.append(flattened_features)
        except (PIL.UnidentifiedImageError, OSError) as e:
            print(f"Skipping image {path} due to error: {str(e)}")

    return np.array(images)

# Function to preprocess images
def preprocess_images(image_paths):
    image_features = load_and_preprocess_images(image_paths, MAX_SEQUENCE_LENGTH)
    return image_features

# Defining a function to make predictions on external data
def predict_external_data(text_data, image_paths):
    external_text_features = preprocess_text(text_data)
    external_image_features = preprocess_images(image_paths)

    # Making predictions on the external data
    external_predictions = fusion_model.predict([external_text_features, external_image_features])
    threshold = 0.5 
    external_predicted_labels = (external_predictions[:, 1] >= threshold).astype(int)
    return external_predicted_labels

# Streamlit app
st.title("Multimodal Fake News Detection")
st.sidebar.header("Navigation")

# Page selection
page = st.sidebar.radio("Go to", ("Main Page", "Predict Fake News"))

selected_model = st.sidebar.selectbox("Select Model", ["fusion_model_01.h5", "fusion__model.h5", "fUsionn_model_02.h5"])
fusion_model = load_model(selected_model)

if page == "Main Page":
    st.header("Welcome to our Multimodal Fake News Detection WebApp!")
    st.write("Welcome to our Multimodal Fake News Detection WebApp!")
    st.write("Detecting fake news in the Malayalam language has become increasingly challenging in today's digital age. With the rapid spread of misinformation and manipulated content, distinguishing between truth and falsehood has never been more crucial.")
    st.write("Our innovative web application tackles this problem head-on by utilizing the power of both text and image modalities. We understand that fake news often involves not just misleading text but also manipulated visuals, making it imperative to analyze both aspects simultaneously.")
    st.write("Through our user-friendly interface, simply input the text and the corresponding image associated with the news article you want to verify. Our advanced algorithms then meticulously analyze both the textual and visual components to determine the authenticity of the news.")
    st.write("Say goodbye to misinformation. Empower yourself with truth!")
    st.image("fake_banner01.jpg")

elif page == "Predict Fake News":
    st.header("Predict Fake News")
    st.sidebar.success("Please input text and upload an image to predict.")

    # Text input for news headline
    st.subheader("Text Input")
    text_input = st.text_area("Enter the text", value="")

    # Image upload for news image
    st.subheader("Image Input")
    image_upload = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    # Make prediction when the user clicks the "Predict" button
    if st.button("Predict"):
        if text_input == "":
            st.warning("Please enter text data.")
        elif image_upload is None:
            st.warning("Please upload an image.")
        else:
            # Read and preprocess the uploaded image
            image = Image.open(image_upload)
            image_paths = ["uploaded_image.png"]
            image.save(image_paths[0])

            # Predict using the provided text and image
            predicted_labels = predict_external_data(text_input, image_paths)

            category = {0:"The Given News is fake", 1: "The given News is True"}
            result = category[predicted_labels[0]]
            st.success(f"Predicted Label: {result}")
