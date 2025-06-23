import streamlit as st
import pandas as pd
import numpy as np
import os
import string
import time
import nltk
import zipfile
import operator
import logging
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from io import StringIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Disable TensorFlow to avoid CUDA-related errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="NLP Applications",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {str(e)}")
        return False

# Initialize the app
if not download_nltk_data():
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Application", ["Home", "Email Spam Detection", "Resume Recommendation"])

# Main content
if page == "Home":
    st.title("🚀 NLP Applications")
    
    st.markdown("""
    ## Welcome to the NLP Applications Dashboard
    
    This application provides two main functionalities:
    
    ### 1. Email Spam Detection
    Upload email data and train models (SVM or DistilBERT) to classify emails as spam or ham (not spam).
    
    
    ### 2. Resume Recommendation System
    Upload resume data and find the best candidates based on skills, experience, and other criteria.
    
    Use the sidebar to navigate between different applications.
    """)
    
    st.info("Get started by selecting an application from the sidebar.")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Email Spam Detection")
           
        
        with col2:
            st.markdown("### Resume Recommendation")
            

elif page == "Email Spam Detection":
    st.title("📧 Email Spam Detection")
    
    # Function to clean up the text
    def text_cleanup(text):
        try:
            text_no_punct = ''.join([c for c in text if c not in string.punctuation])
            stop_words = set(stopwords.words('english'))
            words = text_no_punct.split()
            words_filtered = [word.lower() for word in words if word.lower() not in stop_words]
            return words_filtered
        except Exception as e:
            logger.error(f"Error in text_cleanup: {str(e)}")
            return []
    
    # Function to extract words and their frequencies
    def extract_word_frequencies(file_list):
        if not file_list:
            st.error("No files to process.")
            return [], []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        lmtzr = WordNetLemmatizer()
        word_count = {}
        processed_files = 0
        total_files = len(file_list)
        
        for i, file_content in enumerate(file_list):
            try:
                if not isinstance(file_content, str) or not file_content.strip():
                    logger.warning(f"Skipping file {i}: Empty or invalid content")
                    continue
                words = text_cleanup(file_content)
                for word in words:
                    if not word.isdigit() and len(word) > 2:
                        lemma = lmtzr.lemmatize(word)
                        word_count[lemma] = word_count.get(lemma, 0) + 1
                
                processed_files += 1
                progress_bar.progress(processed_files / total_files)
                if processed_files % max(1, total_files // 10) == 0:
                    status_text.text(f"Processed {processed_files}/{total_files} files.")
            except Exception as e:
                logger.error(f"Error processing file {i}: {str(e)}")
                continue
        
        progress_bar.progress(1.0)
        status_text.text(f"Done. Total files processed: {processed_files}")
        
        sorted_word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
        min_freq = st.slider("Minimum Word Frequency", 5, 50, 10)
        frequent_words = [word for word, freq in sorted_word_count if freq >= min_freq]
        
        return frequent_words, sorted_word_count
    
    # Function to create feature vectors for SVM
    def create_feature_vectors(file_list, file_names, word_list):
        if not file_list or not file_names or not word_list:
            st.error("Invalid input for feature vector creation.")
            return [], []
        
        # Validate input lengths
        if len(file_list) != len(file_names):
            st.error(f"Mismatch in file_list ({len(file_list)}) and file_names ({len(file_names)}) lengths.")
            logger.error(f"Mismatch in file_list ({len(file_list)}) and file_names ({len(file_names)}) lengths.")
            return [], []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        lmtzr = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        feature_vectors = []
        labels = []
        
        for i, (file_content, file_name) in enumerate(zip(file_list, file_names)):
            try:
                # Skip empty or invalid content
                if not file_content or not isinstance(file_content, str):
                    logger.warning(f"Skipping file {file_name}: Empty or invalid content")
                    continue
                    
                word_vector = np.zeros(len(word_list), dtype=int)
                
                words = file_content.split()
                for word in words:
                    word = word.lower().strip(string.punctuation)
                    if word in stop_words or len(word) <= 2 or word.isdigit():
                        continue
                    word = lmtzr.lemmatize(word)
                    if word in word_list:
                        index = word_list.index(word)
                        word_vector[index] += 1
                
                label = -1 if 'spam' in file_name.lower() else 1
                feature_vectors.append(word_vector)
                labels.append(label)
                
                progress_bar.progress((i + 1) / len(file_list))
                if i % max(1, len(file_list) // 10) == 0:
                    status_text.text(f"Created feature vectors for {i + 1}/{len(file_list)} files.")
            except Exception as e:
                logger.error(f"Error creating feature vector for file {file_name}: {str(e)}")
                continue
        
        # Validate output lengths
        if len(feature_vectors) != len(labels):
            st.error(f"Mismatch in feature_vectors ({len(feature_vectors)}) and labels ({len(labels)}) lengths.")
            logger.error(f"Mismatch in feature_vectors ({len(feature_vectors)}) and labels ({len(labels)}) lengths.")
            return [], []
        
        progress_bar.progress(1.0)
        status_text.text(f"Done. Created {len(feature_vectors)} feature vectors.")
        
        return feature_vectors, labels
    
    # Function to train and evaluate SVM model
    def train_evaluate_svm(X_train, Y_train, X_test, Y_test, kernel_type, params):
        if len(X_train) == 0 or len(X_test) == 0:
            st.error("Insufficient data for training or testing.")
            return None, None, None, None, None
        
        # Validate array lengths
        if len(X_train) != len(Y_train) or len(X_test) != len(Y_test):
            st.error(f"Mismatch in training data (X_train: {len(X_train)}, Y_train: {len(Y_train)}) or test data (X_test: {len(X_test)}, Y_test: {len(Y_test)}).")
            logger.error(f"Mismatch in training data (X_train: {len(X_train)}, Y_train: {len(Y_train)}) or test data (X_test: {len(X_test)}, Y_test: {len(Y_test)}).")
            return None, None, None, None, None
        
        start_time = time.time()
        try:
            with st.spinner(f"Training {kernel_type} SVM model..."):
                model = SVC(kernel=kernel_type, C=0.1, **params)
                model.fit(X_train, Y_train)
                predictions = model.predict(X_test)
            
            cm = confusion_matrix(Y_test, predictions, labels=[1, -1])
            precision = precision_score(Y_test, predictions, pos_label=1)
            recall = recall_score(Y_test, predictions, pos_label=1)
            
            st.success(f"Model training complete in {round(time.time() - start_time, 2)} seconds")
            
            classification = pd.DataFrame({
                'Actual': Y_test,
                'Predicted': predictions,
                'Status': ['Correct' if true == pred else 'Incorrect' for true, pred in zip(Y_test, predictions)]
            })
            
            return model, cm, precision, recall, classification
        except Exception as e:
            logger.error(f"Error training SVM model: {str(e)}")
            st.error(f"Error training SVM model: {str(e)}")
            return None, None, None, None, None
    
    # Function to preprocess data for DistilBERT
    def preprocess_for_distilbert(file_list, file_names, max_length=64):
        if not file_list or not file_names:
            st.error("Invalid input for DistilBERT preprocessing.")
            return None, None, None
        
        # Validate input lengths
        if len(file_list) != len(file_names):
            st.error(f"Mismatch in file_list ({len(file_list)}) and file_names ({len(file_names)}) lengths.")
            logger.error(f"Mismatch in file_list ({len(file_list)}) and file_names ({len(file_names)}) lengths.")
            return None, None, None
        
        # Filter out invalid entries
        valid_texts = []
        valid_names = []
        for text, name in zip(file_list, file_names):
            if isinstance(text, str) and text.strip():
                valid_texts.append(text)
                valid_names.append(name)
            else:
                logger.warning(f"Skipping file {name}: Empty or invalid content")
        
        if not valid_texts:
            st.error("No valid texts to process for DistilBERT.")
            logger.error("No valid texts to process for DistilBERT.")
            return None, None, None
        
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        texts = valid_texts
        labels = [0 if 'spam' in name.lower() else 1 for name in valid_names]  # 0 for spam, 1 for ham
        
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        labels = torch.tensor(labels)
        
        # Validate output lengths
        if len(input_ids) != len(attention_mask) or len(input_ids) != len(labels):
            st.error(f"Mismatch in input_ids ({len(input_ids)}), attention_mask ({len(attention_mask)}), or labels ({len(labels)}) lengths.")
            logger.error(f"Mismatch in input_ids ({len(input_ids)}), attention_mask ({len(attention_mask)}), or labels ({len(labels)}) lengths.")
            return None, None, None
        
        return input_ids, attention_mask, labels
    
    # Function to train and evaluate DistilBERT model
    def train_evaluate_distilbert(input_ids, attention_mask, labels, test_size=0.3, epochs=1, batch_size=8):
        if input_ids is None or attention_mask is None or labels is None:
            st.error("Invalid input for DistilBERT training.")
            return None, None, None, None, None
        
        # Validate lengths
        if len(input_ids) != len(attention_mask) or len(input_ids) != len(labels):
            st.error(f"Mismatch in input_ids ({len(input_ids)}), attention_mask ({len(attention_mask)}), or labels ({len(labels)}) lengths.")
            logger.error(f"Mismatch in input_ids ({len(input_ids)}), attention_mask ({len(attention_mask)}), or labels ({len(labels)}) lengths.")
            return None, None, None, None, None
        
        # Split data
        dataset = TensorDataset(input_ids, attention_mask, labels)
        train_size = int((1 - test_size) * len(dataset))
        if train_size == 0 or len(dataset) - train_size == 0:
            st.error("Insufficient data for training or testing after split.")
            logger.error("Insufficient data for training or testing after split.")
            return None, None, None, None, None
        
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Load DistilBERT model
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        device = torch.device('cpu')  # Force CPU usage
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        
        # Training
        start_time = time.time()
        try:
            with st.spinner("Training DistilBERT model (optimized for speed)..."):
                model.train()
                for epoch in range(epochs):
                    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
                    for batch in progress_bar:
                        b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]
                        outputs = model(
                            input_ids=b_input_ids,
                            attention_mask=b_attention_mask,
                            labels=b_labels
                        )
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        progress_bar.set_postfix({'loss': loss.item()})
            
            # Evaluation
            model.eval()
            predictions = []
            true_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]
                    outputs = model(
                        input_ids=b_input_ids,
                        attention_mask=b_attention_mask
                    )
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    predictions.extend(preds)
                    true_labels.extend(b_labels.cpu().numpy())
            
            # Convert labels for consistency with SVM (1 for ham, -1 for spam)
            predictions = [1 if pred == 1 else -1 for pred in predictions]
            true_labels = [1 if label == 1 else -1 for label in true_labels]
            
            cm = confusion_matrix(true_labels, predictions, labels=[1, -1])
            precision = precision_score(true_labels, predictions, pos_label=1)
            recall = recall_score(true_labels, predictions, pos_label=1)
            
            st.success(f"DistilBERT training complete in {round(time.time() - start_time, 2)} seconds")
            st.info("Note: DistilBERT is optimized for speed with a shorter max_length, fewer epochs, and smaller batch size, which may slightly reduce accuracy.")
            
            classification = pd.DataFrame({
                'Actual': true_labels,
                'Predicted': predictions,
                'Status': ['Correct' if true == pred else 'Incorrect' for true, pred in zip(true_labels, predictions)]
            })
            
            return model, cm, precision, recall, classification
        except Exception as e:
            logger.error(f"Error training DistilBERT model: {str(e)}")
            st.error(f"Error training DistilBERT model: {str(e)}")
            return None, None, None, None, None
    
    # Function to predict with DistilBERT for a single email
    def predict_with_distilbert(model, email_text, tokenizer, max_length=64):
        model.eval()
        device = torch.device('cpu')  # Force CPU usage
        model.to(device)
        
        encoding = tokenizer(
            [email_text],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        
        return 1 if pred == 1 else -1  # 1 for ham, -1 for spam
    
    # File Upload Section
    st.header("Step 1: Upload Email Data")
    uploaded_file = st.file_uploader("Upload a zip file containing emails", type=["zip"])
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.write("Or try with sample data:")
        with col2:
            use_sample_data = st.checkbox("Use sample data")
    
    if uploaded_file is not None or use_sample_data:
        st.header("Step 2: Extract Words and Create Features")
        
        if st.button("Process Emails"):
            for key in ['file_list', 'file_names', 'frequent_words', 'word_count', 'X_train', 'X_test', 'Y_train', 'Y_test', 'linear_model', 'poly_model', 'distilbert_model', 'distilbert_tokenizer']:
                if key in st.session_state:
                    del st.session_state[key]
            
            with st.spinner("Extracting emails..."):
                if uploaded_file is not None:
                    try:
                        file_list = []
                        file_names = []
                        with zipfile.ZipFile(uploaded_file) as z:
                            for file_info in z.infolist():
                                with z.open(file_info) as f:
                                    file_list.append(f.read().decode('utf-8', errors='ignore'))
                                    file_names.append(file_info.filename)
                        # Validate lengths
                        if len(file_list) != len(file_names):
                            st.error(f"Mismatch in file_list ({len(file_list)}) and file_names ({len(file_names)}) lengths.")
                            logger.error(f"Mismatch in file_list ({len(file_list)}) and file_names ({len(file_names)}) lengths.")
                            st.stop()
                        st.session_state.file_list = file_list
                        st.session_state.file_names = file_names
                        st.success(f"Extracted {len(file_list)} emails from ZIP file.")
                    except Exception as e:
                        st.error(f"Error processing ZIP file: {str(e)}")
                        logger.error(f"ZIP file processing error: {str(e)}")
                        st.stop()
                else:
                    sample_emails = [
                        "Hello, how are you? I'm doing well.",
                        "URGENT: You've won a prize! Claim now!",
                        "Meeting scheduled for tomorrow at 2pm.",
                        "Get rich quick! Amazing investment opportunity!",
                        "Please review the attached document for our project.",
                        "Increase your followers! Buy our service now!",
                        "Reminder: Doctor's appointment on Friday.",
                        "Free money! Click here to claim $5000!",
                        "Weekly team report: progress on all tasks.",
                        "Hot singles in your area want to meet you!"
                    ]
                    sample_filenames = [
                        "email1.txt", "spam1.txt", "email2.txt", "spam2.txt",
                        "email3.txt", "spam3.txt", "email4.txt", "spam4.txt",
                        "email5.txt", "spam5.txt"
                    ]
                    # Validate lengths for sample data
                    if len(sample_emails) != len(sample_filenames):
                        st.error(f"Mismatch in sample_emails ({len(sample_emails)}) and sample_filenames ({len(sample_filenames)}) lengths.")
                        logger.error(f"Mismatch in sample_emails ({len(sample_emails)}) and sample_filenames ({len(sample_filenames)}) lengths.")
                        st.stop()
                    st.session_state.file_list = sample_emails
                    st.session_state.file_names = sample_filenames
                    st.success("Sample data loaded successfully.")
            
            with st.spinner("Extracting frequent words..."):
                frequent_words, word_count = extract_word_frequencies(st.session_state.file_list)
                if not frequent_words:
                    st.error("No frequent words found. Please adjust the minimum frequency or check input data.")
                    st.stop()
                st.session_state.frequent_words = frequent_words
                st.session_state.word_count = word_count
                
                word_count_df = pd.DataFrame(word_count[:20], columns=["Word", "Frequency"])
                st.write("Top 20 Most Frequent Words:")
                st.dataframe(word_count_df)
                
                fig = px.bar(
                    word_count_df,
                    x="Word",
                    y="Frequency",
                    title="Top 20 Most Frequent Words",
                    template="plotly_white"
                )
                fig.update_traces(marker=dict(line=dict(width=0.5)))
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with st.spinner("Creating feature vectors..."):
                X, Y = create_feature_vectors(
                    st.session_state.file_list,
                    st.session_state.file_names,
                    st.session_state.frequent_words
                )
                if not X or not Y:
                    st.error("Failed to create feature vectors. Please check input data.")
                    st.stop()
                
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, test_size=0.3, random_state=42
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.Y_train = Y_train
                st.session_state.Y_test = Y_test
                
                st.success("Feature vectors created successfully!")
                st.write(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        if 'file_list' in st.session_state:
            st.header("Step 3: Train and Evaluate Models")
            
            tab1, tab2, tab3 = st.tabs(["Linear Kernel", "Polynomial Kernel", "DistilBERT"])
            
            with tab1:
                if st.button("Train Linear SVM"):
                    model, cm, precision, recall, classification = train_evaluate_svm(
                        st.session_state.X_train,
                        st.session_state.Y_train,
                        st.session_state.X_test,
                        st.session_state.Y_test,
                        "linear",
                        {}
                    )
                    if model is None:
                        st.stop()
                    
                    cm_df = pd.DataFrame(
                        cm,
                        index=['Ham (Actual)', 'Spam (Actual)'],
                        columns=['Ham (Predicted)', 'Spam (Predicted)']
                    )
                    
                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Confusion Matrix:")
                            st.dataframe(cm_df)
                        
                        with col2:
                            st.write("Performance Metrics:")
                            st.metric("Precision", f"{precision:.2f}")
                            st.metric("Recall", f"{recall:.2f}")
                            st.metric("Accuracy", f"{(cm[0,0] + cm[1,1]) / np.sum(cm):.2f}")
                    
                    fig = px.imshow(
                        cm,
                        x=['Ham (Predicted)', 'Spam (Predicted)'],
                        y=['Ham (Actual)', 'Spam (Actual)'],
                        text_auto=True,
                        color_continuous_scale='Blues',
                        title='Confusion Matrix - Linear SVM'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("Classification Examples:")
                    st.dataframe(classification.head(10))
                    
                    st.session_state.linear_model = model
            
            with tab2:
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        degree = st.slider("Polynomial Degree", 2, 5, 2)
                    with col2:
                        coef0 = st.slider("Coefficient", 0, 10, 1)
                
                if st.button("Train Polynomial SVM"):
                    model, cm, precision, recall, classification = train_evaluate_svm(
                        st.session_state.X_train,
                        st.session_state.Y_train,
                        st.session_state.X_test,
                        st.session_state.Y_test,
                        "poly",
                        {"degree": degree, "coef0": coef0}
                    )
                    if model is None:
                        st.stop()
                    
                    cm_df = pd.DataFrame(
                        cm,
                        index=['Ham (Actual)', 'Spam (Actual)'],
                        columns=['Ham (Predicted)', 'Spam (Predicted)']
                    )
                    
                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Confusion Matrix:")
                            st.dataframe(cm_df)
                        
                        with col2:
                            st.write("Performance Metrics:")
                            st.metric("Precision", f"{precision:.2f}")
                            st.metric("Recall", f"{recall:.2f}")
                            st.metric("Accuracy", f"{(cm[0,0] + cm[1,1]) / np.sum(cm):.2f}")
                    
                    fig = px.imshow(
                        cm,
                        x=['Ham (Predicted)', 'Spam (Predicted)'],
                        y=['Ham (Actual)', 'Spam (Actual)'],
                        text_auto=True,
                        color_continuous_scale='Greens',
                        title=f'Confusion Matrix - Polynomial SVM (degree={degree}, coef0={coef0})'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("Classification Examples:")
                    st.dataframe(classification.head(10))
                    
                    st.session_state.poly_model = model
            
            with tab3:
                if st.button("Train DistilBERT"):
                    with st.spinner("Preprocessing data for DistilBERT..."):
                        input_ids, attention_mask, labels = preprocess_for_distilbert(
                            st.session_state.file_list,
                            st.session_state.file_names
                        )
                        if input_ids is None:
                            st.stop()
                    
                    model, cm, precision, recall, classification = train_evaluate_distilbert(
                        input_ids,
                        attention_mask,
                        labels
                    )
                    if model is None:
                        st.stop()
                    
                    cm_df = pd.DataFrame(
                        cm,
                        index=['Ham (Actual)', 'Spam (Actual)'],
                        columns=['Ham (Predicted)', 'Spam (Predicted)']
                    )
                    
                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Confusion Matrix:")
                            st.dataframe(cm_df)
                        
                        with col2:
                            st.write("Performance Metrics:")
                            st.metric("Precision", f"{precision:.2f}")
                            st.metric("Recall", f"{recall:.2f}")
                            st.metric("Accuracy", f"{(cm[0,0] + cm[1,1]) / np.sum(cm):.2f}")
                    
                    fig = px.imshow(
                        cm,
                        x=['Ham (Predicted)', 'Spam (Predicted)'],
                        y=['Ham (Actual)', 'Spam (Actual)'],
                        text_auto=True,
                        color_continuous_scale='Purples',
                        title='Confusion Matrix - DistilBERT'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("Classification Examples:")
                    st.dataframe(classification.head(10))
                    
                    st.session_state.distilbert_model = model
                    st.session_state.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            if 'linear_model' in st.session_state or 'poly_model' in st.session_state or 'distilbert_model' in st.session_state:
                st.header("Step 4: Test with New Email")
                
                test_email = st.text_area(
                    "Enter an email to classify:",
                    "Dear friend, I have a great opportunity for you to make money quickly..."
                )
                
                model_type = st.radio("Select model for classification:", ("Linear SVM", "Polynomial SVM", "DistilBERT"))
                
                if st.button("Classify Email"):
                    if not test_email.strip():
                        st.error("Please enter an email to classify.")
                    else:
                        try:
                            if model_type == "Linear SVM" and 'linear_model' in st.session_state:
                                lmtzr = WordNetLemmatizer()
                                stop_words = set(stopwords.words('english'))
                                word_vector = np.zeros(len(st.session_state.frequent_words), dtype=int)
                                
                                words = test_email.split()
                                for word in words:
                                    word = word.lower().strip(string.punctuation)
                                    if word in stop_words or len(word) <= 2 or word.isdigit():
                                        continue
                                    word = lmtzr.lemmatize(word)
                                    if word in st.session_state.frequent_words:
                                        index = st.session_state.frequent_words.index(word)
                                        word_vector[index] += 1
                                
                                prediction = st.session_state.linear_model.predict([word_vector])[0]
                            
                            elif model_type == "Polynomial SVM" and 'poly_model' in st.session_state:
                                lmtzr = WordNetLemmatizer()
                                stop_words = set(stopwords.words('english'))
                                word_vector = np.zeros(len(st.session_state.frequent_words), dtype=int)
                                
                                words = test_email.split()
                                for word in words:
                                    word = word.lower().strip(string.punctuation)
                                    if word in stop_words or len(word) <= 2 or word.isdigit():
                                        continue
                                    word = lmtzr.lemmatize(word)
                                    if word in st.session_state.frequent_words:
                                        index = st.session_state.frequent_words.index(word)
                                        word_vector[index] += 1
                                
                                prediction = st.session_state.poly_model.predict([word_vector])[0]
                            
                            elif model_type == "DistilBERT" and 'distilbert_model' in st.session_state:
                                prediction = predict_with_distilbert(
                                    st.session_state.distilbert_model,
                                    test_email,
                                    st.session_state.distilbert_tokenizer
                                )
                            
                            else:
                                st.error("Selected model not trained yet!")
                                st.stop()
                            
                            if prediction == 1:
                                st.success("This email is classified as: HAM (not spam)")
                            else:
                                st.error("This email is classified as: SPAM")
                            
                            if model_type in ["Linear SVM", "Polynomial SVM"] and 'frequent_words' in st.session_state:
                                words_present = [(st.session_state.frequent_words[i], count) 
                                                for i, count in enumerate(word_vector) if count > 0]
                                if words_present:
                                    st.write("Key words found in the email:")
                                    words_df = pd.DataFrame(words_present, columns=["Word", "Count"])
                                    st.dataframe(words_df)
                        except Exception as e:
                            st.error(f"Error classifying email: {str(e)}")
                            logger.error(f"Email classification error: {str(e)}")
            
            # Interactive Analysis Canvas
            with st.expander("Interactive Analysis Canvas"):
                st.write("Visualize or analyze data interactively")
                chart_type = st.selectbox("Select Chart Type", ["Word Frequency", "Sample Scatter"])
                if chart_type == "Word Frequency" and 'word_count' in st.session_state:
                    word_count_df = pd.DataFrame(st.session_state.word_count[:20], columns=["Word", "Frequency"])
                    fig = px.bar(word_count_df, x="Word", y="Frequency", title="Top 20 Most Frequent Words")
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "Sample Scatter":
                    fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6], title="Sample Scatter Plot")
                    st.plotly_chart(fig, use_container_width=True)

elif page == "Resume Recommendation":
    st.title("📄 Resume Recommendation System")
    
    # Function to send emails to candidates
    def send_acceptance_email(candidate_name, candidate_email, company_name, job_position, email_subject, email_body):
        if not st.session_state.get('email_configured', False):
            st.error("Email settings not configured. Please configure email settings first.")
            return False
        
        smtp_server = st.session_state.get('smtp_server')
        smtp_port = st.session_state.get('smtp_port')
        sender_email = st.session_state.get('sender_email')
        sender_password = st.session_state.get('sender_password')
        
        if not all([smtp_server, smtp_port, sender_email, sender_password]):
            st.error("Incomplete email configuration.")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = candidate_email
            msg['Subject'] = email_subject
            
            personalized_body = email_body.replace("{candidate_name}", candidate_name)
            personalized_body = personalized_body.replace("{company_name}", company_name)
            personalized_body = personalized_body.replace("{job_position}", job_position)
            
            msg.attach(MIMEText(personalized_body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            st.error(f"Failed to send email: {str(e)}")
            logger.error(f"Email sending error: {str(e)}")
            return False
    
    # Sample resume data
    @st.cache_data
    def load_sample_resume_data():
        data = {
            "Resume | Name": [
                "Alex Johnson", "Sarah Smith", "Michael Chen", 
                "Jessica Kim", "Daniel Brown", "Emma Wilson",
                "David Martin", "Sophie Lee", "James Thompson",
                "Olivia Davis"
            ],
            "Emails": [
                "alex.johnson@example.com", "sarah.smith@example.com", "michael.chen@example.com",
                "jessica.kim@example.com", "daniel.brown@example.com", "emma.wilson@example.com",
                "david.martin@example.com", "sophie.lee@example.com", "james.thompson@example.com",
                "olivia.davis@example.com"
            ],
            "Job Role": [
                "Data Scientist", "Software Engineer", "Data Analyst",
                "ML Engineer", "Software Developer", "Data Engineer",
                "Full Stack Developer", "Data Scientist", "Backend Developer",
                "Frontend Developer"
            ],
            "Skills": [
                "Python, TensorFlow, SQL, NLP, Machine Learning",
                "Java, Spring, Docker, Kubernetes, Microservices",
                "Python, SQL, Tableau, Excel, Statistics",
                "Python, PyTorch, TensorFlow, Computer Vision, NLP",
                "JavaScript, React, Node.js, MongoDB, Express",
                "Python, Spark, Hadoop, SQL, Airflow",
                "JavaScript, React, Node.js, Python, MongoDB",
                "Python, R, Machine Learning, Statistics, NLP",
                "Java, Spring Boot, Hibernate, SQL, Docker",
                "HTML, CSS, JavaScript, React, TypeScript"
            ],
            "Certificati": [
                "AWS Certified Machine Learning", "None", "Google Data Analytics",
                "AWS Machine Learning Specialty", "AWS Developer Associate",
                "Google Cloud Professional Data Engineer", "AWS Solutions Architect",
                "Microsoft Azure Data Scientist", "None", "Google UX Design"
            ],
            "Education": [
                "MS Computer Science", "BS Computer Science", "BS Statistics",
                "PhD Computer Science", "BS Computer Engineering",
                "MS Data Science", "BS Information Technology",
                "MS Machine Learning", "BS Computer Science", "BS Web Development"
            ],
            "Experienc": [5, 3, 2, 7, 4, 6, 3, 8, 5, 2],
            "Projects C": [8, 5, 3, 12, 7, 9, 6, 10, 8, 4],
            "Salary Exp": [110000, 95000, 85000, 130000, 100000, 115000, 90000, 125000, 105000, 85000],
            "Recruiter": [
                "AI Resear Hire", "Reject", "Hire",
                "AI Resear Hire", "Reject", "Hire",
                "Reject", "AI Resear Hire", "Hire",
                "Reject"
            ]
        }
        return pd.DataFrame(data)
    
    # Function to normalize skill names
    def normalize_skills(skill_input):
        try:
            if isinstance(skill_input, list):
                return [s.strip().lower() for s in skill_input if s and isinstance(s, str) and s.strip()]
            elif isinstance(skill_input, str):
                return [s.strip().lower() for s in skill_input.split(',') if s and s.strip()]
            elif pd.isna(skill_input) or skill_input is None:
                return []
            else:
                logger.warning(f"Unexpected skill input type: {type(skill_input)}")
                return []
        except Exception as e:
            logger.error(f"Error normalizing skills: {str(e)}")
            return []
    
    # Function for candidate recommendation
    @st.cache_data
    def recommend_candidates(df, desired_skills, min_experience=0, cert_required=None, top_n=5):
        try:
            # Dynamically find the email column (case-insensitive search)
            email_col = None
            for col in df.columns:
                if 'email' in col.lower():
                    email_col = col
                    break
            
            if email_col:
                df = df.rename(columns={email_col: "Email"})
            else:
                logger.warning("No email column found. Adding a default 'Email' column with 'N/A'.")
                df["Email"] = "N/A"
            
            # Rename other columns to match expected internal names
            df = df.rename(columns={
                "Resume | Name": "Name",
                "Experienc": "Experience (Years)",
                "Certificati": "Certifications",
                "Salary Exp": "Salary Expectation ($)",
                "Projects C": "Projects Count"
            })
            
            # Validate input DataFrame
            required_columns = ["Name", "Email", "Job Role", "Skills", "Certifications",
                              "Education", "Experience (Years)", "Projects Count", "Salary Expectation ($)"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                st.error(f"Missing required columns in resume data: {missing_columns}")
                return pd.DataFrame()
            
            # Ensure DataFrame is not empty
            if df.empty:
                logger.error("Input DataFrame is empty")
                st.error("Resume data is empty. Please upload valid data.")
                return pd.DataFrame()
            
            # Convert experience to numeric, handling potential string inputs
            df["Experience (Years)"] = pd.to_numeric(df["Experience (Years)"], errors='coerce').fillna(0).astype(int)
            
            df_copy = df.copy()
            desired_skills = [skill.lower().strip() for skill in desired_skills if skill and isinstance(skill, str) and skill.strip()]
            if not desired_skills:
                logger.error("No valid desired skills provided")
                st.error("No valid skills provided. Please enter at least one skill.")
                return pd.DataFrame()
            
            recommendations = []
            
            for idx, row in df_copy.iterrows():
                try:
                    candidate_skills = normalize_skills(row["Skills"])
                    skill_match_count = len(set(desired_skills).intersection(candidate_skills))
                    skill_coverage = skill_match_count / len(desired_skills) if desired_skills else 0
                    
                    # Validate experience and projects
                    experience = row["Experience (Years)"]
                    projects = row["Projects Count"]
                    if not isinstance(experience, (int, float)) or pd.isna(experience):
                        logger.warning(f"Invalid experience value for {row['Name']}: {experience}")
                        experience = 0
                    if not isinstance(projects, (int, float)) or pd.isna(projects):
                        logger.warning(f"Invalid projects value for {row['Name']}: {projects}")
                        projects = 0
                    
                    experience_score = np.clip(experience / 10, 0, 1)
                    project_score = np.clip(projects / 10, 0, 1)
                    
                    cert_score = 0
                    if cert_required and isinstance(row["Certifications"], str) and pd.notna(row["Certifications"]):
                        cert_required_lower = cert_required.lower()
                        if cert_required_lower in row["Certifications"].lower():
                            cert_score = 1
                    
                    total_score = (skill_coverage * 0.5) + (experience_score * 0.2) + (project_score * 0.2) + (cert_score * 0.1)
                    
                    if experience >= min_experience:
                        recommendations.append({
                            "Name": row["Name"] if pd.notna(row["Name"]) else "Unknown",
                            "Email": row["Email"] if pd.notna(row["Email"]) else "N/A",
                            "Job Role": row["Job Role"] if pd.notna(row["Job Role"]) else "N/A",
                            "Skills": row["Skills"] if pd.notna(row["Skills"]) else [],
                            "Certifications": row["Certifications"] if pd.notna(row["Certifications"]) else "None",
                            "Education": row["Education"] if pd.notna(row["Education"]) else "N/A",
                            "Experience (Years)": experience,
                            "Projects Count": projects,
                            "Score": round(total_score, 3),
                            "Salary Expectation ($)": row["Salary Expectation ($)"] if pd.notna(row["Salary Expectation ($)"]) else 0
                        })
                except Exception as e:
                    logger.error(f"Error processing candidate {row.get('Name', 'Unknown')} at index {idx}: {str(e)}")
                    continue
            
            if not recommendations:
                logger.error("No candidates meet the criteria")
                st.warning("No candidates match your criteria. Please adjust your filters.")
                return pd.DataFrame()
            
            recommendations_sorted = sorted(recommendations, key=lambda x: x["Score"], reverse=True)[:top_n]
            return pd.DataFrame(recommendations_sorted)
        except Exception as e:
            logger.error(f"Error in recommend_candidates: {str(e)}")
            st.error(f"Error generating recommendations: {str(e)}")
            return pd.DataFrame()
    
    # Email Configuration in Sidebar
    with st.sidebar.expander("📧 Email Configuration"):
        if 'email_configured' not in st.session_state:
            st.session_state['email_configured'] = False
            
        smtp_server = st.text_input("SMTP Server", "smtp.example.com")
        smtp_port = st.number_input("SMTP Port", 1, 65535, 587)
        sender_email = st.text_input("Sender Email", "recruitment@yourcompany.com")
        sender_password = st.text_input("Password", type="password")
        
        if st.button("Save Email Configuration"):
            if not all([smtp_server, smtp_port, sender_email, sender_password]):
                st.error("Please fill in all email configuration fields.")
            else:
                st.session_state['smtp_server'] = smtp_server
                st.session_state['smtp_port'] = smtp_port
                st.session_state['sender_email'] = sender_email
                st.session_state['sender_password'] = sender_password
                st.session_state['email_configured'] = True
                st.success("Email configuration saved!")
    
    # Step 1: Load resume data
    st.header("Step 1: Upload Resume Data")
    
    uploaded_file = st.file_uploader("Upload a CSV file with resume data", type=["csv"])
    use_sample_data = st.checkbox("Use sample data instead")
    
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            logger.error(f"CSV file reading error: {str(e)}")
    elif use_sample_data:
        df = load_sample_resume_data()
        st.success("Sample data loaded successfully!")
    
    if df is not None:
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        st.header("Step 2: Set Recommendation Criteria")
        
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                skills_input = st.text_input("Required Skills (comma-separated)", "Python, Machine Learning, NLP")
                min_experience = st.slider("Minimum Experience (Years)", 0, 10, 2)
            with col2:
                cert_required = st.text_input("Required Certification (optional)", "AWS")
                top_n = st.slider("Number of candidates to recommend", 1, 10, 5)
        
        with st.expander("Company Information for Emails"):
            company_name = st.text_input("Company Name", "TechInnovate Solutions")
            job_position = st.text_input("Job Position", "Senior Data Scientist")
        
        if st.button("Find Best Candidates"):
            if not skills_input.strip():
                st.error("Please provide at least one skill.")
            else:
                desired_skills = [skill.strip() for skill in skills_input.split(',') if skill.strip()]
                if not desired_skills:
                    st.error("No valid skills provided.")
                else:
                    with st.spinner("Finding the best candidates..."):
                        recommendations = recommend_candidates(
                            df, desired_skills, min_experience, cert_required if cert_required else None, top_n
                        )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} candidates matching your criteria!")
                        st.session_state['recommendations'] = recommendations
                        
                        st.write("Recommended Candidates:")
                        st.dataframe(recommendations)
                        
                        st.header("Visualizations")
                        tab1, tab2 = st.tabs(["Candidate Scores", "Experience vs Salary"])
                        
                        with tab1:
                            fig1 = px.bar(
                                recommendations,
                                x="Name",
                                y="Score",
                                color="Job Role",
                                text="Score",
                                title="Candidate Match Scores",
                                template="plotly_white"
                            )
                            fig1.update_layout(yaxis_title="Match Score", xaxis_title="Candidate", height=400)
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with tab2:
                            fig2 = px.scatter(
                                recommendations,
                                x="Experience (Years)",
                                y="Salary Expectation ($)",
                                text="Name",
                                color="Job Role",
                                size="Score",
                                title="Salary vs Experience for Top Candidates",
                                template="plotly_white"
                            )
                            fig2.update_traces(textposition='top center')
                            fig2.update_layout(height=400)
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        if len(recommendations) > 0:
                            st.header("Top Candidate Profile")
                            best_candidate = recommendations.iloc[0]
                            
                            with st.container():
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader(best_candidate["Name"])
                                    st.write(f"**Job Role:** {best_candidate['Job Role']}")
                                    st.write(f"**Education:** {best_candidate['Education']}")
                                    st.write(f"**Certifications:** {best_candidate['Certifications']}")
                                    st.write(f"**Email:** {best_candidate['Email']}")
                                
                                with col2:
                                    st.metric("Experience", f"{best_candidate['Experience (Years)']} years")
                                    st.metric("Projects", f"{best_candidate['Projects Count']}")
                                    st.metric("Match Score", f"{best_candidate['Score']:.3f}")
                                    st.metric("Salary Expectation", f"${best_candidate['Salary Expectation ($)']:,}")
                            
                            skills = normalize_skills(best_candidate["Skills"])
                            skill_match = [skill for skill in skills if skill in [s.lower() for s in desired_skills]]
                            skill_nomatch = [skill for skill in skills if skill not in [s.lower() for s in desired_skills]]
                            
                            st.write("**Skills:**")
                            with st.container():
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("Matching Skills:")
                                    for skill in skill_match:
                                        st.success(skill)
                                
                                with col2:
                                    st.write("Additional Skills:")
                                    for skill in skill_nomatch:
                                        st.info(skill)
                            
                            st.header("Contact Candidate")
                            
                            email_subject = st.text_input("Email Subject", f"Interview Opportunity with {company_name}")
                            default_email_body = f"""Dear {best_candidate["Name"]},

We are pleased to inform you that your resume has been selected for the {job_position} position at {company_name}. We would like to invite you for an interview.

Please let us know your availability for the coming week.

Best regards,
Recruitment Team
{company_name}
"""
                            email_body = st.text_area("Email Content", default_email_body, height=200)
                            
                            if st.button("Send Invitation Email"):
                                if not email_subject.strip() or not email_body.strip():
                                    st.error("Please provide email subject and content.")
                                else:
                                    with st.spinner(f"Sending email to {best_candidate['Name']}..."):
                                        success = send_acceptance_email(
                                            best_candidate["Name"],
                                            best_candidate["Email"],
                                            company_name,
                                            job_position,
                                            email_subject,
                                            email_body
                                        )
                                    
                                    if success:
                                        st.success(f"Email sent successfully to {best_candidate['Name']} at {best_candidate['Email']}!")
                        
                        if 'recommendations' in st.session_state and len(st.session_state['recommendations']) > 1:
                            st.header("Step 4: Bulk Actions")
                            
                            selected_candidates = st.multiselect(
                                "Select candidates for bulk actions",
                                st.session_state['recommendations']["Name"].tolist()
                            )
                            
                            if selected_candidates:
                                bulk_action = st.radio(
                                    "Choose bulk action",
                                    ["Send invitation emails", "Export selected profiles"]
                                )
                                
                                if bulk_action == "Send invitation emails" and st.button("Process Bulk Action"):
                                    if not email_subject.strip() or not email_body.strip():
                                        st.error("Please provide email subject and content.")
                                    else:
                                        sent_count = 0
                                        with st.spinner("Processing bulk emails..."):
                                            for candidate_name in selected_candidates:
                                                candidate_data = st.session_state['recommendations'][
                                                    st.session_state['recommendations']["Name"] == candidate_name
                                                ].iloc[0]
                                                success = send_acceptance_email(
                                                    candidate_data["Name"],
                                                    candidate_data["Email"],
                                                    company_name,
                                                    job_position,
                                                    email_subject,
                                                    email_body
                                                )
                                                if success:
                                                    sent_count += 1
                                        st.success(f"Processed {sent_count} bulk emails.")
                                
                                elif bulk_action == "Export selected profiles" and st.button("Export Profiles"):
                                    export_df = st.session_state['recommendations'][
                                        st.session_state['recommendations']["Name"].isin(selected_candidates)
                                    ]
                                    csv = export_df.to_csv(index=False)
                                    st.download_button(
                                        "Download CSV",
                                        csv,
                                        "selected_candidates.csv",
                                        "text/csv",
                                        key='download-csv'
                                    )
                                    st.success(f"Exported {len(export_df)} candidate profiles")
                    else:
                        st.warning("No candidates match your criteria. Please adjust your filters.")
        
        with st.expander("📝 Provide Feedback"):
            st.write("Help us improve our recommendation system:")
            feedback = st.text_area("Your feedback", "", max_chars=500)
            rating = st.slider("Rate your experience", 1, 5, 3)
            
            if st.button("Submit Feedback"):
                if feedback.strip():
                    try:
                        with open("feedback.csv", "a") as f:
                            f.write(f"{feedback},{rating},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        st.success("Thank you for your feedback!")
                    except Exception as e:
                        st.error(f"Error saving feedback: {str(e)}")
                        logger.error(f"Feedback saving error: {str(e)}")
                else:
                    st.error("Please provide feedback text.")
    else:
        st.info("Please upload resume data or use sample data to continue.")

# Footer
st.markdown("---")
