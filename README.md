NLP Applications Dashboard
Overview
This Streamlit-based web application provides two powerful NLP-driven tools: Email Spam Detection and Resume Recommendation System. The application leverages machine learning models (SVM and DistilBERT) for email classification and a custom scoring algorithm for resume analysis, with features like email sending, interactive visualizations, and bulk actions for candidate management.
Features
1. Email Spam Detection

Functionality: Classifies emails as spam or ham using either Support Vector Machines (SVM) with linear/polynomial kernels or a DistilBERT model.
Input: Accepts a zip file containing email text files or uses sample data.
Process:
Extracts frequent words using NLTK for text preprocessing (removing punctuation, stopwords, and lemmatization).
Creates frequency-based feature vectors for SVM models.
Preprocesses text with DistilBERT tokenizer for transformer-based classification.
Trains and evaluates models with customizable parameters (e.g., polynomial degree, minimum word frequency).


Output:
Displays confusion matrices, precision, recall, and accuracy metrics.
Visualizes word frequencies and model performance with Plotly charts.
Allows testing new emails with trained models and shows key words influencing predictions.
Includes an interactive analysis canvas for custom visualizations.



2. Resume Recommendation System

Functionality: Recommends top candidates based on skills, experience, certifications, and project count.
Input: Accepts a CSV file with columns like Name, Emails, Job Role, Skills, Certifications, Education, Experience, Projects Count, Salary Expectation, or uses sample data.
Process:
Normalizes skills and certifications for consistent scoring.
Allows users to set criteria (desired skills, minimum experience, certifications, number of candidates).
Scores candidates using a weighted formula (50% skills, 20% experience, 20% projects, 10% certifications).
Supports email configuration for sending personalized invitation emails via SMTP.


Output:
Displays ranked candidates with match scores.
Visualizes candidate scores and salary vs. experience with Plotly charts.
Shows detailed profiles for top candidates, including skill match analysis.
Enables bulk actions like sending emails or exporting candidate profiles.
Collects user feedback to improve the system.



Installation
Prerequisites

Python 3.8+
Streamlit
PyTorch and Transformers (for DistilBERT)
Required Python packages (listed in requirements.txt)

Steps

Clone the repository:git clone <repository-url>
cd nlp-applications-dashboard


Install dependencies:pip install -r requirements.txt


Download required NLTK data:import nltk
nltk.download('stopwords')
nltk.download('wordnet')


Run the Streamlit app:streamlit run app.py



File Structure
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── feedback.csv            # User feedback log (generated after feedback submission)
└── README.md               # Project documentation

Usage

Launch the App: Run streamlit run app.py to start the application.
Navigate: Use the sidebar to select between "Home", "Email Spam Detection", or "Resume Recommendation".
Email Spam Detection:
Upload a zip file with email texts or use sample data.
Process emails to extract features and train models (Linear SVM, Polynomial SVM, or DistilBERT).
View performance metrics, visualizations, and test new emails.


Resume Recommendation:
Upload a CSV file or use sample data.
Configure email settings in the sidebar (SMTP server, port, credentials).
Set criteria (skills, experience, certifications) and find top candidates.
View candidate profiles, send invitation emails, perform bulk actions, and provide feedback.


Feedback: Submit feedback and ratings to improve the recommendation system.

Dependencies

streamlit: Web interface
pandas, numpy: Data processing
scikit-learn: SVM models
nltk: Text preprocessing
torch, transformers: DistilBERT model
plotly: Interactive visualizations
smtplib: Email sending
tqdm: Progress bars
zipfile: ZIP file handling

Notes

Email Sending: Requires valid SMTP server settings. The application supports sending personalized emails but needs proper configuration.
DistilBERT: Optimized for speed with a shorter max_length (64), fewer epochs (1), and smaller batch size (8), which may slightly impact accuracy.
Sample Data: Provided for both email and resume functionalities to test the app without external data.
Error Handling: Comprehensive logging is implemented to track errors during file processing, model training, and email sending.
Resume CSV: Must include required columns; missing columns will trigger an error. Email columns are dynamically detected (case-insensitive).
Feedback: Stored in feedback.csv for later analysis.

Limitations

Email classification assumes filenames contain "spam" for spam emails.
DistilBERT runs on CPU to avoid CUDA issues, which may slow training for large datasets.
Email sending is SMTP-dependent and may fail with incorrect credentials or server issues.
Temporary files are not automatically cleaned; manual cleanup may be needed for large datasets.

License
This project is licensed under the MIT License.
Acknowledgments

Built with Streamlit, Scikit-learn, and Hugging Face Transformers.
Uses NLTK for text preprocessing and Plotly for visualizations.
Inspired by real-world NLP applications for spam detection and recruitment.
