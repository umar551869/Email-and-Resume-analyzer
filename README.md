Email Classifier & Resume Analyzer
Overview
This is a Streamlit-based web application that provides two main functionalities: Email Spam Classifier and Resume Analyzer. The application leverages machine learning (Support Vector Machines) for email classification and a custom scoring system for resume analysis, with additional features like email sending and calendar invite generation.
Features
1. Email Spam Classifier

Functionality: Classifies emails as spam or ham using a Support Vector Machine (SVM) model.
Input: Accepts a zip file containing email text files.
Process:
Extracts and processes emails using NLTK for text cleaning (removing punctuation, stopwords, and lemmatization).
Builds a frequency-based feature vector for each email.
Trains an SVM model to classify emails as spam (-1) or ham (1).


Output:
Displays a confusion matrix, precision, recall, and accuracy metrics.
Visualizes spam vs. ham distribution with a pie chart.
Shows top 10 words indicative of spam and ham using bar charts.



2. Resume Analyzer

Functionality: Analyzes resumes to identify top candidates based on skills, experience, certifications, and project count.
Input: Accepts a CSV file with columns: Name, Job Role, Experience (Years), Skills, Certifications, Projects Count, Salary Expectation ($). Optionally, sample data can be used.
Process:
Normalizes skills and certifications for consistency.
Allows users to set criteria (desired skills, minimum experience, required certifications, number of candidates).
Scores candidates based on skill match, experience, projects, and certifications.


Output:
Displays a ranked list of top candidates with a match score.
Visualizes candidate scores, salary vs. experience, and skill match analysis using Plotly charts.
Provides email composition and sending functionality (simulated) for selected candidates.
Generates downloadable .ics calendar invites for scheduling interviews.



3. Email History

Functionality: Logs and displays all emails sent through the Resume Analyzer.
Output:
Shows a table with recipient, subject, candidate name, and timestamp.
Visualizes email activity by date with a bar chart.
Allows clearing of email history.



Installation
Prerequisites

Python 3.8+
Streamlit
Required Python packages (listed in requirements.txt)

Steps

Clone the repository:git clone <repository-url>
cd email-classifier-resume-analyzer


Install dependencies:pip install -r requirements.txt


Download required NLTK data:import nltk
nltk.download('stopwords')
nltk.download('wordnet')


Run the Streamlit app:streamlit run app.py



File Structure
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── temp/                   # Temporary directory for file processing
│   ├── emails/             # Extracted email files
│   ├── wordslist.csv       # Word frequency list
│   └── frequency.csv       # Email feature vectors
└── README.md               # Project documentation

Usage

Launch the App: Run streamlit run app.py to start the application.
Navigate: Use the sidebar to select between "Home", "Email Spam Classifier", "Resume Analyzer", or "Email History".
Email Spam Classifier:
Upload a zip file containing email text files.
View classification results, metrics, and visualizations.


Resume Analyzer:
Upload a CSV file or use sample data.
Set search criteria (skills, experience, certifications).
Review top candidates, send acceptance emails, and generate calendar invites.


Email History:
View sent email logs and activity charts.
Clear email history if needed.



Dependencies

streamlit: For the web interface
pandas, numpy: For data processing
scikit-learn: For SVM classification
nltk: For text processing
plotly: For interactive visualizations
zipfile, base64: For file handling and calendar invite generation

Notes

The email sending functionality is simulated and logs emails in the session state.
The application assumes email files in the zip have "spam" in the filename for spam emails.
Resume CSV files must include required columns for proper processing.
Temporary files are stored in the temp/ directory and should be cleaned periodically.

License
This project is licensed under the MIT License.
Acknowledgments

Built with Streamlit and Scikit-learn.
Uses NLTK for natural language processing.
Visualizations powered by Plotly.
