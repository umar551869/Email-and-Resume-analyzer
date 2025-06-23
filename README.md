📧 Email Classifier & Resume Analyzer
A Streamlit-based web application that provides two key functionalities:

Email Spam Classifier using machine learning (Support Vector Machines).

Resume Analyzer for ranking candidates based on skills, experience, certifications, and more.

Additional features include email composition, calendar invite generation, and email history logging.

🔍 Features
1. Email Spam Classifier
Functionality: Classifies emails as Spam or Ham using an SVM model.

Input: Upload a .zip file containing .txt email files.

Processing:

Text cleaning using NLTK (stopword removal, punctuation stripping, lemmatization).

Frequency-based feature vector creation.

Model training using Support Vector Machine (SVM).

Output:

Confusion matrix, accuracy, precision, and recall.

Pie chart of spam vs. ham distribution.

Bar charts for top 10 spam and ham words.

2. Resume Analyzer
Functionality: Ranks resumes based on:

Skills

Experience

Certifications

Project Count

Input: Upload a .csv file with the following columns:

scss
Copy
Edit
Name, Job Role, Experience (Years), Skills, Certifications, Projects Count, Salary Expectation ($)
(Sample data also available in-app.)

Processing:

Normalizes skills and certifications.

Allows user-defined criteria for:

Desired skills

Minimum experience

Required certifications

Number of top candidates to show

Computes a custom match score per candidate.

Output:

Ranked list of candidates.

Interactive Plotly charts:

Candidate score visualization

Salary vs. Experience

Skill match analysis

Simulated email sending to selected candidates.

Downloadable .ics calendar invites for interview scheduling.

3. Email History
Functionality: Logs and displays emails sent via the Resume Analyzer.

Output:

Table with recipient, subject, candidate name, and timestamp.

Email activity chart (by date).

Option to clear history.

⚙️ Installation
✅ Prerequisites
Python 3.8+

Streamlit

📦 Required Packages
Install from requirements.txt:

bash
Copy
Edit
pip install -r requirements.txt
📥 Download NLTK Data
python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
🚀 Running the App
bash
Copy
Edit
streamlit run app.py
📁 File Structure
bash
Copy
Edit
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── temp/                   # Temporary directory for file processing
│   ├── emails/             # Extracted email files
│   ├── wordslist.csv       # Word frequency list
│   └── frequency.csv       # Email feature vectors
└── README.md               # Project documentation
🧠 Usage
📨 Email Spam Classifier
Go to "Email Spam Classifier" from the sidebar.

Upload a .zip file containing .txt email files.

View classification results and insights.

👩‍💼 Resume Analyzer
Go to "Resume Analyzer".

Upload a resume .csv file or use sample data.

Set filtering criteria (skills, experience, certifications).

View ranked candidates, send acceptance emails, and download interview invites.

📚 Email History
View previously sent emails.

See email activity charts.

Option to clear the history.

🛠 Dependencies
Library	Use
streamlit	Web app interface
pandas, numpy	Data manipulation
scikit-learn	Machine learning (SVM)
nltk	Natural Language Processing
plotly	Interactive visualizations
zipfile, base64	File handling & calendar invites

📌 Notes
The email sending functionality is simulated and logs emails locally.

Email files must include "spam" in the filename for spam classification.

Resume CSVs must contain all required columns.

Temporary files are saved in temp/ and should be periodically cleared.

📄 License
This project is licensed under the MIT License.

🙏 Acknowledgments
Built with ❤️ using Streamlit

Email classification using Scikit-learn

NLP powered by NLTK

Visualizations with Plotly

