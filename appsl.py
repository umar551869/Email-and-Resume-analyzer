import streamlit as st
import pandas as pd
import numpy as np
import os
import string
import zipfile
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from datetime import datetime, timedelta
import base64

# Set page config
st.set_page_config(page_title="Email Classifier & Resume Analyzer", 
                   page_icon="📧", 
                   layout="wide")

# Initialize session state variables
if 'sent_emails' not in st.session_state:
    st.session_state['sent_emails'] = []

# Create sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "Email Spam Classifier", "Resume Analyzer", "Email History"])

# Display sent emails count if any
if st.session_state['sent_emails']:
    st.sidebar.markdown(f"**Sent Emails:** {len(st.session_state['sent_emails'])}")



# Home page
if app_mode == "Home":
    st.title("Email Classifier & Resume Analyzer")
    
    st.markdown("""
    Welcome to this multi-function application! This app has three main features:
    
    ### 1. Email Spam Classifier
    Upload a zip file of emails to classify them as spam or ham using SVM.
    
    ### 2. Resume Analyzer
    Upload a CSV file of resumes to analyze and find the best candidates based on skills, experience, and certifications.
    Send acceptance emails to selected candidates directly from the app.
    
    ### 3. Email History
    View a log of all sent emails including recipient and timestamp information.
    
    Select an option from the sidebar to get started!
    """)
    
    st.image("https://via.placeholder.com/800x400.png?text=Email+and+Resume+Analysis", use_column_width=True)

# Email Spam Classifier
elif app_mode == "Email Spam Classifier":
    st.title("Email Spam Classifier")
    
    st.markdown("""
    This tool classifies emails as spam or ham using Support Vector Machines (SVM).
    
    ### How it works:
    1. Upload a zip file containing emails
    2. The system will process the emails and extract features
    3. An SVM model will be trained to classify the emails
    4. Results will be displayed with performance metrics
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a zip file containing emails", type="zip")
    
    if uploaded_file is not None:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create directories if they don't exist
        if not os.path.exists("temp"):
            os.makedirs("temp")
            
        # Save the uploaded zip file
        with open("temp/emails.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        status_text.text("Extracting emails...")
        progress_bar.progress(10)
        
        # Extract the zip file
        extract_dir = "temp/emails/"
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            
        with zipfile.ZipFile("temp/emails.zip", 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        status_text.text("Download required NLTK data...")
        progress_bar.progress(20)
        
        # Download required NLTK data files if not already available
        try:
            nltk.download('stopwords')
            nltk.download('wordnet')
        except Exception as e:
            st.error(f"Error downloading NLTK data: {e}")
            
        # Function to clean up the text
        @st.cache_data
        def text_cleanup(text):
            # Remove punctuation
            text_no_punct = ''.join([c for c in text if c not in string.punctuation])
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            words = text_no_punct.split()
            words_filtered = [word.lower() for word in words if word.lower() not in stop_words]
            return words_filtered
        
        status_text.text("Processing emails and creating word list...")
        progress_bar.progress(30)
        
        # Process emails and create word list
        lmtzr = WordNetLemmatizer()
        word_count = {}
        processed_files = 0
        
        directory_path = os.path.join("temp", "emails", "emails")
        if not os.path.exists(directory_path):
            directory_path = os.path.join("temp", "emails")
        
        start_time = time.time()
        
        # Iterate through files
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                        words = text_cleanup(file.read())
                        for word in words:
                            if not word.isdigit() and len(word) > 2:
                                lemma = lmtzr.lemmatize(word)
                                word_count[lemma] = word_count.get(lemma, 0) + 1
                except Exception as e:
                    pass
                
                processed_files += 1
                if processed_files % 20 == 0:
                    status_text.text(f"Processed {processed_files} files.")
                    progress_bar.progress(30 + min(40, int(processed_files / 5)))
        
        # Sort words by frequency
        sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        
        # Write frequent words (count >= 100) to CSV
        with open("temp/wordslist.csv", "w", encoding="utf-8") as out_file:
            out_file.write("word,count\n")
            for word, freq in sorted_word_count:
                if freq < 100:
                    break
                out_file.write(f"{word},{freq}\n")
        
        status_text.text("Creating frequency vectors...")
        progress_bar.progress(75)
        
        # Load words from previously created wordslist.csv
        df = pd.read_csv('temp/wordslist.csv')
        words = df['word']
        word_list = words.tolist()  # Convert to regular list for efficiency
        
        # Convert all elements in word_list to strings
        word_list = [str(word) for word in word_list]
        
        # Initialize lemmatizer and stopwords
        lmtzr = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Prepare the output CSV with header
        with open("temp/frequency.csv", "w", encoding='utf-8') as f_out:
            f_out.write(','.join(word_list) + ',output\n')
        
        # Process each file
        file_count = 0
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            if not os.path.isfile(filepath):
                continue
            
            word_vector = np.zeros(len(word_list), dtype=int)
            
            with open(filepath, "r", encoding='utf-8', errors='ignore') as file:
                for word in file.read().split():
                    word = word.lower().strip(string.punctuation)
                    if word in stop_words or len(word) <= 2 or word.isdigit():
                        continue
                    word = lmtzr.lemmatize(word)
                    if word in word_list:
                        index = word_list.index(word)
                        word_vector[index] += 1
            
            # Assign output label based on file name (assuming 'spam' in filename indicates spam)
            if 'spam' in filename.lower():
                label = -1  # Spam
            else:
                label = 1   # Ham
            
            # Write the vector and label to CSV
            with open("temp/frequency.csv", "a", encoding='utf-8') as f_out:
                f_out.write(','.join(map(str, word_vector)) + f',{label}\n')
            
            file_count += 1
            if file_count % 20 == 0:
                status_text.text(f"Processed {file_count} files for frequency analysis")
        
        status_text.text("Training and evaluating SVM model...")
        progress_bar.progress(90)
        
        # Load Data
        df2 = pd.read_csv('temp/frequency.csv')
        
        X = df2.iloc[:, :-1].values
        Y = df2.iloc[:, -1].values.ravel()  # flatten to 1D array
        
        # Split data (70% train, 30% test)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        def evaluate_model(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
            precision = precision_score(y_true, y_pred, pos_label=1)
            recall = recall_score(y_true, y_pred, pos_label=1)
            return cm, precision, recall
        
        # Train and evaluate model
        model = SVC(kernel='linear', C=0.1)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        cm, precision, recall = evaluate_model(Y_test, predictions)
        
        status_text.text("Analysis complete!")
        progress_bar.progress(100)
        
        # Display results
        st.success("Email Classification Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(cm, index=['Ham (1)', 'Spam (-1)'], columns=['Ham (1)', 'Spam (-1)'])
            st.dataframe(cm_df)
            st.text(f"Total emails processed: {processed_files}")
            st.text(f"Time taken: {round(time.time() - start_time, 2)} seconds")
        
        with col2:
            st.subheader("Performance Metrics")
            st.metric("Precision", f"{round(precision * 100, 2)}%")
            st.metric("Recall", f"{round(recall * 100, 2)}%")
            st.metric("Accuracy", f"{round(((cm[0][0] + cm[1][1]) / sum(sum(cm))) * 100, 2)}%")
        
        # Create a pie chart showing spam vs ham distribution
        labels = ['Ham', 'Spam']
        values = [sum(Y == 1), sum(Y == -1)]
        
        fig = px.pie(
            values=values,
            names=labels,
            title="Distribution of Emails",
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        
        st.plotly_chart(fig)
        
        # Feature importance
        feature_weights = model.coef_[0]
        feature_names = word_list
        
        # Get top 10 most important features for spam and ham
        top_spam_indices = feature_weights.argsort()[:10]
        top_ham_indices = feature_weights.argsort()[-10:][::-1]
        
        top_spam_features = [(feature_names[i], feature_weights[i]) for i in top_spam_indices]
        top_ham_features = [(feature_names[i], feature_weights[i]) for i in top_ham_indices]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Ham Indicators")
            ham_df = pd.DataFrame(top_ham_features, columns=['Word', 'Weight'])
            fig = px.bar(ham_df, x='Word', y='Weight', title="Top Ham Indicators",
                         color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Top 10 Spam Indicators")
            spam_df = pd.DataFrame(top_spam_features, columns=['Word', 'Weight'])
            fig = px.bar(spam_df, x='Word', y='Weight', title="Top Spam Indicators",
                         color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig)

# Resume Analyzer
elif app_mode == "Resume Analyzer":
    st.title("Resume Analyzer")
    
    st.markdown("""
    This tool analyzes resumes to find the best candidates based on skills, experience, and certifications.
    
    ### How it works:
    1. Upload a CSV file containing resume data
    2. Set your criteria for candidate selection
    3. The system will analyze the resumes and recommend the best candidates
    4. Results will be displayed with visualizations
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file containing resume data", type="csv")
    
    # Sample data toggle
    use_sample_data = st.checkbox("Use sample data instead")
    
    if uploaded_file is not None or use_sample_data:
        # Load data
        if use_sample_data:
            # Create sample data
            data = {
                'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Williams', 'Tom Brown', 
                         'Sara Davis', 'Mike Wilson', 'Emily Jones', 'David Miller', 'Susan Anderson'],
                'Job Role': ['Data Scientist', 'ML Engineer', 'Data Analyst', 'Data Scientist', 'ML Engineer',
                             'Data Engineer', 'Data Scientist', 'ML Engineer', 'Data Analyst', 'Data Engineer'],
                'Experience (Years)': [5, 3, 2, 7, 4, 6, 8, 2, 5, 10],
                'Skills': ['Python, TensorFlow, NLP, SQL', 'Python, PyTorch, Computer Vision, AWS', 
                           'SQL, Excel, Python, Tableau', 'Python, R, NLP, TensorFlow, AWS', 'Python, TensorFlow, Computer Vision',
                           'Python, SQL, ETL, AWS, Spark', 'Python, R, Statistics, ML, AWS', 'Python, PyTorch, Docker, AWS', 
                           'SQL, Python, PowerBI, Excel', 'Python, Spark, Hadoop, AWS, Kafka'],
                'Certifications': ['AWS Certified ML', 'GCP Professional ML Engineer', 'None', 'AWS Certified ML', 'None',
                                   'AWS Certified Data Engineer', 'None', 'Azure AI Engineer', 'None', 'AWS Certified Big Data'],
                'Projects Count': [8, 5, 3, 10, 6, 9, 12, 4, 7, 15],
                'Salary Expectation ($)': [110000, 95000, 75000, 125000, 100000, 105000, 130000, 90000, 85000, 140000]
            }
            df = pd.DataFrame(data)
        else:
            # Load the uploaded file
            df = pd.read_csv(uploaded_file)
            # Check if required columns exist
            required_columns = ['Name', 'Job Role', 'Experience (Years)', 'Skills', 'Certifications', 
                                'Projects Count', 'Salary Expectation ($)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"The uploaded CSV is missing these required columns: {', '.join(missing_columns)}")
                st.stop()
        
        # Clean the data
        # Normalize skill names
        def normalize_skills(skill_str):
            if pd.isnull(skill_str):
                return []
            skills = [s.strip().lower() for s in skill_str.split(',')]
            return list(set(skills))  # remove duplicates
        
        df['Skills'] = df['Skills'].apply(normalize_skills)
        
        # Handle missing or "None" certifications
        df['Certifications'] = df['Certifications'].replace(['None', 'none', '', np.nan], 'No Certification')
        
        # Optional: Normalize Education field if it exists
        if 'Education' in df.columns:
            df['Education'] = df['Education'].str.strip().str.upper()
        
        # Display the data
        with st.expander("View Resume Data"):
            st.dataframe(df)
        
        st.subheader("Candidate Search Criteria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Multi-select for desired skills
            all_skills = []
            for skills_list in df['Skills']:
                all_skills.extend(skills_list)
            unique_skills = list(set(all_skills))
            
            desired_skills = st.multiselect(
                "Select desired skills", 
                options=unique_skills,
                default=unique_skills[:3] if len(unique_skills) >= 3 else unique_skills
            )
            
            # Select minimum experience
            min_experience = st.slider("Minimum experience (years)", 0, 15, 2)
        
        with col2:
            # Certification requirement
            all_certs = df['Certifications'].unique().tolist()
            cert_required = st.selectbox("Required certification", ["None"] + all_certs)
            cert_required = None if cert_required == "None" else cert_required
            
            # Number of candidates to show
            top_n = st.slider("Number of candidates to show", 1, 10, 5)
        
        # Advanced recommendation function
        def advanced_recommendation(df, desired_skills, min_experience=0, cert_required=None, top_n=5):
            desired_skills = [skill.lower().strip() for skill in desired_skills]
            
            recommendations = []
            
            for _, row in df.iterrows():
                candidate_skills = row["Skills"]
                skill_match_count = len(set(desired_skills).intersection(candidate_skills))
                skill_coverage = skill_match_count / len(desired_skills) if desired_skills else 0
                
                experience_score = np.clip(row["Experience (Years)"] / 10, 0, 1)
                project_score = np.clip(row["Projects Count"] / 10, 0, 1)
                cert_score = 1 if cert_required and cert_required.lower() in row["Certifications"].lower() else 0
                
                total_score = (skill_coverage * 0.5) + (experience_score * 0.2) + (project_score * 0.2) + (cert_score * 0.1)
                
                if row["Experience (Years)"] >= min_experience:
                    recommendations.append({
                        "Name": row["Name"],
                        "Job Role": row["Job Role"],
                        "Skills": row["Skills"],
                        "Certifications": row["Certifications"],
                        "Experience (Years)": row["Experience (Years)"],
                        "Projects Count": row["Projects Count"],
                        "Score": round(total_score, 3),
                        "Salary Expectation ($)": row["Salary Expectation ($)"]
                    })
            
            top_candidates = sorted(recommendations, key=lambda x: x["Score"], reverse=True)[:top_n]
            return pd.DataFrame(top_candidates)
        
        if st.button("Find Top Candidates"):
            # Get top candidates from recommendation system
            with st.spinner("Analyzing resumes..."):
                top_candidates_df = advanced_recommendation(
                    df,
                    desired_skills=desired_skills,
                    min_experience=min_experience,
                    cert_required=cert_required,
                    top_n=top_n
                )
                
                if top_candidates_df.empty:
                    st.error("No candidates match your criteria. Try relaxing some requirements.")
                else:
                    st.success(f"Found {len(top_candidates_df)} matching candidates!")
                    
                    # Display top candidates
                    st.subheader("🏆 Top Recommended Candidates")
                    
                    # Format the Skills column for display
                    display_df = top_candidates_df.copy()
                    display_df['Skills'] = display_df['Skills'].apply(lambda x: ", ".join(x))
                    
                    st.dataframe(
                        display_df.style.background_gradient(subset=['Score'], cmap='viridis')
                    )
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart of scores
                        fig1 = px.bar(
                            top_candidates_df,
                            x="Name",
                            y="Score",
                            color="Job Role",
                            text="Score",
                            title="Top Candidates by Match Score",
                            template="plotly_white"
                        )
                        fig1.update_layout(yaxis_title="Match Score", xaxis_title="Candidate")
                        st.plotly_chart(fig1)
                    
                    with col2:
                        # Scatter plot of salary vs experience
                        fig2 = px.scatter(
                            top_candidates_df,
                            x="Experience (Years)",
                            y="Salary Expectation ($)",
                            text="Name",
                            color="Job Role",
                            size="Score",
                            title="Salary vs Experience for Top Candidates",
                            template="plotly_white"
                        )
                        fig2.update_traces(textposition='top center')
                        st.plotly_chart(fig2)
                    
                    # Display the best candidate
                    best_candidate = top_candidates_df.iloc[0]
                    
                    st.subheader("🌟 Best Match")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        ### {best_candidate['Name']}
                        **Job Role:** {best_candidate['Job Role']}  
                        **Experience:** {best_candidate['Experience (Years)']} years  
                        **Projects:** {best_candidate['Projects Count']}  
                        **Match Score:** {best_candidate['Score']}  
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **Certifications:** {best_candidate['Certifications']}  
                        **Salary Expectation:** ${best_candidate['Salary Expectation ($)']:,}  
                        **Skills:** {", ".join(best_candidate['Skills'])}  
                        """)
                    
                    # Skill match visualization
                    st.subheader("Skill Match Analysis")
                    
                    for i, candidate in top_candidates_df.head(3).iterrows():
                        candidate_skills = candidate['Skills']
                        matched_skills = [s for s in desired_skills if s in candidate_skills]
                        missing_skills = [s for s in desired_skills if s not in candidate_skills]
                        
                        skill_match_df = pd.DataFrame({
                            'Skill': matched_skills + missing_skills,
                            'Status': ['Matched'] * len(matched_skills) + ['Missing'] * len(missing_skills)
                        })
                        
                        fig = px.bar(
                            skill_match_df,
                            x='Skill',
                            color='Status',
                            title=f"Skill Analysis: {candidate['Name']}",
                            color_discrete_map={'Matched': '#2ecc71', 'Missing': '#e74c3c'}
                        )
                        
                        st.plotly_chart(fig)
                    
                    # Email Sending Functionality
                    st.subheader("📧 Send Acceptance Email")
                    
                    with st.expander("Compose Email to Top Candidate"):
                        # Select candidate to email
                        email_candidate = st.selectbox(
                            "Select candidate to send email to:",
                            options=top_candidates_df['Name'].tolist(),
                            index=0  # Default to top candidate
                        )
                        
                        # Get selected candidate info
                        candidate_info = top_candidates_df[top_candidates_df['Name'] == email_candidate].iloc[0]
                        
                        # Email fields
                        col1, col2 = st.columns(2)
                        with col1:
                            email_to = st.text_input("To:", value=f"{email_candidate.lower().replace(' ', '.')}@example.com")
                            email_cc = st.text_input("CC:", value="hr@yourcompany.com")
                        with col2:
                            email_subject = st.text_input("Subject:", value="Congratulations! Your application has been accepted")
                        
                        # Generate email content based on candidate info
                        default_email_content = f"""Dear {email_candidate},

I am pleased to inform you that your application for the {candidate_info['Job Role']} position at Our Company has been accepted. Your experience of {candidate_info['Experience (Years)']} years and skills in {', '.join(candidate_info['Skills'][:3])} make you an excellent fit for our team.

We would like to invite you for a final interview to discuss the role in more detail. Please let us know your availability for next week.

Your offered salary would be ${candidate_info['Salary Expectation ($)']:,} per year.

Best regards,
Hiring Manager
Our Company
"""
                        
                        email_content = st.text_area("Email Content:", value=default_email_content, height=300)
                        
                        # Preview email before sending
                        if st.button("Preview Email"):
                            st.subheader("Email Preview")
                            email_preview = f"""
**To:** {email_to}
**CC:** {email_cc}
**Subject:** {email_subject}

{email_content}
                            """
                            st.markdown(email_preview)
                            
                            # Add confirmation before sending
                            send_confirmation = st.checkbox("I confirm this email is ready to send")
                            
                            # Email sending simulation with confirmation
                            if send_confirmation and st.button("Send Email", key="send_email_button"):
                                with st.spinner("Sending email..."):
                                    # Simulate email sending delay
                                    import time
                                    time.sleep(2)
                                    
                                    st.success(f"✅ Email successfully sent to {email_to}!")
                                    
                                    # Display sent email details
                                    st.code(f"""
To: {email_to}
CC: {email_cc}
Subject: {email_subject}

{email_content}
                                    """)
                                    
                                    # Log the email in the session state
                                    st.session_state['sent_emails'] = st.session_state.get('sent_emails', []) + [
                                        {
                                            'to': email_to,
                                            'subject': email_subject,
                                            'candidate': email_candidate,
                                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                                        }
                                    ]
                                    
                                    # Provide option to create calendar invite
                                    st.markdown("### 📅 Schedule Interview")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        interview_date = st.date_input("Interview Date", value=pd.to_datetime("today") + pd.Timedelta(days=7))
                                    
                                    with col2:
                                        interview_time = st.time_input("Interview Time", value=pd.to_datetime("10:00").time())
                                    
                                    interview_location = st.text_input("Interview Location", value="Conference Room A, 12th Floor")
                                    
                                    if st.button("Generate Calendar Invite"):
                                        st.markdown("#### Calendar Invite Details")
                                        
                                        # Format date and time for calendar
                                        full_date = f"{interview_date.strftime('%Y-%m-%d')} {interview_time.strftime('%H:%M')}"
                                        
                                        calendar_details = f"""
Subject: Interview with {email_candidate} for {candidate_info['Job Role']} Position
Date: {interview_date.strftime('%A, %B %d, %Y')}
Time: {interview_time.strftime('%I:%M %p')}
Location: {interview_location}
Attendees: {email_candidate}, Hiring Manager, HR Representative
                                        """
                                        
                                        st.code(calendar_details)
                                        
                                        # Generate downloadable .ics file
                                        st.markdown("#### Download Calendar File")
                                        
                                        # Create actual .ics content
                                        start_dt = datetime.combine(interview_date, interview_time)
                                        end_dt = start_dt + timedelta(hours=1)  # 1-hour interview
                                        
                                        ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Company//Interview Calendar//EN
CALSCALE:GREGORIAN
BEGIN:VEVENT
DTSTART:{start_dt.strftime('%Y%m%dT%H%M%S')}
DTEND:{end_dt.strftime('%Y%m%dT%H%M%S')}
LOCATION:{interview_location}
SUMMARY:Interview with {email_candidate} for {candidate_info['Job Role']} Position
DESCRIPTION:Interview with candidate {email_candidate} for the {candidate_info['Job Role']} position.
STATUS:CONFIRMED
END:VEVENT
END:VCALENDAR
"""
                                        
                                        # Create a download button for the .ics file
                                        ics_bytes = ics_content.encode()
                                        b64 = base64.b64encode(ics_bytes).decode()
                                        href = f'<a href="data:text/calendar;base64,{b64}" download="interview_with_{email_candidate.replace(" ", "_")}.ics">📎 Download .ics calendar invite</a>'
                                        st.markdown(href, unsafe_allow_html=True)
                                        
                                        st.success("Calendar invite generated successfully!")
                            elif not st.button("Preview Email"):
                                st.info("Click 'Preview Email' to review your message before sending.")

# Email History Page
elif app_mode == "Email History":
    st.title("Email History")
    
    st.markdown("""
    This page shows a log of all emails sent through the Resume Analyzer.
    """)
    
    if not st.session_state.get('sent_emails', []):
        st.info("No emails have been sent yet. Use the Resume Analyzer to send emails to candidates.")
    else:
        # Display sent emails in a table
        emails_df = pd.DataFrame(st.session_state['sent_emails'])
        
        st.dataframe(
            emails_df,
            column_config={
                "to": "Recipient",
                "subject": "Subject",
                "candidate": "Candidate Name",
                "timestamp": "Sent Date & Time"
            },
            hide_index=True
        )
        
        # Add a chart to visualize email activity
        if len(emails_df) > 1:
            # Extract date from timestamp for grouping
            emails_df['date'] = emails_df['timestamp'].apply(lambda x: x.split(' ')[0])
            
            # Count emails by date
            email_counts = emails_df['date'].value_counts().reset_index()
            email_counts.columns = ['Date', 'Count']
            
            # Create a bar chart
            fig = px.bar(
                email_counts,
                x='Date',
                y='Count',
                title='Email Activity by Date',
                labels={'Count': 'Emails Sent', 'Date': 'Date'},
                color_discrete_sequence=['#3498db']
            )
            
            st.plotly_chart(fig)
        
        # Add option to clear history
        if st.button("Clear Email History"):
            st.session_state['sent_emails'] = []
            st.success("Email history cleared successfully!")
            st.experimental_rerun()