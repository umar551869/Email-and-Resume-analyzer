# Resume Analyzer & Candidate Recommendation System

A powerful Streamlit-based web application for intelligently analyzing, filtering, recommending, and managing job candidates based on resume data.

## Features

### Resume Analysis & Recommendation

* Upload a CSV of resumes with fields like Name, Skills, Experience, Job Role, Certifications, etc.
* Define desired skills, minimum experience, and required certifications.
* System scores and recommends the Top N Candidates using a weighted skill matching algorithm.
* Sorted recommendations with match scores.

### Interactive Visualizations

* Bar chart showing top candidate scores by job role.
* Scatter plot mapping experience vs salary expectations.
* Skill Match Analysis bar charts for top 3 candidates (matched vs missing skills).

### Best Match Section

* Highlights the top candidate with detailed profile info including salary, skills, and match score.

### Email Automation

* Compose and preview personalized email offers.
* Dynamically generate content using candidate data.
* Simulate sending email with confirmation.
* Optional interview scheduling with:

  * Date and Time input
  * Calendar .ics file generation and download

### Email History Log

* Logs every sent email with:

  * Recipient
  * Subject
  * Candidate name
  * Timestamp
* Visual email activity chart over time
* Option to clear history

## File Structure

```
resume-analyzer/
├── app.py                # Streamlit app logic
├── resume_data.csv       # Sample resume data (editable)
├── README.md             # Project documentation
```

## Requirements

* Python 3.8+
* Required packages:

  ```bash
  pip install streamlit pandas plotly
  ```

## How to Run

```bash
streamlit run app.py
```

## Sample Resume Data Format

| Name     | Skills                        | Experience (Years) | Certifications | Projects Count | Job Role       | Salary Expectation (\$) |
| -------- | ----------------------------- | ------------------ | -------------- | -------------- | -------------- | ----------------------- |
| John Doe | Python, SQL, Machine Learning | 3                  | AWS Certified  | 5              | Data Scientist | 60000                   |

Make sure Skills is a comma-separated list.

## Customization Ideas

* Add NLP resume parsing (PDF/DOCX)
* Connect to external mail APIs (Gmail, SMTP)
* Add authentication for admin users
* Export selected candidates to Excel
* Add multiple job role filtering

## License

MIT License — feel free to fork and use

## Acknowledgments

Developed for data-driven hiring and streamlining candidate evaluation processes.
