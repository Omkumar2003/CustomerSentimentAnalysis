# Customer Sentiment Analysis

This project performs sentiment analysis on customer reviews using machine learning models. It includes a dashboard to visualize model performance and to analyze new input reviews in real-time.

---

## 📦 Sample Data

Download the sample CSV file from the following Mega link:  
🔗 [Download sample.csv](https://mega.nz/file/qBFnDRwA#tlx0EvqAfefzGkrYSSuoxNpKCfdJ09k9-sb3927-DqE)

Place the file in the root of your project directory.

---

## ⚙️ Setup Instructions

### 1. Create a Virtual Environment

In the project directory:

```bash
python -m venv venv
```
Activate it:

On Windows:

```bash
venv\Scripts\activate
```
On Linux/macOS:

```bash
source venv/bin/activate
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
### 3. Run the Jupyter Notebook
Open ```Sentiment_Analysis.ipynb`` and run all the cells.

**⚠️ Make sure you select the correct kernel that uses the environment where dependencies were installed.**

This will generate the following files:
```
sentiment_Model.pkl
vectorizer.pkl
```
### 4. Launch the App
Once the .pkl files are generated, run the app using:

```bash
python app.py
```
Visit your browser at:
👉 http://127.0.0.1:8050/

## 🖼️ Interface Preview
The web interface looks like the screenshot below:
![ss](https://github.com/Omkumar2003/CustomerSentimentAnalysis/blob/main/ss.png)

## 📁 Project Structure
```
SentimentAnalysis/
├── app.py
├── requirements.txt
├── sample.csv
├── Sentiment_Analysis.ipynb
├── sentiment_Model.pkl
├── vectorizer.pkl
├── ss.png
└── README.md
```
## 🧠 Features
**Preprocessing:** Tokenization, Stopword Removal, Lemmatization

**Models:** Logistic Regression, XGBoost, Random Forest, MultinomialNB

**Real-time sentiment prediction**

**Model performance dashboard (bar, line, heatmap)**
