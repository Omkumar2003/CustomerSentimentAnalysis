import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State # Import State
import joblib # For loading the model and vectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy # For lemmatization

# --- Pre-computation and Loading ---

# It's critical to load models and data ONCE when the app starts, not inside a callback.

# Download necessary NLTK data
# In a real deployment, this would be done during setup, but for local running, this is fine.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Load the trained model and vectorizer
# NOTE: Your notebook overwrites 'sentiment_Model.pkl' multiple times.
# The last model saved was 'lr_smote' (Logistic Regression with SMOTE), which is what will be loaded here.
model = joblib.load("sentiment_Model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load the spacy model for lemmatization (as used in the notebook)
nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])

# --- Replicating Preprocessing Functions from the Notebook ---

# This function must be IDENTICAL to the one in your notebook
def preprocess(document):
    document = document.lower()
    document = re.sub(r"[^\sA-z]", "", document)
    words = word_tokenize(document)
    words = [word for word in words if word not in stopwords.words("english")]
    words = [w for w in words if len(w) > 1]
    document = " ".join(words)
    return document

# This function must also be IDENTICAL to the one in your notebook
def lemmatize_text(text):
    sent = []
    doc = nlp(text)
    for token in doc:
        sent.append(token.lemma_)
    return " ".join(sent)

# --- Dashboard Data Preparation (from your original app.py) ---
# This part is fine as it's for the static dashboard visuals.
data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"],
    "LR Train": [0.95, 0.94, 1.00, 0.97, 0.65],
    "LR Test": [0.94, 0.94, 1.00, 0.97, 0.59],
    "LR SM Train": [0.98, 1.00, 0.96, 0.98, 0.98],
    "LR SM Test": [0.94, 0.99, 0.95, 0.97, 0.90],
    "MNB Train": [0.95, 0.99, 0.91, 0.95, 0.95],
    "MNB Test": [0.89, 0.98, 0.90, 0.94, 0.82],
    "XGB Train": [0.98, 1.00, 0.97, 0.98, 0.98],
    "XGB Test": [0.95, 0.99, 0.96, 0.97, 0.91],
    "RF Train": [1.00, 1.00, 1.00, 1.00, 1.00],
    "RF Test": [0.95, 0.97, 0.98, 0.97, 0.78],
}
# Note: I've updated the values to match the final table in your notebook for better consistency
# and removed the HP tuned models for clarity, as they generally performed worse.
metrics_df = pd.DataFrame(data)
metrics_df_long = metrics_df.melt(id_vars=["Metric"], var_name="Model", value_name="Score")

# Initialize Dash app
app = Dash(__name__)
server = app.server # For deployment

# --- App Layout ---
app.layout = html.Div([
    html.H1("Sentiment Analysis and Model Performance", style={'textAlign': 'center', 'color': '#2c3e50'}),

    html.Div([
        html.H2("Analyze a New Review", style={'textAlign': 'center', 'color': '#34495e'}),
        dcc.Textarea(
            id="user-input",
            placeholder="Type your review here...",
            style={"width": "60%", "height": "100px", "padding": "10px", "borderRadius": "5px", "border": "1px solid #ccc"}
        ),
        html.Br(),
        html.Button("Analyze Sentiment", id="analyze-button", n_clicks=0, style={"marginTop": "10px", "padding": "10px 20px", "fontSize": "16px", "cursor": "pointer"}),
    ], style={"textAlign": "center", "marginBottom": "20px", "padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px", "backgroundColor": "#f9f9f9"}),

    html.Div(id="sentiment-output", style={"textAlign": "center", "fontSize": "22px", "fontWeight": "bold", "marginBottom": "40px"}),

    html.H2("Machine Learning Model Performance Dashboard", style={'textAlign': 'center', 'color': '#34495e'}),
    
    html.Div([
        html.Div([
            html.Label("Select Model:", style={"marginRight": "10px"}),
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": model, "value": model} for model in metrics_df.columns[1:]],
                value="LR SM Test", # A better default
                clearable=False,
                style={"width": "300px", "display": "inline-block"}
            ),
        ], style={"display": "inline-block", "marginRight": "50px"}),
        
        html.Div([
            html.Label("Select Heatmap Color Scale:", style={"marginRight": "10px"}),
            dcc.Dropdown(
                id="color-scale-dropdown",
                options=[
                    {"label": "Viridis", "value": "Viridis"}, {"label": "Plasma", "value": "Plasma"},
                    {"label": "Blues", "value": "Blues"}, {"label": "Reds", "value": "Reds"},
                    {"label": "Cividis", "value": "Cividis"}
                ],
                value="Blues",
                clearable=False,
                style={"width": "300px", "display": "inline-block"}
            )
        ], style={"display": "inline-block"}),
        
    ], style={"textAlign": "center", "marginBottom": "20px"}),
    
    html.Div([
        dcc.Graph(id="bar-chart", style={'display': 'inline-block'}),
        dcc.Graph(id="line-chart", style={'display': 'inline-block'}),
    ], style={'textAlign': 'center'}),
    
    dcc.Graph(id="heatmap")
])

# --- Callbacks ---

# CALLBACK FOR SENTIMENT ANALYSIS (Corrected)
@app.callback(
    Output("sentiment-output", "children"),
    Input("analyze-button", "n_clicks"),
    State("user-input", "value"), # Use State to get value only on button click
    prevent_initial_call=True # Prevents the callback from running on app start
)
def analyze_sentiment(n_clicks, user_input):
    if not user_input:
        return "Please enter a review to analyze."

    # Step 1: Preprocess the input text using the functions from the notebook
    preprocessed_text = preprocess(user_input)
    lemmatized_text = lemmatize_text(preprocessed_text)

    # Step 2: Vectorize the preprocessed text
    # The vectorizer expects an iterable (like a list)
    vectorized_input = vectorizer.transform([lemmatized_text])

    # Step 3: Predict using the loaded model
    prediction = model.predict(vectorized_input)
    sentiment_flag = prediction[0] # Get the single prediction value (0 or 1)

    # Step 4: Interpret the result and return it
    # Based on notebook: 1 is Positive, 0 is Negative/Neutral
    if sentiment_flag == 1:
        sentiment = "😊 Positive"
        color = "green"
    else:
        sentiment = "😠 Negative / Neutral"
        color = "red"

    return html.Div([
        "Predicted Sentiment: ",
        html.Span(sentiment, style={'color': color})
    ])


# Callbacks for the dashboard charts (These were mostly fine)
@app.callback(
    Output("bar-chart", "figure"),
    Input("model-dropdown", "value")
)
def update_bar_chart(selected_model):
    filtered_df = metrics_df[["Metric", selected_model]]
    fig = px.bar(
        filtered_df, x="Metric", y=selected_model, 
        title=f"Metrics for {selected_model}",
        labels={selected_model: "Score"},
        text_auto='.2f'
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1.05]),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

@app.callback(
    Output("line-chart", "figure"),
    Input("model-dropdown", "value")
)
def update_line_chart(selected_model):
    # This chart is more meaningful when comparing train vs test for a model type
    model_type = selected_model.rsplit(' ', 1)[0] # e.g., 'LR SM' from 'LR SM Test'
    train_col = f"{model_type} Train"
    test_col = f"{model_type} Test"
    
    if train_col in metrics_df.columns and test_col in metrics_df.columns:
        compare_df = metrics_df[["Metric", train_col, test_col]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=compare_df["Metric"], y=compare_df[train_col], mode='lines+markers', name='Train'))
        fig.add_trace(go.Scatter(x=compare_df["Metric"], y=compare_df[test_col], mode='lines+markers', name='Test'))
        fig.update_layout(
            title=f"Train vs. Test Performance for {model_type}",
            yaxis=dict(range=[0, 1.05]),
            title_x=0.5,
            legend_title_text='Dataset'
        )
        return fig
    else: # Fallback for models without a pair
        return update_bar_chart(selected_model)

@app.callback(
    Output("heatmap", "figure"),
    Input("color-scale-dropdown", "value")
)
def update_heatmap(color_scale):
    heatmap_df = metrics_df.set_index('Metric')
    fig = px.imshow(
        heatmap_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=color_scale,
        labels=dict(x="Model", y="Metric", color="Score")
    )
    fig.update_layout(
        title="Overall Model Performance Heatmap",
        title_x=0.5
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)