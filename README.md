# Hybrid Approaches to Fake News Detection: A Comparative Analysis Using ML, LSTM, and BERT

This project explores various methods for detecting fake news using traditional machine learning, deep learning, and transformer-based approaches. Implemented in a Jupyter Notebook, it provides a comparative analysis of different models and demonstrates how Natural Language Processing (NLP) techniques can help identify misinformation.

## Technologies Used

- Python
- Scikit-learn
- Hugging Face Transformers (BERT)
- NLTK
- Keras / TensorFlow
- PyTorch

## Dataset

The dataset used in this project is sourced from Kaggle:  

[Fake News Detection Dataset](https://www.kaggle.com/code/therealsampat/fake-news-detection/input)  

It contains labeled news articles for binary classification (real or fake).

## Models Implemented

### Traditional Machine Learning
- Logistic Regression
- Support Vector Machine (SVM)
- Multinomial Naive Bayes

### Deep Learning
- LSTM (Long Short-Term Memory)

### Transformers
- Fine-tuned BERT model using Hugging Face's Transformers library

## How to Run the Project

1. Clone the Repository
   ```bash
   git clone https://github.com/AimanZaharin/Hybrid-Approaches-to-Fake-News-Detection-A-Comparative-Analysis-Using-ML-LSTM-and-BERT.git
   cd Hybrid-Approaches-to-Fake-News-Detection-A-Comparative-Analysis-Using-ML-LSTM-and-BERT

2. Create a Virtual Environment
    ```bash
    python -m venv fnd-venv
    source fnd-venv/bin/activate  # On Windows use: fnd-venv\Scripts\activate

3. Install dependencies
    ```bash
    pip install -r requirements.txt

4. Set Up NLTK data
    <br> Before running the notebook, create the necessary directory:
    ```bash
    mkdir -p fnd-venv/nltk_data
    ```
    
    Then, run the following in your Python environment (it is already included in the code):

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

5. Run the Jupyter Notebook
    Open the ``.ipynb`` file and run all cells to test different models and view results.

## Folder Structure

```yaml
Hybrid-Approaches-to-Fake-News-Detection-A-Comparative-Analysis-Using-ML-LSTM-and-BERT/
│
├── fnd-venv/                   # Virtual environment (excluded from repo)
├── requirements.txt            # Required dependencies
├── main.ipynb                  # Main notebook
├── README.md                   # Project documentation
├── True.csv                    # The dataset for Real News
└── Fake.csv                    # The dataset for Fake News


