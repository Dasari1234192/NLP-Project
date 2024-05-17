Fake News Detection Project
Overview
This project focuses on detecting fake news using various machine learning models. The models trained include Logistic Regression, SVM, Naive Bayes, Random Forest, LSTM, GRU, and CNN. The dataset consists of true and fake news articles, which are preprocessed and used to train and evaluate the models.

Project Structure
True (1).csv: CSV file containing true news articles.
Fake (1).csv: CSV file containing fake news articles.
fake_news_detection.py: Main script containing the code for preprocessing, training, and evaluating the models.
Requirements
Python 3.x
Pandas
Scikit-learn
NLTK
TensorFlow
Keras
pip install pandas scikit-learn nltk tensorflow keras
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Download the dataset:

Place the True (1).csv and Fake (1).csv files in the same directory as the script.

Preprocessing
The preprocessing steps include tokenization, converting text to lowercase, removing punctuation, and removing stopwords. The preprocessed text is then used for training the models.

Models
Traditional Machine Learning Models
Logistic Regression
Support Vector Machine (SVM)
Naive Bayes
Random Forest
Deep Learning Models
LSTM
GRU
CNN
Training and Evaluation
The script trains each model on the preprocessed text and evaluates their performance using accuracy, mean squared error (MSE), and a classification report. The evaluate_model_with_mse function is used to calculate these metrics.

Running the Script
To run the script, execute the following command in your terminal:
python fake_news_detection.py
Results
The script prints the accuracy, mean squared error, and classification report for each model. These metrics help in comparing the performance of different models on the true and fake news datasets.

Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please create a pull request or open an issue.

License
This project is licensed under the MIT License.

