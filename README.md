# Predicting Restaurant Success: A Sentiment-Weighted Rating Model

**Overview:**

This project analyzes online restaurant reviews and ratings to develop a predictive model for restaurant success.  The model incorporates sentiment analysis of review text to create a sentiment-weighted rating, combining it with existing numerical ratings to identify key factors influencing a restaurant's performance.  The analysis aims to predict future performance based on these combined factors.

**Technologies Used:**

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* NLTK (Natural Language Toolkit)
* Matplotlib
* Seaborn

**How to Run:**

1. **Install Dependencies:**  Ensure you have Python 3.x installed. Then, navigate to the project directory in your terminal and install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Model:** Execute the main script using:

   ```bash
   python main.py
   ```

**Example Output:**

The script will print key findings of the analysis to the console, including descriptive statistics of the dataset and model performance metrics.  Additionally, the script will generate several visualizations, including:

* **Sentiment Distribution:** A histogram showing the distribution of sentiment scores across all reviews.
* **Rating vs. Sentiment Scatter Plot:** A scatter plot illustrating the relationship between numerical ratings and sentiment scores.
* **Predicted vs. Actual Performance:**  A comparison plot showing the model's predicted performance against actual restaurant performance (if applicable, depending on the data and model used).  

These plots will be saved as PNG files in the `output` directory.  The exact names of the output files may vary.


**Data:**

The project requires a dataset containing restaurant reviews and ratings.  The specific format and source of this data are detailed within the code and documentation.  Please ensure you have the necessary data file(s) in the correct location before running the script.  (Note:  Sample data may be included for demonstration purposes).

**Further Development:**

Future development could include exploring more advanced sentiment analysis techniques, incorporating additional features (e.g., location, cuisine type), and using more sophisticated predictive modeling algorithms.