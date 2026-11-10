# AcadPredict - Student Performance Prediction Using Statistical Machine Learning

An academic demonstration project built using **Python, Scikit-learn, Pandas, and Flask**. It predicts student academic performance out of 100 based on continuous behavioral and academic indicators using **Linear Regression (Ordinary Least Squares)**.

---

## 📌 Project Specifications

- **Project:** Student Performance Prediction
- **Brand / Platform:** AcadPredict
- **Method:** Statistical Machine Learning
- **Algorithm:** Linear Regression (Ordinary Least Squares - OLS)
- **Framework:** Flask
- **Libraries:** Python, Pandas, NumPy, Scikit-learn
- **Data Partitioning:** 80% Training (256 records) / 20% Testing (64 records)
- **Purpose:** Academic demonstration of student performance prediction using statistical machine learning.

---

## 🌟 Key Features

1. **Modern Academic & Data Science Interface:** Clean, high-contrast UI with cards, status pills, and fast viva demonstration profiles.
2. **Deterministic Scikit-Learn Engine:** Driven by Scikit-Learn's `LinearRegression` trained on 320 student records with zero synthetic/randomized numbers.
3. **Transparent Mathematical Explanations:** Displays the exact formula breakdown:
   $$\text{Predicted Score} = \beta_0 + \sum_{i=1}^{6} \beta_i \cdot x_i$$
4. **Empirical Evaluation Metrics:** Dynamically computed on the 20% unseen test set:
   - **$R^2$ Score:** `0.9306` (93.06% variance explained)
   - **MAE (Mean Absolute Error):** `1.816` marks
   - **MSE (Mean Squared Error):** `5.884`
   - **RMSE:** `2.426` marks
5. **Interactive Data Visualizations:** Three focused scatter charts with regression trendlines:
   - Study Hours vs Performance
   - Attendance vs Performance
   - Previous Exam Score vs Performance
6. **Dataset Explorer & CSV Upload:** Summary statistics, 10-row dataset preview table, dataset download link, and optional custom CSV upload with a reset feature.
7. **Comprehensive Viva Theory Section:** Explanations for all core machine learning concepts.

---

## 📂 Project Structure

```text
student-performance-project/
│
├── app.py                 # Flask web server and API endpoints
├── model.py               # ML pipeline (Data loading, 80/20 split, OLS training, evaluation)
├── dataset.csv            # 320 student records with 6 features + final score
├── requirements.txt       # Python package dependencies
├── generate_data.py       # Script used to synthesize the realistic dataset
├── README.md              # Documentation and viva presentation guide
│
├── templates/
│   └── index.html         # Modern responsive single-page web template
│
└── static/
    ├── style.css          # Refined academic styling (Inter font, responsive grid)
    └── script.js          # Interactive forms, Chart.js visuals, and sample pre-fills
```

---

## ⚙️ Setup and Installation Instructions

### Prerequisites
- Python 3.10+ installed on your system.

### Step 1: Open Terminal in the Project Directory
```powershell
cd student-performance-project
```

### Step 2: Install Required Dependencies
Run the following command in your terminal:
```bash
pip install -r requirements.txt
```
*(Or use `py -m pip install -r requirements.txt` on Windows)*

The dependencies include:
- `Flask`: Lightweight web application framework
- `scikit-learn`: Linear Regression, train/test split, and evaluation metrics
- `pandas`: Data loading, tabular preprocessing, and statistical summaries
- `numpy`: Fast mathematical array and matrix operations

---

## 🚀 Running the Web Application

To start the Flask local server:

```powershell
py app.py
```
*(Or `python app.py`)*

You will see:
```text
============================================================
AcadPredict - Student Performance Prediction Web App
Algorithm: Linear Regression (Statistical Machine Learning)
============================================================
Starting Flask server at http://127.0.0.1:5000 ...
 * Running on http://127.0.0.1:5000
```

Open your web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🎓 College Viva Guide & FAQ

### 1. Why did you choose Linear Regression?
> **Answer:** Linear Regression is the foundational statistical machine learning algorithm for continuous regression problems. It is transparent and interpretable: every coefficient ($\beta_i$) tells us the exact change in predicted exam marks for each 1-unit change in an input feature (e.g. $+1.81$ marks per additional daily study hour). This makes it easy to explain to faculty and examiners without "black-box" obscurity.

### 2. What are the Independent and Dependent Variables?
> **Answer:**
> - **Independent Variables ($X$):**
>   1. Study Hours per Day
>   2. Attendance Percentage
>   3. Previous Exam Score
>   4. Assignment Score
>   5. Sleep Hours per Day
>   6. Class Participation Score
> - **Dependent Variable ($y$):** Final Exam Performance Score (continuous, 0 to 100).

### 3. What does an 80/20 Train/Test Split mean?
> **Answer:** We partition our 320 student records into:
> - **Training Set (80% = 256 students):** The model uses these records to calculate the line of best fit and learn the optimal weights ($\beta$).
> - **Testing Set (20% = 64 students):** Kept completely unseen during training. We predict their scores and compare them against actual values to evaluate real-world generalization and prevent overfitting.

### 4. What do $R^2$, MAE, and MSE mean?
> **Answer:**
> - **$R^2$ Score (Coefficient of Determination):** Represents the percentage of variance in student scores explained by our features. Our model achieves $R^2 \approx 0.93$ ($93.06\%$).
> - **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual scores ($\approx 1.816$ marks).
> - **MSE (Mean Squared Error):** The average squared difference ($\approx 5.884$). Squaring heavily penalizes large errors.
> - **RMSE (Root Mean Squared Error):** $\sqrt{\text{MSE}} \approx 2.426$ marks.

### 5. What is the mathematical equation of your trained model?
> **Answer:**
> $$\text{Score} = -4.30 + 1.81(x_1) + 0.17(x_2) + 0.34(x_3) + 0.24(x_4) + 0.64(x_5) + 0.10(x_6)$$
> Where $x_1$ = Study Hours, $x_2$ = Attendance, $x_3$ = Previous Score, $x_4$ = Assignment, $x_5$ = Sleep Hours, and $x_6$ = Participation.
