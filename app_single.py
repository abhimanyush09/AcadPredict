# ==============================================================================
# AcadPredict: Student Performance Prediction Using Linear Regression
# Single-File Standalone Web Application for VS Code
# 
# How to run in VS Code:
# 1. Open this file in VS Code
# 2. Click the Run (Play) button in top right OR press Ctrl + F5
#    (or run in terminal: py app_single.py)
# 3. The website will automatically open at http://127.0.0.1:5000 in your browser!
# ==============================================================================

import os
import sys
import threading
import webbrowser
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, Response
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------------------
# 1. Dataset Generation & Loading
# ------------------------------------------------------------------------------
def get_dataset():
    """
    Loads dataset.csv if available, or generates realistic student performance
    data (320 records) using reproducible seed 42.
    """
    csv_filename = "dataset.csv"
    if os.path.exists(csv_filename):
        try:
            return pd.read_csv(csv_filename)
        except Exception:
            pass

    # Synthesize realistic academic dataset
    np.random.seed(42)
    n_samples = 320

    study_hours = np.round(np.random.uniform(1.5, 9.0, n_samples), 1)
    attendance = np.round(np.random.uniform(55.0, 98.0, n_samples), 1)
    previous_score = np.round(np.random.normal(70, 14, n_samples), 1)
    previous_score = np.clip(previous_score, 40.0, 98.0)

    assignment_score = np.round(
        0.5 * previous_score + 2.5 * study_hours + np.random.normal(15, 6, n_samples), 1
    )
    assignment_score = np.clip(assignment_score, 45.0, 99.0)

    sleep_hours = np.round(np.random.normal(6.8, 1.1, n_samples), 1)
    sleep_hours = np.clip(sleep_hours, 4.5, 9.5)

    participation_score = np.round(
        0.3 * attendance + 0.4 * assignment_score + np.random.normal(18, 7, n_samples), 1
    )
    participation_score = np.clip(participation_score, 35.0, 98.0)

    final_score = (
        0.34 * previous_score
        + 0.22 * assignment_score
        + 0.16 * attendance
        + 1.75 * study_hours
        + 0.12 * participation_score
        + 0.60 * (sleep_hours - 5.0)
        + np.random.normal(0, 2.3, n_samples)
    )
    final_score = np.round(np.clip(final_score, 35.0, 99.5), 1)

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "assignment_score": assignment_score,
        "sleep_hours": sleep_hours,
        "participation_score": participation_score,
        "final_score": final_score
    })

    # Save to disk for user inspection
    try:
        df.to_csv(csv_filename, index=False)
    except Exception:
        pass

    return df

# ------------------------------------------------------------------------------
# 2. Machine Learning Model (Linear Regression)
# ------------------------------------------------------------------------------
class StudentPerformanceModel:
    FEATURE_NAMES = [
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_score",
        "sleep_hours",
        "participation_score"
    ]

    FEATURE_LABELS = {
        "study_hours": "Study Hours per Day",
        "attendance": "Attendance Percentage (%)",
        "previous_score": "Previous Exam Score",
        "assignment_score": "Assignment Score",
        "sleep_hours": "Sleep Hours per Day",
        "participation_score": "Class Participation Score"
    }

    def __init__(self):
        self.model = LinearRegression()
        self.metrics = {}
        self.coefficients = {}
        self.intercept = 0.0
        self.df = None
        self.dataset_stats = {}
        self.train_and_evaluate()

    def train_and_evaluate(self):
        self.df = get_dataset()

        X = self.df[self.FEATURE_NAMES]
        y = self.df["final_score"]

        # 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        self.metrics = {
            "r2_score": round(float(r2), 4),
            "r2_percentage": round(float(r2) * 100, 2),
            "mae": round(float(mae), 3),
            "mse": round(float(mse), 3),
            "rmse": round(float(np.sqrt(mse)), 3),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "total_students": len(self.df)
        }

        self.intercept = round(float(self.model.intercept_), 4)
        self.coefficients = {
            feat: round(float(coef), 4)
            for feat, coef in zip(self.FEATURE_NAMES, self.model.coef_)
        }

        self.dataset_stats = {
            "total_students": int(len(self.df)),
            "total_features": len(self.FEATURE_NAMES),
            "avg_study_hours": round(float(self.df["study_hours"].mean()), 2),
            "avg_attendance": round(float(self.df["attendance"].mean()), 2),
            "avg_previous_score": round(float(self.df["previous_score"].mean()), 2),
            "avg_assignment_score": round(float(self.df["assignment_score"].mean()), 2),
            "avg_sleep_hours": round(float(self.df["sleep_hours"].mean()), 2),
            "avg_participation_score": round(float(self.df["participation_score"].mean()), 2),
            "avg_final_score": round(float(self.df["final_score"].mean()), 2)
        }

    def predict(self, input_data):
        features_vector = [
            float(input_data.get("study_hours", 0)),
            float(input_data.get("attendance", 0)),
            float(input_data.get("previous_score", 0)),
            float(input_data.get("assignment_score", 0)),
            float(input_data.get("sleep_hours", 0)),
            float(input_data.get("participation_score", 0))
        ]

        X_new = pd.DataFrame([features_vector], columns=self.FEATURE_NAMES)
        raw_pred = self.model.predict(X_new)[0]
        predicted_score = round(float(np.clip(raw_pred, 0.0, 100.0)), 1)

        if predicted_score >= 85.0:
            category = "Excellent"
            badge_color = "success"
            explanation = "Outstanding academic potential. The student demonstrates solid study habits, high engagement, and strong fundamentals across all subjects."
        elif predicted_score >= 70.0:
            category = "Good"
            badge_color = "primary"
            explanation = "Good academic standing. The student performs reliably with steady attendance and assignment grades, with clear potential to reach Excellent."
        elif predicted_score >= 50.0:
            category = "Average"
            badge_color = "warning"
            explanation = "Moderate performance. The student meets baseline requirements but could benefit significantly from increasing regular study hours and attendance."
        else:
            category = "Needs Improvement"
            badge_color = "danger"
            explanation = "Academic intervention recommended. Low attendance or insufficient study hours are significantly dragging down projected marks."

        recommendations = []
        if float(input_data.get("study_hours", 0)) < 4.0:
            recommendations.append("Increase daily study sessions to at least 4-5 hours.")
        if float(input_data.get("attendance", 0)) < 75.0:
            recommendations.append("Attendance is below 75%. Attending more lectures directly boosts comprehension and grades.")
        if float(input_data.get("sleep_hours", 0)) < 6.0:
            recommendations.append("Target 7-8 hours of sleep per night for optimal memory retention.")
        if float(input_data.get("assignment_score", 0)) < 65.0:
            recommendations.append("Focus on regular homework and practical lab assignments to build confidence.")
        if not recommendations:
            recommendations.append("Maintain the current disciplined routine and active participation in class discussions.")

        formula_breakdown = {
            "intercept": self.intercept,
            "terms": [
                {
                    "feature": self.FEATURE_LABELS[feat],
                    "var": feat,
                    "coefficient": self.coefficients[feat],
                    "value": val,
                    "contribution": round(self.coefficients[feat] * val, 2)
                }
                for feat, val in zip(self.FEATURE_NAMES, features_vector)
            ]
        }

        return {
            "predicted_score": predicted_score,
            "category": category,
            "badge_color": badge_color,
            "explanation": explanation,
            "recommendations": recommendations,
            "formula_breakdown": formula_breakdown
        }

    def get_preview_data(self, limit=10):
        records = self.df.head(limit).to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if isinstance(v, float):
                    r[k] = round(v, 1)
        return records

# Initialize ML model
student_model = StudentPerformanceModel()

# ------------------------------------------------------------------------------
# 3. Flask Web Application & Routes
# ------------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "acadpredict-college-project-2026"

PROJECT_INFO = {
    "brand_name": "AcadPredict",
    "project_title": "Student Performance Prediction",
    "subtitle": "Using Statistical Machine Learning & Linear Regression",
    "method": "Statistical Machine Learning",
    "algorithm": "Linear Regression (Ordinary Least Squares)",
    "framework": "Flask",
    "libraries": "Python, Pandas, NumPy, Scikit-learn",
    "purpose": "Academic demonstration of student performance prediction using statistical machine learning.",
    "split_ratio": "80% Training / 20% Testing"
}

@app.route("/")
def home():
    preview_data = student_model.get_preview_data(limit=10)
    return render_template_string(
        HTML_TEMPLATE,
        project_info=PROJECT_INFO,
        student_info=PROJECT_INFO,
        metrics=student_model.metrics,
        dataset_stats=student_model.dataset_stats,
        coefficients=student_model.coefficients,
        intercept=student_model.intercept,
        preview_data=preview_data,
        feature_labels=student_model.FEATURE_LABELS
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        required_fields = [
            "study_hours", "attendance", "previous_score",
            "assignment_score", "sleep_hours", "participation_score"
        ]

        cleaned_data = {}
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                return jsonify({
                    "success": False,
                    "error": f"Missing required input: {student_model.FEATURE_LABELS.get(field, field)}"
                }), 400
            try:
                cleaned_data[field] = float(data[field])
            except ValueError:
                return jsonify({
                    "success": False,
                    "error": f"Invalid numerical value for {student_model.FEATURE_LABELS.get(field, field)}"
                }), 400

        # Boundary checks
        if not (0.0 <= cleaned_data["study_hours"] <= 24.0):
            return jsonify({"success": False, "error": "Study hours must be between 0 and 24 hours/day"}), 400
        if not (0.0 <= cleaned_data["attendance"] <= 100.0):
            return jsonify({"success": False, "error": "Attendance must be between 0% and 100%"}), 400
        if not (0.0 <= cleaned_data["previous_score"] <= 100.0):
            return jsonify({"success": False, "error": "Previous Exam Score must be between 0 and 100"}), 400
        if not (0.0 <= cleaned_data["assignment_score"] <= 100.0):
            return jsonify({"success": False, "error": "Assignment Score must be between 0 and 100"}), 400
        if not (0.0 <= cleaned_data["sleep_hours"] <= 24.0):
            return jsonify({"success": False, "error": "Sleep hours must be between 0 and 24 hours/day"}), 400
        if not (0.0 <= cleaned_data["participation_score"] <= 100.0):
            return jsonify({"success": False, "error": "Participation Score must be between 0 and 100"}), 400

        prediction_result = student_model.predict(cleaned_data)
        return jsonify({
            "success": True,
            "input": cleaned_data,
            "prediction": prediction_result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/download-dataset")
def download_dataset():
    csv_data = student_model.df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=student_performance_dataset.csv"}
    )

@app.route("/api/metrics")
def api_metrics():
    return jsonify({
        "metrics": student_model.metrics,
        "dataset_stats": student_model.dataset_stats,
        "coefficients": student_model.coefficients,
        "intercept": student_model.intercept
    })

# ------------------------------------------------------------------------------
# 4. Inlined Complete Web Template (HTML + Modern CSS + Dynamic JS)
# ------------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AcadPredict | Student Performance Prediction Using Linear Regression</title>
  <style>
/* ==========================================================================
   AcadPredict - Student Performance Prediction Using Linear Regression
   Clean, Academic, Consistent Design System
   ========================================================================== */

:root {
  /* Color Palette */
  --bg-page: #f8fafc;          /* Light grey/off-white background */
  --bg-card: #ffffff;          /* Clean white cards */
  --bg-subtle: #f1f5f9;        /* Soft section and table background */
  
  --navy: #0f172a;             /* Deep navy for primary headings and text */
  --navy-light: #1e293b;       /* Dark slate */
  --text-body: #334155;        /* Balanced body text */
  --text-muted: #64748b;       /* Helper text and labels */
  --text-light: #94a3b8;       /* Subtle hints & borders */
  
  --blue: #2563eb;             /* Primary blue accent */
  --blue-hover: #1d4ed8;       /* Blue hover state */
  --blue-light: #eff6ff;       /* Soft blue tint */
  --blue-border: #bfdbfe;      /* Blue border accent */
  
  --border-color: #e2e8f0;     /* Consistent subtle card borders */
  --border-focus: #3b82f6;     /* Input focus border */
  
  /* Status Colors */
  --success: #059669;
  --success-bg: #ecfdf5;
  --success-border: #a7f3d0;
  
  --good: #2563eb;
  --good-bg: #eff6ff;
  --good-border: #bfdbfe;
  
  --warning: #d97706;
  --warning-bg: #fffbeb;
  --warning-border: #fde68a;
  
  --danger: #dc2626;
  --danger-bg: #fef2f2;
  --danger-border: #fecaca;
  
  /* Typography */
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono: 'Fira Code', 'Courier New', monospace;
  
  /* Radii & Shadows */
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  
  --shadow-sm: 0 1px 3px 0 rgba(15, 23, 42, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(15, 23, 42, 0.07), 0 2px 4px -2px rgba(15, 23, 42, 0.04);
}

/* Base Reset */
*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html {
  scroll-behavior: smooth;
  font-size: 16px;
}

body {
  font-family: var(--font-sans);
  background-color: var(--bg-page);
  color: var(--text-body);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.container {
  width: 100%;
  max-width: 1160px;
  margin: 0 auto;
  padding: 0 1.5rem;
}

/* ==========================================================================
   Navigation Bar
   ========================================================================== */
.navbar {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: rgba(255, 255, 255, 0.96);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border-color);
  box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.03);
}

.nav-container {
  max-width: 1160px;
  margin: 0 auto;
  padding: 0.85rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  text-decoration: none;
  color: var(--navy);
}

.brand-icon-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: var(--radius-md);
  background: var(--blue-light);
  border: 1px solid var(--blue-border);
  color: var(--blue);
}

.brand-icon-wrapper.mini {
  width: 30px;
  height: 30px;
}

.brand-title {
  font-size: 1.25rem;
  font-weight: 800;
  color: var(--navy);
  letter-spacing: -0.02em;
}

.nav-links {
  display: flex;
  align-items: center;
  gap: 1.5rem;
}

.nav-link {
  text-decoration: none;
  color: var(--text-muted);
  font-size: 0.92rem;
  font-weight: 500;
  padding: 0.35rem 0.1rem;
  position: relative;
  transition: color 0.2s ease;
}

.nav-link:hover {
  color: var(--blue);
}

.nav-link.active {
  color: var(--blue);
  font-weight: 600;
}

.nav-link.active::after {
  content: '';
  position: absolute;
  bottom: -4px;
  left: 0;
  right: 0;
  height: 2px;
  background-color: var(--blue);
  border-radius: 999px;
}

.mobile-toggle {
  display: none;
  background: none;
  border: 1px solid var(--border-color);
  padding: 0.45rem;
  border-radius: var(--radius-sm);
  cursor: pointer;
}

.mobile-toggle .bar {
  display: block;
  width: 20px;
  height: 2px;
  background-color: var(--navy);
  margin: 4px 0;
}

/* ==========================================================================
   1. Hero Section
   ========================================================================== */
.hero-section {
  padding: 4.5rem 0 3.5rem;
  background-color: #ffffff;
  border-bottom: 1px solid var(--border-color);
}

.hero-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.hero-text {
  max-width: 760px;
  margin-bottom: 2.75rem;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: var(--blue-light);
  color: var(--blue);
  font-size: 0.82rem;
  font-weight: 600;
  padding: 0.35rem 0.9rem;
  border-radius: 999px;
  border: 1px solid var(--blue-border);
  margin-bottom: 1.25rem;
}

.badge-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background-color: var(--blue);
}

.hero-title {
  font-size: 2.75rem;
  font-weight: 800;
  color: var(--navy);
  line-height: 1.15;
  letter-spacing: -0.03em;
  margin-bottom: 0.5rem;
}

.hero-subtitle {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--blue);
  margin-bottom: 1.25rem;
}

.hero-desc {
  font-size: 1.05rem;
  color: var(--text-muted);
  line-height: 1.6;
  margin-bottom: 2rem;
}

.hero-actions {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}

/* 3 Consistent Overview Cards */
.hero-cards-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.25rem;
}

.summary-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1.5rem 1.25rem;
  text-align: left;
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.summary-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: var(--blue-border);
}

.summary-num {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--blue);
  background: var(--blue-light);
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: var(--radius-sm);
  margin-bottom: 0.75rem;
}

.summary-card h3 {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 0.4rem;
}

.summary-card p {
  font-size: 0.86rem;
  color: var(--text-muted);
  line-height: 1.5;
}

/* ==========================================================================
   Buttons
   ========================================================================== */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  font-weight: 600;
  padding: 0.65rem 1.35rem;
  border-radius: var(--radius-md);
  text-decoration: none;
  cursor: pointer;
  border: 1px solid transparent;
  transition: all 0.2s ease;
  font-size: 0.95rem;
}

.btn-lg {
  padding: 0.75rem 1.65rem;
  font-size: 1rem;
}

.btn-sm {
  padding: 0.4rem 0.85rem;
  font-size: 0.84rem;
}

.btn-primary {
  background-color: var(--blue);
  color: #ffffff;
}

.btn-primary:hover {
  background-color: var(--blue-hover);
  box-shadow: var(--shadow-md);
}

.btn-secondary {
  background-color: #ffffff;
  color: var(--navy);
  border-color: var(--border-color);
  box-shadow: var(--shadow-sm);
}

.btn-secondary:hover {
  background-color: var(--bg-subtle);
  border-color: var(--text-light);
  color: var(--blue);
}

.btn-block {
  width: 100%;
}

/* ==========================================================================
   Sections Layout & Headers
   ========================================================================== */
.section {
  padding: 4.5rem 0;
}

.bg-light {
  background-color: var(--bg-page);
}

.section-header {
  text-align: center;
  max-width: 700px;
  margin: 0 auto 2.75rem;
}

.section-tag {
  display: inline-block;
  font-size: 0.76rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--blue);
  background: var(--blue-light);
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  margin-bottom: 0.65rem;
  border: 1px solid var(--blue-border);
}

.section-title {
  font-size: 2.1rem;
  font-weight: 800;
  color: var(--navy);
  line-height: 1.25;
  margin-bottom: 0.65rem;
  letter-spacing: -0.02em;
}

.section-subtitle {
  font-size: 1rem;
  color: var(--text-muted);
  line-height: 1.6;
}

/* Consistent Card Style */
.card {
  background: var(--bg-card);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-sm);
  padding: 1.75rem;
}

.card-header {
  margin-bottom: 1.5rem;
  padding-bottom: 0.85rem;
  border-bottom: 1px solid var(--border-color);
}

.card-header h3 {
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--navy);
}

.card-desc {
  font-size: 0.86rem;
  color: var(--text-muted);
  margin-top: 0.2rem;
}

/* ==========================================================================
   2. Prediction Form & Result
   ========================================================================== */
.demo-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  margin-bottom: 2rem;
  flex-wrap: wrap;
  background: var(--bg-card);
  padding: 0.75rem 1.25rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-sm);
}

.demo-label {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--navy);
}

.demo-buttons-group {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.btn-demo {
  background: var(--bg-page);
  border: 1px solid var(--border-color);
  color: var(--text-body);
  font-size: 0.82rem;
  font-weight: 600;
  padding: 0.35rem 0.75rem;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-demo:hover {
  background: var(--blue-light);
  border-color: var(--blue-border);
  color: var(--blue);
}

.alert-error {
  background-color: var(--danger-bg);
  color: var(--danger);
  border: 1px solid var(--danger-border);
  padding: 0.75rem 1rem;
  border-radius: var(--radius-md);
  font-size: 0.88rem;
  font-weight: 500;
  margin-bottom: 1.25rem;
}

.prediction-grid {
  display: grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: 2rem;
  align-items: start;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  margin-bottom: 1.25rem;
}

.form-group label {
  display: block;
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--navy);
  margin-bottom: 0.25rem;
}

.input-hint {
  display: block;
  font-size: 0.74rem;
  font-weight: 400;
  color: var(--text-muted);
  margin-bottom: 0.35rem;
}

.input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.input-wrapper input {
  width: 100%;
  padding: 0.65rem 0.85rem;
  padding-right: 4.25rem;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  font-family: var(--font-sans);
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--navy);
  background-color: #ffffff;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.input-wrapper input:focus {
  outline: none;
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
}

.input-unit {
  position: absolute;
  right: 0.85rem;
  font-size: 0.76rem;
  font-weight: 600;
  color: var(--text-muted);
  pointer-events: none;
}

.form-actions {
  margin-top: 1.5rem;
}

/* Result Card */
.result-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  position: sticky;
  top: 5rem;
}

.result-placeholder {
  text-align: center;
  padding: 3rem 1.5rem;
  color: var(--text-muted);
}

.placeholder-icon {
  margin-bottom: 1rem;
  color: var(--text-light);
}

.result-placeholder h4 {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 0.5rem;
}

.result-placeholder p {
  font-size: 0.86rem;
  line-height: 1.5;
}

.score-display-block {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  margin-bottom: 1.25rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--border-color);
}

.score-ring {
  width: 105px;
  height: 105px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: var(--bg-page);
  border: 3.5px solid var(--blue);
  box-shadow: var(--shadow-sm);
  flex-shrink: 0;
  transition: border-color 0.3s ease;
}

.score-number {
  font-size: 2rem;
  font-weight: 800;
  line-height: 1;
  color: var(--navy);
  letter-spacing: -0.02em;
}

.score-denominator {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-muted);
  margin-top: 0.2rem;
}

.category-info {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.category-badge {
  display: inline-block;
  font-size: 1.05rem;
  font-weight: 700;
  padding: 0.35rem 0.95rem;
  border-radius: var(--radius-md);
  text-align: center;
}

.badge-excellent {
  background: var(--success-bg);
  color: var(--success);
  border: 1px solid var(--success-border);
}

.badge-good {
  background: var(--good-bg);
  color: var(--good);
  border: 1px solid var(--good-border);
}

.badge-average {
  background: var(--warning-bg);
  color: var(--warning);
  border: 1px solid var(--warning-border);
}

.badge-needs_improvement {
  background: var(--danger-bg);
  color: var(--danger);
  border: 1px solid var(--danger-border);
}

.category-caption {
  font-size: 0.8rem;
  color: var(--text-muted);
  font-weight: 500;
}

.explanation-box, .recommendations-box {
  background: var(--bg-page);
  padding: 0.9rem 1.15rem;
  border-radius: var(--radius-md);
  margin-bottom: 1.15rem;
  border: 1px solid var(--border-color);
}

.explanation-box h4, .recommendations-box h4 {
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 0.35rem;
}

.explanation-box p {
  font-size: 0.85rem;
  color: var(--text-body);
  line-height: 1.5;
}

.recommendations-box ul {
  padding-left: 1.2rem;
  font-size: 0.83rem;
  color: var(--text-body);
}

.recommendations-box li {
  margin-bottom: 0.25rem;
}

.math-breakdown-box {
  background: #ffffff;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin-bottom: 1.25rem;
}

.breakdown-header {
  background: var(--navy);
  color: #ffffff;
  padding: 0.5rem 0.85rem;
  font-size: 0.78rem;
  font-weight: 600;
}

.breakdown-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.78rem;
}

.breakdown-table th, .breakdown-table td {
  padding: 0.45rem 0.65rem;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}

.breakdown-table th {
  background: var(--bg-page);
  font-weight: 600;
  color: var(--text-muted);
}

.breakdown-table tfoot td {
  background: var(--bg-page);
}

.breakdown-table tr.final-row td {
  background: var(--blue-light);
  color: var(--blue);
  font-size: 0.84rem;
}

.predict-again-wrap {
  margin-top: 1rem;
}

/* Spinner */
.spinner {
  display: inline-block;
  width: 1rem;
  height: 1rem;
  border: 2px solid rgba(255, 255, 255, 0.4);
  border-top-color: #ffffff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ==========================================================================
   3. Dataset Section
   ========================================================================== */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 1rem;
  margin-bottom: 2rem;
}

.stat-card {
  background: var(--bg-card);
  padding: 1.25rem 1rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
  text-align: center;
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
  border-color: var(--blue-border);
}

.stat-icon-wrapper {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 38px;
  height: 38px;
  border-radius: var(--radius-sm);
  background: var(--blue-light);
  color: var(--blue);
  margin-bottom: 0.5rem;
}

.stat-value {
  font-size: 1.55rem;
  font-weight: 800;
  color: var(--navy);
  line-height: 1.2;
  font-family: var(--font-mono);
}

.stat-value small {
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--text-muted);
}

.stat-value.text-blue {
  color: var(--blue);
}

.stat-label {
  font-size: 0.76rem;
  font-weight: 600;
  color: var(--text-muted);
  margin-top: 0.25rem;
}

/* Feature Guide Card */
.feature-guide-card {
  margin-bottom: 2rem;
}

.features-guide-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-top: 1.25rem;
}

.guide-item {
  background: var(--bg-page);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1rem;
  font-size: 0.85rem;
  line-height: 1.45;
}

.guide-item strong {
  display: block;
  color: var(--navy);
  margin-top: 0.35rem;
  margin-bottom: 0.25rem;
}

.guide-badge {
  display: inline-block;
  font-size: 0.72rem;
  font-family: var(--font-mono);
  font-weight: 700;
  background: var(--blue-light);
  color: var(--blue);
  padding: 0.15rem 0.5rem;
  border-radius: var(--radius-xs);
}

.guide-item.target-item {
  grid-column: 1 / -1;
  background: #f0fdf4;
  border-color: #bbf7d0;
}

.target-badge {
  background: #dcfce7;
  color: #15803d;
}

/* Table Card */
.table-card {
  padding: 1.25rem;
}

.table-header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
  flex-wrap: wrap;
  gap: 1rem;
}

.table-responsive {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
  text-align: left;
}

.data-table th, .data-table td {
  padding: 0.7rem 0.9rem;
  border-bottom: 1px solid var(--border-color);
}

.data-table th {
  background: var(--bg-page);
  font-weight: 700;
  color: var(--navy);
  white-space: nowrap;
}

.data-table tbody tr:hover {
  background-color: var(--bg-page);
}

.row-num {
  font-family: var(--font-mono);
  font-weight: 600;
  color: var(--text-light);
}

.score-pill {
  display: inline-block;
  font-weight: 700;
  background: var(--blue-light);
  color: var(--blue);
  padding: 0.2rem 0.55rem;
  border-radius: var(--radius-xs);
  border: 1px solid var(--blue-border);
}

.badge-coef {
  display: inline-block;
  font-family: var(--font-mono);
  font-weight: 700;
  background: var(--success-bg);
  color: var(--success);
  padding: 0.2rem 0.5rem;
  border-radius: var(--radius-xs);
  border: 1px solid var(--success-border);
}

.intercept-pill {
  background: var(--navy);
  color: #ffffff;
  padding: 0.35rem 0.85rem;
  border-radius: var(--radius-md);
  font-size: 0.84rem;
}

/* ==========================================================================
   4. Model Evaluation Section
   ========================================================================== */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1.25rem;
}

.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: 1.5rem 1.25rem;
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
  transform: translateY(-2px);
  border-color: var(--blue-border);
}

.metric-badge {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  color: var(--blue);
  background: var(--blue-light);
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  margin-bottom: 0.75rem;
  border: 1px solid var(--blue-border);
}

.metric-value {
  font-size: 2.25rem;
  font-weight: 800;
  color: var(--navy);
  line-height: 1;
  font-family: var(--font-mono);
  letter-spacing: -0.03em;
}

.metric-value small {
  font-size: 1rem;
  font-weight: 500;
  color: var(--text-muted);
}

.metric-percent {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--success);
  margin-top: 0.35rem;
  margin-bottom: 0.85rem;
}

.metric-name {
  font-size: 0.96rem;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 0.35rem;
}

.metric-desc {
  font-size: 0.83rem;
  color: var(--text-muted);
  line-height: 1.5;
}

/* ==========================================================================
   5. About the Project Section
   ========================================================================== */
.workflow-box {
  padding: 1.75rem;
}

.simple-pipeline {
  display: flex;
  align-items: stretch;
  justify-content: space-between;
  gap: 0.5rem;
  margin-top: 1.5rem;
  overflow-x: auto;
}

.pipe-step {
  flex: 1;
  min-width: 150px;
  background: var(--bg-page);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1.25rem 0.85rem;
  text-align: center;
}

.pipe-step .step-num {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--blue);
  color: #ffffff;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 0.82rem;
  margin-bottom: 0.65rem;
}

.pipe-step h4 {
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 0.35rem;
}

.pipe-step p {
  font-size: 0.78rem;
  color: var(--text-muted);
  line-height: 1.4;
}

.pipe-arrow {
  display: flex;
  align-items: center;
  color: var(--text-light);
  font-weight: 700;
  font-size: 1.1rem;
}

/* Tech Grid */
.tech-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1.25rem;
}

.tech-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1.25rem 1rem;
  text-align: center;
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s ease;
}

.tech-card:hover {
  transform: translateY(-2px);
  border-color: var(--blue-border);
}

.tech-icon {
  font-size: 1.8rem;
  margin-bottom: 0.5rem;
}

.tech-card h4 {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 0.35rem;
}

.tech-card p {
  font-size: 0.8rem;
  color: var(--text-muted);
  line-height: 1.45;
}

.formula-note-card {
  padding: 1.5rem;
}

.math-banner {
  background: var(--navy);
  color: #38bdf8;
  font-family: var(--font-mono);
  font-size: 1.15rem;
  font-weight: 600;
  padding: 1rem 1.25rem;
  border-radius: var(--radius-md);
  margin: 1rem 0;
  overflow-x: auto;
}

.formula-explanation {
  font-size: 0.88rem;
  color: var(--text-muted);
  line-height: 1.55;
}

/* ==========================================================================
   Footer
   ========================================================================== */
.footer {
  background: var(--navy);
  color: #94a3b8;
  padding: 2.75rem 0 2.25rem;
  border-top: 1px solid #1e293b;
}

.footer-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.footer-brand {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.65rem;
}

.footer-title {
  font-size: 1.2rem;
  font-weight: 800;
  color: #ffffff;
}

.footer-desc {
  font-size: 0.9rem;
  color: #cbd5e1;
  margin-bottom: 0.35rem;
}

.footer-copy {
  font-size: 0.78rem;
  color: #64748b;
}

/* ==========================================================================
   Responsive Breakpoints
   ========================================================================== */
@media (max-width: 1024px) {
  .hero-cards-grid {
    grid-template-columns: 1fr;
  }
  .prediction-grid {
    grid-template-columns: 1fr;
  }
  .result-card {
    position: static;
  }
  .stats-grid {
    grid-template-columns: repeat(3, 1fr);
  }
  .features-guide-grid {
    grid-template-columns: 1fr 1fr;
  }
  .metrics-grid {
    grid-template-columns: 1fr 1fr;
  }
  .tech-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .hero-title {
    font-size: 2.2rem;
  }
  .nav-links {
    display: none;
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: #ffffff;
    flex-direction: column;
    padding: 1.25rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    border-top: 1px solid var(--border-color);
    gap: 0.85rem;
  }
  .nav-links.active {
    display: flex;
  }
  .mobile-toggle {
    display: block;
  }
  .form-row {
    grid-template-columns: 1fr;
  }
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  .features-guide-grid {
    grid-template-columns: 1fr;
  }
  .metrics-grid {
    grid-template-columns: 1fr;
  }
  .tech-grid {
    grid-template-columns: 1fr;
  }
  .simple-pipeline {
    flex-direction: column;
  }
  .pipe-arrow {
    transform: rotate(90deg);
    justify-content: center;
    padding: 0.4rem 0;
  }
}

  </style>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet">
</head>
<body>

  <!-- Navigation Bar -->
  <header class="navbar" id="main-navbar">
    <div class="nav-container">
      <a href="#home" class="nav-brand" aria-label="AcadPredict Home">
        <span class="brand-icon-wrapper">
          <svg class="brand-svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
            <path d="M6 12v5c3 3 9 3 12 0v-5"/>
            <circle cx="12" cy="12" r="1.5" fill="currentColor"/>
          </svg>
        </span>
        <span class="brand-title">AcadPredict</span>
      </a>

      <nav class="nav-links" id="nav-links">
        <a href="#home" class="nav-link active">Home</a>
        <a href="#predict" class="nav-link">Prediction</a>
        <a href="#dataset" class="nav-link">Dataset</a>
        <a href="#evaluation" class="nav-link">Model Evaluation</a>
        <a href="#about" class="nav-link">About Project</a>
      </nav>

      <button class="mobile-toggle" id="mobile-toggle" aria-label="Toggle navigation menu">
        <span class="bar"></span>
        <span class="bar"></span>
        <span class="bar"></span>
      </button>
    </div>
  </header>

  <!-- 1. Home / Hero Section -->
  <section id="home" class="hero-section">
    <div class="container hero-content">
      <div class="hero-text">
        <div class="hero-badge">
          <span class="badge-dot"></span>
          <span>Machine Learning College Project</span>
        </div>
        
        <h1 class="hero-title">AcadPredict</h1>
        <p class="hero-subtitle">Student Performance Prediction Using Linear Regression</p>
        
        <p class="hero-desc">
          An intuitive and interpretable statistical machine learning application that forecasts a student's academic performance out of 100 using daily study hours, attendance percentage, and previous exam grades.
        </p>

        <div class="hero-actions">
          <a href="#predict" class="btn btn-primary btn-lg">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
            </svg>
            <span>Predict Performance</span>
          </a>
          <a href="#dataset" class="btn btn-secondary btn-lg">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="16" y1="13" x2="8" y2="13"></line>
              <line x1="16" y1="17" x2="8" y2="17"></line>
            </svg>
            <span>Explore Dataset</span>
          </a>
        </div>
      </div>

      <!-- Hero Summary Highlights -->
      <div class="hero-cards-grid">
        <div class="summary-card">
          <div class="summary-num">01</div>
          <h3>6 Academic Features</h3>
          <p>Captures study hours, lecture attendance, past tests, assignments, sleep, and participation.</p>
        </div>
        <div class="summary-card">
          <div class="summary-num">02</div>
          <h3>Linear Regression</h3>
          <p>Trained using Ordinary Least Squares (OLS) via Scikit-Learn for complete mathematical transparency.</p>
        </div>
        <div class="summary-card">
          <div class="summary-num">03</div>
          <h3>Continuous Scoring</h3>
          <p>Predicts a score out of 100 categorized into Excellent, Good, Average, or Needs Improvement.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- 2. Prediction Form & Result Section -->
  <section id="predict" class="section bg-light">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Interactive Prediction</span>
        <h2 class="section-title">Student Performance Prediction</h2>
        <p class="section-subtitle">
          Enter the student's six academic and behavioral metrics below to compute an actual predicted score using the trained Linear Regression model.
        </p>
      </div>

      <!-- Quick Viva Demonstration Presets -->
      <div class="demo-bar">
        <span class="demo-label">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
          Quick Test Profiles:
        </span>
        <div class="demo-buttons-group">
          <button type="button" class="btn-demo" onclick="fillSample('excellent')">High Performer</button>
          <button type="button" class="btn-demo" onclick="fillSample('good')">Good Student</button>
          <button type="button" class="btn-demo" onclick="fillSample('average')">Average Student</button>
          <button type="button" class="btn-demo" onclick="fillSample('struggling')">Needs Support</button>
        </div>
      </div>

      <div class="prediction-grid">
        <!-- Input Form Card -->
        <div class="card form-card">
          <div class="card-header">
            <h3>Student Information</h3>
            <p class="card-desc">Fill in all 6 fields to run the Linear Regression model</p>
          </div>

          <form id="prediction-form" class="prediction-form" onsubmit="handlePrediction(event)">
            <!-- Error message container -->
            <div id="form-error-msg" class="alert-error" style="display: none;"></div>

            <div class="form-row">
              <!-- Feature 1: Study Hours -->
              <div class="form-group">
                <label for="study_hours">
                  Study Hours per Day
                  <span class="input-hint">Average self-study hours (0 - 24)</span>
                </label>
                <div class="input-wrapper">
                  <input type="number" id="study_hours" name="study_hours" step="0.1" min="0" max="24" value="5.5" required placeholder="e.g. 5.5">
                  <span class="input-unit">hrs/day</span>
                </div>
              </div>

              <!-- Feature 2: Attendance -->
              <div class="form-group">
                <label for="attendance">
                  Attendance Percentage
                  <span class="input-hint">Class attendance rate (0 - 100%)</span>
                </label>
                <div class="input-wrapper">
                  <input type="number" id="attendance" name="attendance" step="0.1" min="0" max="100" value="82.0" required placeholder="e.g. 82.0">
                  <span class="input-unit">%</span>
                </div>
              </div>
            </div>

            <div class="form-row">
              <!-- Feature 3: Previous Exam Score -->
              <div class="form-group">
                <label for="previous_score">
                  Previous Exam Score
                  <span class="input-hint">Score in past exam (0 - 100 marks)</span>
                </label>
                <div class="input-wrapper">
                  <input type="number" id="previous_score" name="previous_score" step="0.1" min="0" max="100" value="74.0" required placeholder="e.g. 74.0">
                  <span class="input-unit">/ 100</span>
                </div>
              </div>

              <!-- Feature 4: Assignment Score -->
              <div class="form-group">
                <label for="assignment_score">
                  Assignment Score
                  <span class="input-hint">Average assignment marks (0 - 100)</span>
                </label>
                <div class="input-wrapper">
                  <input type="number" id="assignment_score" name="assignment_score" step="0.1" min="0" max="100" value="78.0" required placeholder="e.g. 78.0">
                  <span class="input-unit">/ 100</span>
                </div>
              </div>
            </div>

            <div class="form-row">
              <!-- Feature 5: Sleep Hours -->
              <div class="form-group">
                <label for="sleep_hours">
                  Sleep Hours per Day
                  <span class="input-hint">Daily sleep duration (0 - 24 hrs)</span>
                </label>
                <div class="input-wrapper">
                  <input type="number" id="sleep_hours" name="sleep_hours" step="0.1" min="0" max="24" value="7.0" required placeholder="e.g. 7.0">
                  <span class="input-unit">hrs/day</span>
                </div>
              </div>

              <!-- Feature 6: Class Participation -->
              <div class="form-group">
                <label for="participation_score">
                  Class Participation Score
                  <span class="input-hint">Class engagement score (0 - 100)</span>
                </label>
                <div class="input-wrapper">
                  <input type="number" id="participation_score" name="participation_score" step="0.1" min="0" max="100" value="70.0" required placeholder="e.g. 70.0">
                  <span class="input-unit">/ 100</span>
                </div>
              </div>
            </div>

            <div class="form-actions">
              <button type="submit" class="btn btn-primary btn-block btn-lg" id="predict-btn">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                <span class="btn-text">Predict Performance</span>
                <span class="spinner" id="spinner" style="display: none;"></span>
              </button>
            </div>
          </form>
        </div>

        <!-- Prediction Result Display Card -->
        <div class="card result-card" id="result-container">
          <div class="card-header">
            <h3>Prediction Result</h3>
            <p class="card-desc">Computed using the trained Scikit-Learn Linear Regression model</p>
          </div>

          <!-- Empty Placeholder -->
          <div id="result-placeholder" class="result-placeholder">
            <div class="placeholder-icon">
              <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
              </svg>
            </div>
            <h4>Ready to Predict</h4>
            <p>Select a quick test profile above or enter custom student metrics, then click <strong>"Predict Performance"</strong>.</p>
          </div>

          <!-- Live Result Display -->
          <div id="result-display" class="result-display" style="display: none;">
            <div class="score-display-block">
              <div class="score-ring" id="score-ring">
                <span class="score-number" id="res-score">--</span>
                <span class="score-denominator">out of 100</span>
              </div>
              <div class="category-info">
                <span class="category-badge" id="res-category">Category</span>
                <span class="category-caption">Academic Standing</span>
              </div>
            </div>

            <div class="explanation-box">
              <h4>Explanation</h4>
              <p id="res-explanation">Evaluation explanation will appear here.</p>
            </div>

            <div class="recommendations-box">
              <h4>Actionable Feedback</h4>
              <ul id="res-recommendations"></ul>
            </div>

            <!-- Mathematical Breakdown -->
            <div class="math-breakdown-box">
              <div class="breakdown-header">
                <span>Model Contribution Breakdown (β · x)</span>
              </div>
              <div class="table-responsive">
                <table class="breakdown-table">
                  <thead>
                    <tr>
                      <th>Factor</th>
                      <th>Input</th>
                      <th>Weight (β)</th>
                      <th>Contribution</th>
                    </tr>
                  </thead>
                  <tbody id="breakdown-body">
                    <!-- Populated dynamically by script.js -->
                  </tbody>
                  <tfoot>
                    <tr>
                      <td colspan="3"><strong>Model Intercept (β₀):</strong></td>
                      <td id="res-intercept"><strong>--</strong></td>
                    </tr>
                    <tr class="final-row">
                      <td colspan="3"><strong>Predicted Final Score:</strong></td>
                      <td id="res-final-calc"><strong>--</strong></td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </div>

            <!-- Predict Again Option -->
            <div class="predict-again-wrap">
              <button type="button" class="btn btn-secondary btn-block" onclick="resetPredictionForm()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
                <span>Predict Again</span>
              </button>
            </div>

          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- 3. Dataset Overview Section -->
  <section id="dataset" class="section">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Empirical Data</span>
        <h2 class="section-title">Student Dataset Overview</h2>
        <p class="section-subtitle">
          Real records used to fit and validate the Linear Regression model, showing typical student academic habits.
        </p>
      </div>

      <!-- Dataset Metric Summary Cards -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon-wrapper">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
          </div>
          <div class="stat-value" id="stat-students">{{ dataset_stats.total_students }}</div>
          <div class="stat-label">Total Student Records</div>
        </div>

        <div class="stat-card">
          <div class="stat-icon-wrapper">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>
          </div>
          <div class="stat-value" id="stat-features">{{ dataset_stats.total_features }}</div>
          <div class="stat-label">Independent Features</div>
        </div>

        <div class="stat-card">
          <div class="stat-icon-wrapper">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
          </div>
          <div class="stat-value" id="stat-study">{{ dataset_stats.avg_study_hours }} <small>hrs</small></div>
          <div class="stat-label">Avg. Study Hours</div>
        </div>

        <div class="stat-card">
          <div class="stat-icon-wrapper">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
          </div>
          <div class="stat-value" id="stat-attendance">{{ dataset_stats.avg_attendance }} <small>%</small></div>
          <div class="stat-label">Avg. Attendance</div>
        </div>

        <div class="stat-card">
          <div class="stat-icon-wrapper">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
          </div>
          <div class="stat-value" id="stat-prev">{{ dataset_stats.avg_previous_score }}</div>
          <div class="stat-label">Avg. Previous Score</div>
        </div>

        <div class="stat-card">
          <div class="stat-icon-wrapper">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="m4.93 4.93 4.24 4.24"></path><path d="m14.83 9.17 4.24-4.24"></path></svg>
          </div>
          <div class="stat-value text-blue" id="stat-final">{{ dataset_stats.avg_final_score }}</div>
          <div class="stat-label">Avg. Final Score</div>
        </div>
      </div>

      <!-- Feature Description Guide (Simple Language for Viva) -->
      <div class="card feature-guide-card">
        <h3>Feature Guide & Variables</h3>
        <p class="card-desc">Understanding the input parameters and the target variable in simple terms:</p>
        
        <div class="features-guide-grid">
          <div class="guide-item">
            <span class="guide-badge">Input (x₁)</span>
            <strong>Study Hours per Day:</strong> Average daily hours dedicated to self-study and revision outside class.
          </div>
          <div class="guide-item">
            <span class="guide-badge">Input (x₂)</span>
            <strong>Attendance Percentage:</strong> Ratio of lectures attended during the semester.
          </div>
          <div class="guide-item">
            <span class="guide-badge">Input (x₃)</span>
            <strong>Previous Exam Score:</strong> Baseline marks scored in prior examinations (0 - 100).
          </div>
          <div class="guide-item">
            <span class="guide-badge">Input (x₄)</span>
            <strong>Assignment Score:</strong> Marks obtained in continuous assignments and homework (0 - 100).
          </div>
          <div class="guide-item">
            <span class="guide-badge">Input (x₅)</span>
            <strong>Sleep Hours per Day:</strong> Daily resting duration supporting cognitive retention (0 - 24).
          </div>
          <div class="guide-item">
            <span class="guide-badge">Input (x₆)</span>
            <strong>Class Participation:</strong> In-class engagement and discussions score (0 - 100).
          </div>
          <div class="guide-item target-item">
            <span class="guide-badge target-badge">Target Variable (y)</span>
            <strong>Final Exam Score:</strong> The continuous outcome (0 - 100) predicted by the Linear Regression model.
          </div>
        </div>
      </div>

      <!-- Dataset Preview Table -->
      <div class="card table-card">
        <div class="table-header-bar">
          <div>
            <h3>Dataset Sample (First 10 Students)</h3>
            <p class="card-desc">Preprocessed numerical observations used for model training and evaluation</p>
          </div>
          <a href="{{ url_for('download_dataset') }}" class="btn btn-sm btn-secondary">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
            <span>Download dataset.csv</span>
          </a>
        </div>
        <div class="table-responsive">
          <table class="data-table" id="preview-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Study Hours</th>
                <th>Attendance</th>
                <th>Previous Exam</th>
                <th>Assignment</th>
                <th>Sleep Hours</th>
                <th>Participation</th>
                <th>Final Score</th>
              </tr>
            </thead>
            <tbody>
              {% for row in preview_data %}
              <tr>
                <td><span class="row-num">{{ loop.index }}</span></td>
                <td>{{ row.study_hours }} hrs</td>
                <td>{{ row.attendance }}%</td>
                <td>{{ row.previous_score }}</td>
                <td>{{ row.assignment_score }}</td>
                <td>{{ row.sleep_hours }} hrs</td>
                <td>{{ row.participation_score }}</td>
                <td><span class="score-pill">{{ row.final_score }}</span></td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <!-- 4. Model Evaluation Section -->
  <section id="evaluation" class="section bg-light">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Empirical Verification</span>
        <h2 class="section-title">Model Evaluation Metrics</h2>
        <p class="section-subtitle">
          Actual evaluation statistics calculated directly on the 20% held-out test cohort (64 unseen students) by Scikit-Learn.
        </p>
      </div>

      <!-- Metric Cards with Short Beginner-Friendly Explanations -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-badge">Model Fit</div>
          <div class="metric-value" id="val-r2">{{ metrics.r2_score }}</div>
          <div class="metric-percent">{{ metrics.r2_percentage }}% variance explained</div>
          <h4 class="metric-name">R² Score (Coefficient of Determination)</h4>
          <p class="metric-desc">
            Measures how well the independent features explain student grades. A score of {{ metrics.r2_score }} means {{ metrics.r2_percentage }}% of score variation is captured by our model.
          </p>
        </div>

        <div class="metric-card">
          <div class="metric-badge">Average Error</div>
          <div class="metric-value" id="val-mae">{{ metrics.mae }} <small>marks</small></div>
          <div class="metric-percent">Linear Deviation</div>
          <h4 class="metric-name">Mean Absolute Error (MAE)</h4>
          <p class="metric-desc">
            The average difference between predicted and actual marks. An error of ~{{ metrics.mae }} marks means predictions are typically accurate within 2 marks.
          </p>
        </div>

        <div class="metric-card">
          <div class="metric-badge">Squared Loss</div>
          <div class="metric-value" id="val-mse">{{ metrics.mse }}</div>
          <div class="metric-percent">RMSE: {{ metrics.rmse }} marks</div>
          <h4 class="metric-name">Mean Squared Error (MSE)</h4>
          <p class="metric-desc">
            Average of squared prediction errors. Penalizes larger mistakes heavily. The corresponding RMSE of ~{{ metrics.rmse }} marks shows consistent accuracy.
          </p>
        </div>

        <div class="metric-card">
          <div class="metric-badge">Data Partition</div>
          <div class="metric-value">80 / 20</div>
          <div class="metric-percent">{{ metrics.train_size }} Train / {{ metrics.test_size }} Test</div>
          <h4 class="metric-name">Train / Test Split</h4>
          <p class="metric-desc">
            80% of students were used to train the Linear Regression line, while 20% were kept strictly hidden to evaluate accuracy on unseen records.
          </p>
        </div>
      </div>

      <!-- Fitted Coefficients Table -->
      <div class="card table-card" style="margin-top: 2rem;">
        <div class="table-header-bar">
          <div>
            <h3>Fitted Model Coefficients (Weights)</h3>
            <p class="card-desc">Ordinary Least Squares values learned by Scikit-Learn during training</p>
          </div>
          <div class="intercept-pill">
            Intercept (β₀): <strong>{{ intercept }}</strong>
          </div>
        </div>
        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>Feature Variable (xᵢ)</th>
                <th>Feature Name</th>
                <th>Weight (βᵢ)</th>
                <th>Viva Explanation</th>
              </tr>
            </thead>
            <tbody>
              {% for feat, coef in coefficients.items() %}
              <tr>
                <td><code>{{ feat }}</code></td>
                <td><strong>{{ feature_labels[feat] }}</strong></td>
                <td><span class="badge-coef">+{{ coef }}</span></td>
                <td>
                  Each 1-unit increase adds approximately <strong>{{ coef }} marks</strong> to the student's predicted final performance.
                </td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <!-- 5. About the Project Section -->
  <section id="about" class="section">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Project Overview</span>
        <h2 class="section-title">About the Project</h2>
        <p class="section-subtitle">
          A straightforward implementation of statistical machine learning for college academic demonstrations and viva presentation.
        </p>
      </div>

      <!-- Simple Workflow Sequence -->
      <div class="card workflow-box">
        <h3>Machine Learning Workflow</h3>
        <p class="card-desc">The project follows a standard 5-step machine learning sequence:</p>
        
        <div class="simple-pipeline">
          <div class="pipe-step">
            <span class="step-num">1</span>
            <h4>Dataset</h4>
            <p>Collection of 320 student records with 6 academic indicators.</p>
          </div>
          <div class="pipe-arrow">➔</div>
          <div class="pipe-step">
            <span class="step-num">2</span>
            <h4>Data Preparation</h4>
            <p>Data validation, range checks, and splitting into 80% train / 20% test.</p>
          </div>
          <div class="pipe-arrow">➔</div>
          <div class="pipe-step">
            <span class="step-num">3</span>
            <h4>Train Linear Regression</h4>
            <p>Fitting Ordinary Least Squares (OLS) to calculate optimal β weights.</p>
          </div>
          <div class="pipe-arrow">➔</div>
          <div class="pipe-step">
            <span class="step-num">4</span>
            <h4>Test Model</h4>
            <p>Measuring generalization accuracy on test records using R², MAE, MSE.</p>
          </div>
          <div class="pipe-arrow">➔</div>
          <div class="pipe-step">
            <span class="step-num">5</span>
            <h4>Predict Performance</h4>
            <p>Accepting new student inputs from the web UI to generate live predictions.</p>
          </div>
        </div>
      </div>

      <!-- Technology Stack (Explained in Simple Terms) -->
      <div class="tech-grid" style="margin-top: 1.75rem;">
        <div class="tech-card">
          <div class="tech-icon">🐍</div>
          <h4>Python</h4>
          <p>The core programming language used to build the data processing, mathematical modeling, and web backend.</p>
        </div>
        <div class="tech-card">
          <div class="tech-icon">🌐</div>
          <h4>Flask</h4>
          <p>A lightweight web server framework that handles incoming browser requests and serves the prediction API.</p>
        </div>
        <div class="tech-card">
          <div class="tech-icon">🐼</div>
          <h4>Pandas</h4>
          <p>A data analysis library used to load, read, and manipulate the student performance CSV dataset.</p>
        </div>
        <div class="tech-card">
          <div class="tech-icon">🔢</div>
          <h4>NumPy</h4>
          <p>A mathematical numerical library providing high-performance array operations and feature calculations.</p>
        </div>
        <div class="tech-card">
          <div class="tech-icon">⚙️</div>
          <h4>Scikit-Learn</h4>
          <p>The machine learning library providing the LinearRegression algorithm, train_test_split, and evaluation metrics.</p>
        </div>
      </div>

      <!-- Mathematical Formulation Note -->
      <div class="card formula-note-card" style="margin-top: 1.75rem;">
        <h3>Linear Regression Equation</h3>
        <p class="card-desc">The model predicts the student's score using this linear algebraic equation:</p>
        <div class="math-banner">
          y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + β₄x₄ + β₅x₅ + β₆x₆
        </div>
        <p class="formula-explanation">
          Where <strong>y</strong> is the predicted score, <strong>β₀</strong> is the intercept ({{ intercept }}), <strong>β₁ … β₆</strong> are the learned feature weights, and <strong>x₁ … x₆</strong> are the student's inputs.
        </p>
      </div>
    </div>
  </section>

  <!-- Footer -->
  <footer class="footer">
    <div class="container footer-content">
      <div class="footer-brand">
        <span class="brand-icon-wrapper mini">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
            <path d="M6 12v5c3 3 9 3 12 0v-5"/>
            <circle cx="12" cy="12" r="1.5" fill="currentColor"/>
          </svg>
        </span>
        <span class="footer-title">AcadPredict</span>
      </div>
      <p class="footer-desc">Student Performance Prediction Using Linear Regression</p>
      <p class="footer-copy">Built with Python, Flask, Pandas, and Scikit-Learn • Statistical Machine Learning</p>
    </div>
  </footer>

  <script>
/**
 * AcadPredict - Student Performance Prediction Frontend Script
 * Statistical Machine Learning using Linear Regression (Scikit-Learn)
 */

document.addEventListener('DOMContentLoaded', () => {
  initMobileNav();
  initActiveNavOnScroll();
});

// Mobile navigation toggle
function initMobileNav() {
  const toggleBtn = document.getElementById('mobile-toggle');
  const navLinks = document.getElementById('nav-links');
  if (toggleBtn && navLinks) {
    toggleBtn.addEventListener('click', () => {
      navLinks.classList.toggle('active');
    });

    // Close mobile menu on clicking any navigation link
    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('active');
      });
    });
  }
}

// Active navigation highlight on scroll
function initActiveNavOnScroll() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  const onScroll = () => {
    const scrollY = window.pageYOffset;

    sections.forEach(current => {
      const sectionHeight = current.offsetHeight;
      const sectionTop = current.offsetTop - 120;
      const sectionId = current.getAttribute('id');

      if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
        navLinks.forEach(link => {
          link.classList.remove('active');
          if (link.getAttribute('href') === `#${sectionId}`) {
            link.classList.add('active');
          }
        });
      }
    });
  };

  window.addEventListener('scroll', onScroll, { passive: true });
}

// Quick Sample Pre-fills for easy Viva Presentation
function fillSample(type) {
  const samples = {
    excellent: {
      study_hours: 8.5,
      attendance: 94.0,
      previous_score: 92.0,
      assignment_score: 95.0,
      sleep_hours: 7.5,
      participation_score: 90.0
    },
    good: {
      study_hours: 6.0,
      attendance: 84.0,
      previous_score: 75.0,
      assignment_score: 80.0,
      sleep_hours: 7.0,
      participation_score: 74.0
    },
    average: {
      study_hours: 3.8,
      attendance: 68.0,
      previous_score: 58.0,
      assignment_score: 64.0,
      sleep_hours: 6.2,
      participation_score: 55.0
    },
    struggling: {
      study_hours: 1.8,
      attendance: 56.0,
      previous_score: 44.0,
      assignment_score: 48.0,
      sleep_hours: 5.0,
      participation_score: 42.0
    }
  };

  const data = samples[type];
  if (!data) return;

  for (const [key, val] of Object.entries(data)) {
    const input = document.getElementById(key);
    if (input) input.value = val;
  }

  // Clear any existing error message
  const errorBox = document.getElementById('form-error-msg');
  if (errorBox) {
    errorBox.style.display = 'none';
    errorBox.textContent = '';
  }

  // Automatically trigger prediction for instant viva demo
  const form = document.getElementById('prediction-form');
  if (form) {
    const event = new Event('submit', { cancelable: true });
    form.dispatchEvent(event);
  }
}

// Prediction form submit handler
async function handlePrediction(event) {
  event.preventDefault();

  const form = document.getElementById('prediction-form');
  const btn = document.getElementById('predict-btn');
  const btnText = btn.querySelector('.btn-text');
  const spinner = document.getElementById('spinner');
  const errorBox = document.getElementById('form-error-msg');

  // Clear previous error
  errorBox.style.display = 'none';
  errorBox.textContent = '';

  const formData = new FormData(form);
  const payload = {};
  
  // Validation checks
  const fieldRules = [
    { key: 'study_hours', label: 'Study Hours per Day', min: 0, max: 24 },
    { key: 'attendance', label: 'Attendance Percentage', min: 0, max: 100 },
    { key: 'previous_score', label: 'Previous Exam Score', min: 0, max: 100 },
    { key: 'assignment_score', label: 'Assignment Score', min: 0, max: 100 },
    { key: 'sleep_hours', label: 'Sleep Hours per Day', min: 0, max: 24 },
    { key: 'participation_score', label: 'Class Participation Score', min: 0, max: 100 }
  ];

  for (const rule of fieldRules) {
    const rawVal = formData.get(rule.key);
    if (rawVal === null || rawVal.trim() === '') {
      showError(`Please enter a value for ${rule.label}.`);
      return;
    }
    const num = parseFloat(rawVal);
    if (isNaN(num)) {
      showError(`Please enter a valid number for ${rule.label}.`);
      return;
    }
    if (num < rule.min || num > rule.max) {
      showError(`${rule.label} must be between ${rule.min} and ${rule.max}.`);
      return;
    }
    payload[rule.key] = num;
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.style.display = 'block';
    errorBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  // UI loading state
  btn.disabled = true;
  btnText.textContent = 'Calculating Model...';
  spinner.style.display = 'inline-block';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const res = await response.json();

    if (!res.success) {
      showError('Prediction Error: ' + (res.error || 'Unable to calculate score'));
      return;
    }

    displayPredictionResult(res.prediction);

  } catch (err) {
    console.error('Fetch error:', err);
    showError('Failed to connect to Python backend: ' + err.message);
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Predict Performance';
    spinner.style.display = 'none';
  }
}

// Render prediction output and breakdown
function displayPredictionResult(pred) {
  const placeholder = document.getElementById('result-placeholder');
  const display = document.getElementById('result-display');
  const scoreVal = document.getElementById('res-score');
  const categoryBadge = document.getElementById('res-category');
  const scoreRing = document.getElementById('score-ring');
  const explanation = document.getElementById('res-explanation');
  const recommendations = document.getElementById('res-recommendations');
  const breakdownBody = document.getElementById('breakdown-body');
  const resIntercept = document.getElementById('res-intercept');
  const resFinalCalc = document.getElementById('res-final-calc');

  placeholder.style.display = 'none';
  display.style.display = 'block';

  // Animate score counter
  animateValue(scoreVal, 0, pred.predicted_score, 600);

  // Set category and color themes
  categoryBadge.textContent = pred.category;
  categoryBadge.className = 'category-badge';

  if (pred.category === 'Excellent') {
    categoryBadge.classList.add('badge-excellent');
    scoreRing.style.borderColor = '#059669';
  } else if (pred.category === 'Good') {
    categoryBadge.classList.add('badge-good');
    scoreRing.style.borderColor = '#2563eb';
  } else if (pred.category === 'Average') {
    categoryBadge.classList.add('badge-average');
    scoreRing.style.borderColor = '#d97706';
  } else {
    categoryBadge.classList.add('badge-needs_improvement');
    scoreRing.style.borderColor = '#dc2626';
  }

  // Explanation
  explanation.textContent = pred.explanation;

  // Recommendations
  recommendations.innerHTML = '';
  pred.recommendations.forEach(rec => {
    const li = document.createElement('li');
    li.textContent = rec;
    recommendations.appendChild(li);
  });

  // Linear Regression breakdown table
  if (pred.formula_breakdown) {
    breakdownBody.innerHTML = '';
    const terms = pred.formula_breakdown.terms || [];
    terms.forEach(term => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td><strong>${term.feature}</strong></td>
        <td><code>${term.value}</code></td>
        <td><code>${term.coefficient > 0 ? '+' : ''}${term.coefficient}</code></td>
        <td><strong>${term.contribution > 0 ? '+' : ''}${term.contribution}</strong></td>
      `;
      breakdownBody.appendChild(row);
    });

    resIntercept.textContent = (pred.formula_breakdown.intercept > 0 ? '+' : '') + pred.formula_breakdown.intercept;
    resFinalCalc.textContent = pred.predicted_score + ' / 100';
  }

  // Smooth scroll to results on smaller screens
  if (window.innerWidth < 1024) {
    document.getElementById('result-container').scrollIntoView({ behavior: 'smooth' });
  }
}

// Reset form for "Predict Again" option
function resetPredictionForm() {
  const form = document.getElementById('prediction-form');
  const placeholder = document.getElementById('result-placeholder');
  const display = document.getElementById('result-display');
  const errorBox = document.getElementById('form-error-msg');

  if (form) form.reset();
  if (errorBox) {
    errorBox.style.display = 'none';
    errorBox.textContent = '';
  }

  // Return to ready state
  if (display) display.style.display = 'none';
  if (placeholder) placeholder.style.display = 'block';

  // Smooth scroll back to form
  document.getElementById('predict').scrollIntoView({ behavior: 'smooth' });

  // Focus on the first input
  const firstInput = document.getElementById('study_hours');
  if (firstInput) firstInput.focus();
}

// Counter animation
function animateValue(obj, start, end, duration) {
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    const current = (progress * (end - start) + start).toFixed(1);
    obj.innerHTML = current;
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
}

  </script>
</body>
</html>
"""

# ------------------------------------------------------------------------------
# 5. Main Execution Entry Point
# ------------------------------------------------------------------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    print("=" * 65)
    print("  AcadPredict: Student Performance Prediction Web Application")
    print("  Algorithm: Linear Regression (Statistical Machine Learning)")
    print("=" * 65)
    print("  -> Website starting on: http://127.0.0.1:5000")
    print("  -> Automatically launching in your default browser...")
    print("  -> Press Ctrl+C in terminal / stop in VS Code to quit.")
    print("=" * 65)

    threading.Timer(1.2, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
