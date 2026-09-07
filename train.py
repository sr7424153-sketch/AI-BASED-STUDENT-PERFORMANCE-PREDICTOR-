import pandas as pd
import pickle


# =========================================================
# STUDENT PERFORMANCE PREDICTOR
# RULE-BASED MODEL
# =========================================================

DATA_PATH = "DATASET/Student_data.csv"
MODEL_PATH = "model.pkl"


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("Total Students:", len(df))


# =========================================================
# 2. REQUIRED COLUMNS
# =========================================================

required_columns = [
    "name",
    "attendance",
    "study_hours",
    "previous_marks",
    "assignment_score",
    "internal_marks",
    "previous_cgpa",
    "result"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    print("\nERROR! Missing columns:")
    print(missing)
    exit()


# =========================================================
# 3. CONVERT NUMERIC COLUMNS
# =========================================================

numeric_columns = [
    "attendance",
    "study_hours",
    "previous_marks",
    "assignment_score",
    "internal_marks",
    "previous_cgpa"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_columns)


# =========================================================
# 4. THRESHOLDS
# =========================================================
#
# Previous CGPA:
# < 7      = Previous semester weak
# >= 7     = Previous semester good
#
# Current semester:
# Attendance >= 75       = Good
# Internal >= 50/100     = Good
# Assignment >= 10/20    = Good
#
# =========================================================

CGPA_WEAK = 7

ATTENDANCE_GOOD = 75
INTERNAL_GOOD = 50
ASSIGNMENT_GOOD = 10


# =========================================================
# 5. CURRENT SEMESTER STATUS
# =========================================================

def current_status(row):

    attendance_good = row["attendance"] >= ATTENDANCE_GOOD
    internal_good = row["internal_marks"] >= INTERNAL_GOOD
    assignment_good = row["assignment_score"] >= ASSIGNMENT_GOOD

    good_count = sum([
        attendance_good,
        internal_good,
        assignment_good
    ])

    # 2 or more factors good = Current performance good
    if good_count >= 2:
        return "Good"

    return "Poor"


df["current_status"] = df.apply(current_status, axis=1)


# =========================================================
# 6. FINAL CATEGORY LOGIC
# =========================================================

def get_category(row):

    previous_cgpa = row["previous_cgpa"]
    current = row["current_status"]

    previous_weak = previous_cgpa < CGPA_WEAK


    # -----------------------------------------------------
    # CASE 1:
    # Previous semester weak
    # Current semester also weak
    #
    # => PERMANENT FAIL RISK
    # -----------------------------------------------------

    if previous_weak and current == "Poor":
        return "Permanent Fail Risk"


    # -----------------------------------------------------
    # CASE 2:
    # Previous semester weak
    # Current semester good
    #
    # => NEED ATTENTION
    #
    # Student is improving but previous performance
    # was weak.
    # -----------------------------------------------------

    if previous_weak and current == "Good":
        return "Need Attention"


    # -----------------------------------------------------
    # CASE 3:
    # Previous semester good
    # Current semester poor
    #
    # => CURRENT PERFORMANCE DROP
    # -----------------------------------------------------

    if not previous_weak and current == "Poor":
        return "Current Performance Drop"


    # -----------------------------------------------------
    # CASE 4:
    # Previous semester good
    # Current semester good
    #
    # => GOOD
    # -----------------------------------------------------

    return "Good"


df["category"] = df.apply(get_category, axis=1)


# =========================================================
# 7. FINAL PREDICTION
# =========================================================

def final_prediction(category):

    if category == "Permanent Fail Risk":
        return "Fail"

    return "Pass"


df["predicted_result"] = df["category"].apply(final_prediction)


# =========================================================
# 8. NEED ATTENTION FLAG
# =========================================================

def attention_flag(category):

    if category in [
        "Need Attention",
        "Current Performance Drop"
    ]:
        return "Yes"

    return "No"


df["need_attention"] = df["category"].apply(attention_flag)


# =========================================================
# 9. PERFORMANCE SCORE
# =========================================================
#
# This is only for displaying overall performance.
#
# Attendance       = 30%
# Assignment      = 20%
# Internal         = 30%
# Previous Marks   = 10%
# Previous CGPA    = 10%
#
# Internal marks are assumed to be out of 100.
# =========================================================

df["performance_score"] = (
    (df["attendance"] / 100) * 30
    +
    (df["assignment_score"] / 20) * 20
    +
    (df["internal_marks"] / 100) * 30
    +
    (df["previous_marks"] / 100) * 10
    +
    (df["previous_cgpa"] / 10) * 10
)

df["performance_score"] = df["performance_score"].round(2)


# =========================================================
# 10. MODEL CLASS
# =========================================================

class StudentPerformancePredictor:

    def __init__(self):

        self.features = [
            "attendance",
            "study_hours",
            "previous_marks",
            "assignment_score",
            "internal_marks",
            "previous_cgpa"
        ]

        # Used by app.py for displaying importance
        self.feature_importances_ = [
            0.30,   # Attendance
            0.00,   # Study Hours
            0.10,   # Previous Marks
            0.20,   # Assignment
            0.30,   # Internal
            0.10    # Previous CGPA
        ]


    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, X):

        predictions = []

        for _, row in X.iterrows():

            previous_weak = (
                row["previous_cgpa"] < CGPA_WEAK
            )

            good_count = sum([
                row["attendance"] >= ATTENDANCE_GOOD,
                row["internal_marks"] >= INTERNAL_GOOD,
                row["assignment_score"] >= ASSIGNMENT_GOOD
            ])

            current_good = good_count >= 2


            # Previous weak + current weak
            if previous_weak and not current_good:
                predictions.append("Fail")

            else:
                predictions.append("Pass")

        return predictions


    # =====================================================
    # CATEGORY
    # =====================================================

    def predict_category(self, X):

        categories = []

        for _, row in X.iterrows():

            previous_weak = (
                row["previous_cgpa"] < CGPA_WEAK
            )

            good_count = sum([
                row["attendance"] >= ATTENDANCE_GOOD,
                row["internal_marks"] >= INTERNAL_GOOD,
                row["assignment_score"] >= ASSIGNMENT_GOOD
            ])

            current_good = good_count >= 2


            if previous_weak and not current_good:

                categories.append("Permanent Fail Risk")

            elif previous_weak and current_good:

                categories.append("Need Attention")

            elif not previous_weak and not current_good:

                categories.append("Current Performance Drop")

            else:

                categories.append("Good")

        return categories


    # =====================================================
    # PROBABILITY
    # =====================================================

    def predict_proba(self, X):

        predictions = self.predict(X)

        probabilities = []

        for prediction in predictions:

            if prediction == "Fail":

                probabilities.append([0.85, 0.15])

            else:

                probabilities.append([0.15, 0.85])

        return probabilities


# =========================================================
# 11. CREATE MODEL
# =========================================================

model = StudentPerformancePredictor()


# =========================================================
# 12. SAVE MODEL
# =========================================================

with open(MODEL_PATH, "wb") as file:

    pickle.dump(model, file)


# =========================================================
# 13. SUMMARY
# =========================================================

total = len(df)

good_count = (
    df["category"] == "Good"
).sum()

attention_count = (
    df["category"] == "Need Attention"
).sum()

drop_count = (
    df["category"] == "Current Performance Drop"
).sum()

fail_count = (
    df["category"] == "Permanent Fail Risk"
).sum()


print("\n")
print("=" * 60)
print("       AI STUDENT PERFORMANCE PREDICTOR")
print("=" * 60)

print("\nTOTAL STUDENTS :", total)

print("\nCATEGORY SUMMARY")
print("-" * 60)

print("Good                       :", good_count)
print("Need Attention             :", attention_count)
print("Current Performance Drop   :", drop_count)
print("Permanent Fail Risk        :", fail_count)

print("\nFINAL PREDICTION")
print("-" * 60)

print("Predicted Pass :", (df["predicted_result"] == "Pass").sum())
print("Predicted Fail :", (df["predicted_result"] == "Fail").sum())


# =========================================================
# 14. SHOW SAMPLE RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("SAMPLE STUDENT ANALYSIS")
print("=" * 60)

display_columns = [
    "name",
    "attendance",
    "assignment_score",
    "internal_marks",
    "previous_cgpa",
    "performance_score",
    "category",
    "predicted_result"
]

print(
    df[display_columns]
    .head(20)
    .to_string(index=False)
)


print("\n")
print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nFile:", MODEL_PATH)
print("\nNow run:")
print("streamlit run app.py")