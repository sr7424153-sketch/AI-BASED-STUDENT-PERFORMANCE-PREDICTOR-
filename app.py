import streamlit as st
import pandas as pd
import pickle
import os



st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🤖",
    layout="wide"
)


st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(0,180,255,0.13), transparent 25%),
        radial-gradient(circle at 90% 80%, rgba(170,0,255,0.13), transparent 25%),
        linear-gradient(135deg, #06101d, #0b1728, #06101d);
    background-size: 200% 200%;
    animation: bgMove 12s ease infinite;
}

@keyframes bgMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating particles */
.stApp::before {
    content: "";
    position: fixed;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: rgba(0,210,255,0.65);

    box-shadow:
        100px 120px 20px rgba(0,210,255,0.30),
        250px 300px 25px rgba(170,60,255,0.30),
        500px 150px 20px rgba(0,210,255,0.30),
        700px 400px 25px rgba(170,60,255,0.30),
        900px 180px 20px rgba(0,210,255,0.30),
        1100px 500px 25px rgba(170,60,255,0.30);

    animation: floating 8s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes floating {
    from {
        transform: translateY(20px);
        opacity: 0.35;
    }

    to {
        transform: translateY(-40px);
        opacity: 0.9;
    }
}

.block-container {
    position: relative;
    z-index: 1;
}

/* Main title */
.ai-title {
    text-align: center;
    font-size: 44px;
    font-weight: 900;
    letter-spacing: 1px;
    margin-top: 10px;
    margin-bottom: 5px;
    color: white;
    text-shadow: 0 0 20px rgba(0,210,255,0.55);
}

.ai-subtitle {
    text-align: center;
    color: #b8c7d9;
    font-size: 17px;
    margin-bottom: 30px;
}

/* Cards */
.info-card {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.factor-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(0,200,255,0.15);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.factor-title {
    color: #b8c7d9;
    font-size: 15px;
}

.factor-value {
    color: white;
    font-size: 30px;
    font-weight: 800;
}

/* Result cards */
.pass-card {
    background: rgba(0,180,100,0.13);
    border: 1px solid rgba(0,220,130,0.35);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.fail-card {
    background: rgba(255,50,100,0.13);
    border: 1px solid rgba(255,70,110,0.35);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.attention-card {
    background: rgba(255,170,0,0.12);
    border: 1px solid rgba(255,180,0,0.35);
    border-radius: 18px;
    padding: 20px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border-radius: 15px;
    padding: 10px;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="ai-title">🤖 AI STUDENT PERFORMANCE PREDICTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="ai-subtitle">'
    'Intelligent Student Performance Analysis & Early Risk Detection'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INSTRUCTIONS
# =========================================================

st.markdown("## 📋 Instructions")

st.markdown("""
<div class="info-card">

<b>How to use the system:</b>

<br><br>

1️⃣ Upload your student CSV file.  
<br>
2️⃣ Make sure the required columns are present.  
<br>
3️⃣ The system will automatically analyze every student.  
<br>
4️⃣ Final prediction will be shown as <b>Pass / Fail</b>.  
<br>
5️⃣ Students requiring special attention will be highlighted.  

<br><br>

<b>Important:</b> 
<code>internal_marks</code> already represents the combined 
<b>Sessional + PUT marks</b>. No separate sessional column is required.

</div>
""", unsafe_allow_html=True)


# =========================================================
# FACTORS
# =========================================================

st.markdown("## 🔍 Factors Used for Prediction")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="factor-card">
        <div class="factor-title">Attendance</div>
        <div class="factor-value">30%</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="factor-card">
        <div class="factor-title">Assignment</div>
        <div class="factor-value">20%</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="factor-card">
        <div class="factor-title">Internal</div>
        <div class="factor-value">30%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="factor-card">
        <div class="factor-title">Previous CGPA</div>
        <div class="factor-value">10%</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# CSV UPLOAD
# =========================================================

st.markdown("## 📁 Upload Student Dataset")

uploaded_file = st.file_uploader(
    "Upload Student_data.csv",
    type=["csv"]
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "name",
    "attendance",
    "study_hours",
    "previous_marks",
    "assignment_score",
    "internal_marks",
    "previous_cgpa"
]


# =========================================================
# MODEL LOAD
# =========================================================

model = None

if os.path.exists("model.pkl"):
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
    except Exception:
        model = None


if uploaded_file is not None:

    # -----------------------------------------------------
    # READ CSV
    # -----------------------------------------------------

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ CSV read error: {e}")
        st.stop()

    # -----------------------------------------------------
    # COLUMN CHECK
    # -----------------------------------------------------

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        st.error(
            "❌ Required columns missing: "
            + ", ".join(missing_columns)
        )

        st.info("""
Required columns:

name  
attendance  
study_hours  
previous_marks  
assignment_score  
internal_marks  
previous_cgpa
""")

        st.stop()

    # -----------------------------------------------------
    # CONVERT NUMERIC COLUMNS
    # -----------------------------------------------------

    numeric_columns = [
        "attendance",
        "study_hours",
        "previous_marks",
        "assignment_score",
        "internal_marks",
        "previous_cgpa"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove invalid rows
    df = df.dropna(subset=numeric_columns).copy()

    if len(df) == 0:
        st.error("❌ Dataset me valid student data nahi mila.")
        st.stop()

    # =====================================================
    # PERFORMANCE SCORE
    # =====================================================

    # Normalize each factor

    attendance_score = (
        df["attendance"].clip(0, 100)
    )

    assignment_score = (
        df["assignment_score"].clip(0, 20) / 20 * 100
    )

    internal_score = (
        df["internal_marks"].clip(0, 30) / 30 * 100
    )

    cgpa_score = (
        df["previous_cgpa"].clip(0, 10) / 10 * 100
    )

    study_score = (
        df["study_hours"].clip(0, 10) / 10 * 100
    )

    # -----------------------------------------------------
    # OVERALL PERFORMANCE SCORE
    # -----------------------------------------------------

    df["performance_score"] = (
        attendance_score * 0.30
        + assignment_score * 0.20
        + internal_score * 0.30
        + cgpa_score * 0.10
        + study_score * 0.10
    )

    df["performance_score"] = (
        df["performance_score"].clip(0, 100)
    )

    # =====================================================
    # PREDICTION LOGIC
    # =====================================================

    predictions = []
    attention = []

    for _, row in df.iterrows():

        attendance = row["attendance"]
        assignment = row["assignment_score"]
        internal = row["internal_marks"]
        cgpa = row["previous_cgpa"]
        performance = row["performance_score"]

        # -----------------------------------------------
        # BASE PREDICTION
        # -----------------------------------------------

        prediction = "Pass"

        # Low previous CGPA students
        # + weak current performance

        if cgpa < 7:

            if (
                attendance < 60
                and assignment < 10
                and internal < 15
            ):
                prediction = "Fail"

        # High CGPA students
        # can also be at risk if current semester
        # performance is very poor

        elif cgpa >= 8:

            if (
                attendance < 50
                and assignment < 8
                and internal < 12
            ):
                prediction = "Fail"

        # Medium CGPA
        else:

            if performance < 50:
                prediction = "Fail"

        # Performance score condition
        # 50+ = comparatively better chance of passing

        if performance >= 50:
            if prediction == "Fail":

                # Very weak attendance + academics
                if not (
                    attendance < 45
                    and assignment < 7
                    and internal < 10
                ):
                    prediction = "Pass"

        # ------------------------------------------------
        # ATTENTION LOGIC
        # ------------------------------------------------

        needs_attention = False

        if performance < 50:
            needs_attention = True

        if attendance < 60:
            needs_attention = True

        if assignment < 10:
            needs_attention = True

        if internal < 15:
            needs_attention = True

        if cgpa < 7:
            needs_attention = True

        predictions.append(prediction)
        attention.append(
            "Yes" if needs_attention else "No"
        )

    # =====================================================
    # ADD RESULTS
    # =====================================================

    df["prediction"] = predictions
    df["need_attention"] = attention

    # =====================================================
    # OVERALL SUMMARY
    # =====================================================

    total_students = len(df)

    passed_students = (
        df["prediction"] == "Pass"
    ).sum()

    failed_students = (
        df["prediction"] == "Fail"
    ).sum()

    attention_students = (
        df["need_attention"] == "Yes"
    ).sum()

    pass_percentage = (
        passed_students / total_students * 100
    )

    # =====================================================
    # DASHBOARD
    # =====================================================

    st.markdown("---")

    st.markdown("## 📊 Overall Prediction Result")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "👨‍🎓 Total Students",
            total_students
        )

    with m2:
        st.metric(
            "✅ Predicted Pass",
            passed_students
        )

    with m3:
        st.metric(
            "❌ Predicted Fail",
            failed_students
        )

    with m4:
        st.metric(
            "⚠️ Need Attention",
            attention_students
        )

    st.markdown(
        f"### 📈 Predicted Pass Percentage: "
        f"**{pass_percentage:.2f}%**"
    )

    # =====================================================
    # RESULT MESSAGE
    # =====================================================

    if failed_students == 0:

        st.markdown("""
        <div class="pass-card">
            <h2>🎉 Overall Result: GOOD</h2>
            <p>No student is currently predicted to fail.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="fail-card">
            <h2>⚠️ Students At Risk Detected</h2>
            <p>Some students require academic attention.</p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # ATTENTION STUDENTS
    # =====================================================

    st.markdown("---")

    st.markdown("## ⚠️ Students Requiring Attention")

    attention_df = df[
        df["need_attention"] == "Yes"
    ].copy()

    if len(attention_df) > 0:

        attention_display = attention_df[
            [
                "name",
                "attendance",
                "assignment_score",
                "internal_marks",
                "previous_cgpa",
                "performance_score",
                "prediction"
            ]
        ].copy()

        attention_display["performance_score"] = (
            attention_display["performance_score"]
            .round(2)
        )

        st.dataframe(
            attention_display,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "🎉 No student currently requires special attention."
        )

    # =====================================================
    # FINAL PREDICTIONS
    # =====================================================

    st.markdown("---")

    st.markdown("## 🎯 Final Predictions")

    final_display = df[
        [
            "name",
            "attendance",
            "assignment_score",
            "internal_marks",
            "previous_cgpa",
            "performance_score",
            "prediction",
            "need_attention"
        ]
    ].copy()

    final_display["performance_score"] = (
        final_display["performance_score"]
        .round(2)
    )

    st.dataframe(
        final_display,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # PERFORMANCE DISTRIBUTION
    # =====================================================

    st.markdown("---")

    st.markdown("## 📊 Performance Distribution")

    chart_df = pd.DataFrame({
        "Result": [
            "Pass",
            "Fail",
            "Need Attention"
        ],
        "Students": [
            passed_students,
            failed_students,
            attention_students
        ]
    })

    st.bar_chart(
        chart_df.set_index("Result")
    )

    # =====================================================
    # FACTOR ANALYSIS
    # =====================================================

    st.markdown("---")

    st.markdown("## 🔎 Average Factor Analysis")

    avg1, avg2, avg3, avg4 = st.columns(4)

    with avg1:
        st.metric(
            "Attendance",
            f"{df['attendance'].mean():.1f}%"
        )

    with avg2:
        st.metric(
            "Assignment",
            f"{df['assignment_score'].mean():.1f}/20"
        )

    with avg3:
        st.metric(
            "Internal",
            f"{df['internal_marks'].mean():.1f}/30"
        )

    with avg4:
        st.metric(
            "Previous CGPA",
            f"{df['previous_cgpa'].mean():.2f}"
        )

    # =====================================================
    # DOWNLOAD RESULT
    # =====================================================

    st.markdown("---")

    st.markdown("## 📥 Download Prediction Report")

    csv_output = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Final Prediction CSV",
        data=csv_output,
        file_name="student_prediction_report.csv",
        mime="text/csv"
    )

else:

    st.info(
        "👆 Start by uploading your Student_data.csv file."
    )

    st.markdown("""
    ### 📌 Dataset Format

    Your CSV should contain:

    `name`  
    `attendance` → 0–100  
    `study_hours` → study hours  
    `previous_marks` → 0–100  
    `assignment_score` → 0–20  
    `internal_marks` → 0–30  
    `previous_cgpa` → 0–10  

    **Note:** `internal_marks` = Sessional + PUT combined.
    """)



st.markdown("---")

st.markdown(
    "<center>🤖 AI Student Performance Predictor • "
    "Early Academic Risk Detection System</center>",
    unsafe_allow_html=True
)