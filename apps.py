import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and features
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.txt", "r") as f:
    feature_list = f.read().split(",")

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.markdown("## 💼 Employee Salary Prediction App")
st.write("This app uses a trained Linear Regression model to predict realistic employee salaries based on your inputs.")

# --- USER INPUTS ---
age = st.number_input("Enter your Age (must be ≥ 18)", min_value=18, max_value=70, value=24, step=1)
experience = st.number_input("Years of Experience (must be ≤ Age - 18)", min_value=0, max_value=age - 18, value=2, step=1)
education = st.selectbox("Select Education Level", ["High School", "Bachelor", "Master", "PhD"])
role = st.selectbox("Select Job Role", ["Developer", "Data Analyst", "Manager", "Researcher", "Other"])
location = st.selectbox("Select Job Location", ["Urban", "Suburban", "Rural"])
company_tier = st.selectbox("Select Company Tier", ["Top Tier", "Mid Tier", "Startup", "Other"])

# --- VALIDATE LOGIC ---
if experience > (age - 18):
    st.warning("⚠️ Experience can't be greater than Age - 18.")
    st.stop()

# --- BUILD INPUT DF ---
user_input = pd.DataFrame([{
    "age": age,
    "experience": experience,
    "education": education,
    "role": role,
    "location": location,
    "company_tier": company_tier
}])

# One-hot encode to match model
input_encoded = pd.get_dummies(user_input)
input_encoded = input_encoded.reindex(columns=feature_list, fill_value=0)

# --- PREDICTION ---
if st.button("🧮 Predict Salary"):
    try:
        # Base prediction from model
        base_salary = model.predict(input_encoded)[0]

        # Step 1: Apply core scaling factor to bring it to realistic range
        scaled_salary = base_salary * 100  # Scale up base prediction

        # Step 2: Clamp to avoid extreme predictions
        scaled_salary = max(20000, min(scaled_salary, 500000))

        # Step 3: Adjust by company tier
        if company_tier == "Top Tier":
            scaled_salary *= 1.4
        elif company_tier == "Startup":
            scaled_salary *= 0.85
        elif company_tier == "Mid Tier":
            scaled_salary *= 1.1

        # Step 4: Adjust by experience
        if experience <= 1:
            scaled_salary *= 0.8
        elif 2 <= experience <= 5:
            scaled_salary *= 1.0
        elif 6 <= experience <= 10:
            scaled_salary *= 1.2
        else:
            scaled_salary *= 1.5

        # Step 5: Bonus by job role (optional)
        if role in ["Manager", "Researcher"]:
            scaled_salary *= 1.1
        elif role == "Other":
            scaled_salary *= 0.9

        # Final salary rounding
        final_salary = int(round(scaled_salary, -2))

        st.success(f"🤑 Estimated Monthly Salary: ₹{final_salary:,}")
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# --- CHART: Experience vs Salary ---
st.markdown("---")
st.subheader("📊 Estimated Salary by Experience (Mock Data)")
exp_range = list(range(0, 21))
salary_range = [int(15000 + 5000 * exp + (exp ** 1.5) * 300) for exp in exp_range]

fig, ax = plt.subplots()
ax.plot(exp_range, salary_range, marker="o", color="green")
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Monthly Salary (₹)")
ax.set_title("Estimated Salary Growth with Experience")
st.pyplot(fig)

# --- PROJECT INFO ---
st.markdown("---")
st.subheader("🔍 About This Project")
st.write("""
**Algorithm Used:** Linear Regression  
**Inputs Used:** Age, Experience, Education, Role, Location, Company Tier  
**Purpose:** Predict realistic monthly salary based on real-world job logic  
**Note:** Results may vary by industry, company, and economic factors.
""")
