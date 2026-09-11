"""
Perovskite Solar Cell Efficiency Predictor
Streamlit App
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ============================
#  TRAIN THE MODEL (cached so it runs once)
# ============================
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 2000

    bandgaps = np.random.uniform(1.2, 2.2, n_samples)
    log_defects = np.random.uniform(13, 17, n_samples)
    traps = np.random.uniform(0.1, 1.2, n_samples)
    grains = np.random.uniform(10, 500, n_samples)

    # Realistic formula: Gaussian peak at 1.5 eV
    base_eff = 25 * np.exp(-((bandgaps - 1.5) ** 2) / (2 * 0.25 ** 2))
    defect_penalty = (log_defects - 13) * 1.5
    trap_penalty = traps ** 1.5 * 5
    grain_bonus = np.log10(grains) * 2.5

    efficiency = base_eff - defect_penalty - trap_penalty + grain_bonus
    efficiency = np.clip(efficiency + np.random.normal(0, 0.5, n_samples), 5, 30)

    df = pd.DataFrame({
        "bandgap": bandgaps,
        "log_defect_density": log_defects,
        "trap_depth": traps,
        "grain_size": grains,
        "efficiency": efficiency,
    })

    X = df.drop("efficiency", axis=1)
    y = df["efficiency"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    importances = dict(zip(X.columns, model.feature_importances_))

    return model, r2, mae, importances


model, R2, MAE, IMPORTANCES = train_model()

# ============================
#  PREDICTION FUNCTION
# ============================
def predict_efficiency(bandgap, defect_density_log, trap_depth, grain_size):
    try:
        input_data = pd.DataFrame({
            "bandgap": [bandgap],
            "log_defect_density": [defect_density_log],
            "trap_depth": [trap_depth],
            "grain_size": [grain_size],
        })

        prediction = model.predict(input_data)[0]

        if prediction > 20:
            quality = "Excellent"
            advice = "This configuration would produce high-efficiency cells."
        elif prediction > 15:
            quality = "Good"
            advice = "Small improvements in defect density could boost efficiency."
        else:
            quality = "Poor"
            advice = "Consider reducing defect density or optimizing bandgap."

        return f"""
### Predicted Efficiency: {prediction:.2f}%

**Quality:** {quality}

**Advice:** {advice}

---

**Feature Importance (from model):**
- Bandgap: {IMPORTANCES['bandgap']*100:.1f}%
- Defect Density: {IMPORTANCES['log_defect_density']*100:.1f}%
- Trap Depth: {IMPORTANCES['trap_depth']*100:.1f}%
- Grain Size: {IMPORTANCES['grain_size']*100:.1f}%

**Model Performance (on test set):**
- R2 Score: {R2:.3f}
- Mean Absolute Error: {MAE:.2f}%

*Model: Random Forest Regressor (100 trees)*
"""
    except Exception as e:
        return f"Error: {str(e)}"


# ============================
#  STREAMLIT UI
# ============================
st.set_page_config(
    page_title="Perovskite Solar Cell Predictor",
    layout="centered"
)

st.title("Perovskite Solar Cell Efficiency Predictor")

st.write(
    "Predict solar cell efficiency from condensed matter defect properties. "
    "Adjust the sliders below and click Predict Efficiency."
)

st.caption(
    "Note: This model is trained on synthetic data for demonstration purposes."
)

# Input sliders
bandgap = st.slider("Bandgap (eV)", 1.2, 2.2, 1.6, 0.01)
defect = st.slider("Defect Density (log scale, cm^-3)", 13.0, 17.0, 14.5, 0.1)
trap = st.slider("Trap Depth (eV)", 0.1, 1.2, 0.4, 0.01)
grain = st.slider("Grain Size (nm)", 10, 500, 250, 10)

# Predict button
if st.button("Predict Efficiency"):
    result = predict_efficiency(bandgap, defect, trap, grain)
    st.markdown(result)

st.markdown("---")
st.write(
    "Built with Random Forest regression on 2,000 synthetic perovskite samples."
)
