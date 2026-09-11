import streamlit as st
# ... keep the model training code the same ...

st.title("Perovskite Solar Cell Efficiency Predictor")

bandgap = st.slider("Bandgap (eV)", 1.2, 2.2, 1.6, 0.01)
defect = st.slider("Defect Density (log)", 13.0, 17.0, 14.5, 0.1)
trap = st.slider("Trap Depth (eV)", 0.1, 1.2, 0.4, 0.01)
grain = st.slider("Grain Size (nm)", 10, 500, 250, 10)

if st.button("Predict Efficiency"):
    result = predict_efficiency(bandgap, defect, trap, grain)
    st.markdown(result)
