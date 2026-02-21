
import streamlit as st
import pandas as pd
import logic
import data_layer

st.set_page_config(layout="wide")

st.title("FX AI v5 Risk Monitor")

# --- Fetch external risk data ---
feats, meta = data_layer.fetch_external_features("GLOBAL")

st.subheader("Risk Dashboard")

st.write("### Global Risk Index:", feats.get("global_risk_index", 0.0))
st.write("### War Probability:", feats.get("war_probability", 0.0))
st.write("### Financial Stress:", feats.get("financial_stress", 0.0))
st.write("### Macro Risk Score:", feats.get("macro_risk_score", 0.0))

st.divider()

st.write("### External Data Status")
status_rows = []
for k, v in meta.get("parts", {}).items():
    status_rows.append({
        "source": k,
        "ok": v.get("ok", False),
        "error": v.get("error", ""),
        "detail": str(v.get("detail", ""))[:300]
    })

df_status = pd.DataFrame(status_rows)
st.dataframe(df_status)

st.divider()

st.write("### Debug Meta Raw")
st.json(meta)
