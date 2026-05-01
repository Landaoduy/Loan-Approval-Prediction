import io
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction Chatbot",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Loan Approval Prediction Chatbot")
st.caption("Chatbot demo for your tuned XGBoost loan approval model")


# -----------------------------
# HELPERS
# -----------------------------
def apply_caps(X_df, caps):
    X_out = X_df.copy()
    for col, (lower, upper) in caps.items():
        if col in X_out.columns:
            X_out[col] = X_out[col].clip(lower=lower, upper=upper)
    return X_out


def preprocess_single_input(input_df, scaler, imputer, iqr_caps, feature_names):
    feature_names = list(feature_names)  # FIX 1: ensure list, not Index
    input_df = input_df[feature_names].copy()
    input_df = apply_caps(input_df, iqr_caps)
    scaled = scaler.transform(input_df)
    imputed = imputer.transform(scaled)
    return pd.DataFrame(imputed, columns=feature_names)


def rule_based_reasoning(sample_series, shap_row, top_n=5):
    contrib = pd.DataFrame(
        {
            "Feature": sample_series.index,
            "Value": sample_series.values,
            "SHAP": shap_row.values,
        }
    )
    contrib["AbsSHAP"] = contrib["SHAP"].abs()
    contrib = contrib.sort_values("AbsSHAP", ascending=False).head(top_n)

    positive = contrib[contrib["SHAP"] > 0]
    negative = contrib[contrib["SHAP"] < 0]

    reasons = []
    if len(positive) > 0:
        reasons.append(
            "Main factors pushing toward approval: "
            + ", ".join(positive["Feature"].tolist())
            + "."
        )
    if len(negative) > 0:
        reasons.append(
            "Main factors pushing toward rejection: "
            + ", ".join(negative["Feature"].tolist())
            + "."
        )

    if not reasons:
        reasons.append("The model did not show strong feature pushes for this case.")

    return reasons, contrib[["Feature", "Value", "SHAP"]]


def make_shap_waterfall_figure(shap_row, max_display=10):
    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_row, max_display=max_display, show=False)
    plt.tight_layout()
    return fig


def run_batch_predictions(csv_df, artifacts):
    """
    Takes a raw uploaded DataFrame, runs the full preprocessing pipeline,
    and returns a results DataFrame with Decision, Probability, and top SHAP drivers.
    Missing model columns are filled with NaN (imputer handles them).
    Extra columns not in feature_names are silently ignored.
    """
    feature_names = artifacts["feature_names"]

    # Fill missing columns with NaN, drop extra columns
    for col in feature_names:
        if col not in csv_df.columns:
            csv_df[col] = np.nan

    input_ready = preprocess_single_input(
        csv_df,
        artifacts["scaler"],
        artifacts["imputer"],
        artifacts["iqr_caps"],
        feature_names,
    )

    probs = artifacts["model"].predict_proba(input_ready)[:, 1]
    labels = (probs >= artifacts["best_threshold"]).astype(int)

    # Top positive and negative SHAP driver per row
    shap_vals = artifacts["explainer"](input_ready)
    shap_matrix = pd.DataFrame(shap_vals.values, columns=feature_names)

    def top_drivers(row, n=2):
        sorted_row = row.abs().sort_values(ascending=False)
        top = sorted_row.head(n).index.tolist()
        pos = [f for f in top if row[f] > 0]
        neg = [f for f in top if row[f] < 0]
        return (
            ", ".join(pos) if pos else "—",
            ", ".join(neg) if neg else "—",
        )

    drivers = shap_matrix.apply(top_drivers, axis=1, result_type="expand")
    drivers.columns = ["Top Approval Drivers", "Top Rejection Drivers"]

    results = pd.DataFrame({
        "Customer #": range(1, len(csv_df) + 1),
        "Decision": ["Approved ✅" if l == 1 else "Not Approved ❌" for l in labels],
        "Approval Probability": probs.round(3),
    })
    results = pd.concat([results, drivers], axis=1)
    return results, shap_vals, input_ready


def build_required_fields(feature_names):
    preferred = [
        "RiskScore",
        "TotalDebtToIncomeRatio",
        "InterestRate",
        "AnnualIncome",
        "LoanAmount",
    ]
    return [f for f in preferred if f in list(feature_names)]


# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    iqr_caps = joblib.load("iqr_caps.pkl")
    feature_names = list(joblib.load("feature_names.pkl"))  # FIX 1: always a plain list
    best_threshold = joblib.load("best_threshold.pkl")
    metrics_df = joblib.load("metrics_df.pkl")
    X_train_imputed_df = joblib.load("X_train_imputed_df.pkl")
    X_test_imputed_df = joblib.load("X_test_imputed_df.pkl")
    # shap_values_test loaded but not used — kept for potential future dashboard use

    explainer = shap.Explainer(model, X_train_imputed_df)

    return {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "iqr_caps": iqr_caps,
        "feature_names": feature_names,
        "best_threshold": best_threshold,
        "metrics_df": metrics_df,
        "X_train_imputed_df": X_train_imputed_df,
        "X_test_imputed_df": X_test_imputed_df,
        "explainer": explainer,
    }


try:
    artifacts = load_artifacts()
except Exception as e:
    st.error(
        "Artifact files are missing. Please make sure these files are in the same folder as app.py: "
        "best_xgb_model.pkl, scaler.pkl, imputer.pkl, iqr_caps.pkl, feature_names.pkl, "
        "best_threshold.pkl, metrics_df.pkl, X_train_imputed_df.pkl, X_test_imputed_df.pkl"
    )
    st.exception(e)
    st.stop()


# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you today? If you want to check a customer's loan approval, just tell me that.",
        }
    ]

if "current_stage" not in st.session_state:
    st.session_state.current_stage = "intro"

required_fields = build_required_fields(artifacts["feature_names"])
metrics_map = dict(zip(artifacts["metrics_df"]["Metric"], artifacts["metrics_df"]["Value"]))


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    # ---- Batch CSV Upload ----
    st.header("📂 Batch Prediction")
    st.caption(
        "Upload a CSV file with one customer per row. "
        "Column names must match the model's feature names. "
        "Missing columns are filled with NaN and imputed automatically."
    )

    with st.expander("Required feature columns", expanded=False):
        st.code("\n".join(artifacts["feature_names"]))

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(csv_df)} rows, {len(csv_df.columns)} columns.")

            if st.button("▶ Run Batch Predictions", use_container_width=True):
                with st.spinner("Running predictions..."):
                    results_df, batch_shap_vals, batch_input = run_batch_predictions(
                        csv_df.copy(), artifacts
                    )

                st.session_state["batch_results"] = results_df
                st.session_state["batch_shap_vals"] = batch_shap_vals
                st.session_state["batch_input"] = batch_input

        except Exception as e:
            st.error(f"Could not read the file: {e}")

    # Show batch results if available
    if "batch_results" in st.session_state:
        results_df = st.session_state["batch_results"]
        batch_shap_vals = st.session_state["batch_shap_vals"]
        batch_input = st.session_state["batch_input"]

        st.markdown("---")
        st.subheader("Batch Results")

        approved = (results_df["Decision"].str.startswith("Approved")).sum()
        rejected = len(results_df) - approved
        col1, col2 = st.columns(2)
        col1.metric("Approved", approved)
        col2.metric("Not Approved", rejected)

        st.dataframe(results_df, use_container_width=True, height=220)

        # Download results
        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Results CSV",
            data=csv_out,
            file_name="loan_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # SHAP waterfall for a selected customer
        st.markdown("---")
        st.subheader("Inspect a Customer")
        selected_idx = st.number_input(
            "Customer # to inspect",
            min_value=1,
            max_value=len(results_df),
            value=1,
            step=1,
        ) - 1  # convert to 0-based index

        row = results_df.iloc[selected_idx]
        st.markdown(
            f"**Decision:** {row['Decision']}  \n"
            f"**Probability:** {row['Approval Probability']}"
        )
        with st.expander("SHAP waterfall", expanded=False):
            st.pyplot(make_shap_waterfall_figure(batch_shap_vals[selected_idx], max_display=10))

        if st.button("Clear Batch Results", use_container_width=True):
            del st.session_state["batch_results"]
            del st.session_state["batch_shap_vals"]
            del st.session_state["batch_input"]
            st.rerun()

    st.markdown("---")

    # ---- Chat History ----
    st.header("💬 Chat History")
    for i, msg in enumerate(st.session_state.messages):
        who = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
        preview = msg['content'].replace("*", "").replace("#", "").replace("`", "")
        st.caption(f"{i+1}. {who}: {preview[:70]}")

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! How can I help you today? If you want to check a customer's loan approval, just tell me that.",
            }
        ]
        st.session_state.current_stage = "intro"
        st.rerun()

# -----------------------------
# MAIN CHAT WINDOW
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Type your message here")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    prompt_lower = prompt.lower()

    # FIX 2: Reset stage so users can run multiple predictions without clearing chat
    if st.session_state.current_stage == "done":
        st.session_state.current_stage = "intro"

    if st.session_state.current_stage == "collect_features":
        try:
            pairs = [x.strip() for x in prompt.split(",") if x.strip()]
            parsed = {}
            for pair in pairs:
                key, value = pair.split("=")
                parsed[key.strip()] = float(value.strip())

            missing_required = [k for k in required_fields if k not in parsed]
            if missing_required:
                bot_reply = (
                    "I still need these required features: "
                    + ", ".join(missing_required)
                    + "."
                )
                with st.chat_message("assistant"):
                    st.markdown(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            else:
                # FIX 3: build all_features from list (not Index) to avoid unexpected behavior
                all_features = {col: np.nan for col in artifacts["feature_names"]}
                all_features.update(parsed)
                input_df = pd.DataFrame([all_features])

                input_ready = preprocess_single_input(
                    input_df,
                    artifacts["scaler"],
                    artifacts["imputer"],
                    artifacts["iqr_caps"],
                    artifacts["feature_names"],
                )

                pred_prob = artifacts["model"].predict_proba(input_ready)[:, 1][0]
                pred_label = int(pred_prob >= artifacts["best_threshold"])
                decision = "Yes - Approved" if pred_label == 1 else "No - Not Approved"

                shap_single = artifacts["explainer"](input_ready)[0]
                reasons, contrib_df = rule_based_reasoning(
                    input_ready.iloc[0], shap_single, top_n=5
                )

                bot_reply = (
                    f"### Prediction Result\n"
                    f"**Decision:** {decision}\n\n"
                    f"**Predicted approval probability:** {pred_prob:.3f}\n"
                    f"**Decision threshold:** {artifacts['best_threshold']:.2f}\n\n"
                    f"### Reasoning\n"
                    + "\n".join([f"- {r}" for r in reasons])
                    + "\n\n---\n_Want to check another customer? Just say so!_"
                )

                with st.chat_message("assistant"):
                    st.markdown(bot_reply)

                    with st.expander("See model performance"):
                        st.write(f"Accuracy: {metrics_map['Accuracy']:.3f}")
                        st.write(f"Precision: {metrics_map['Precision']:.3f}")
                        st.write(f"Recall: {metrics_map['Recall']:.3f}")
                        st.write(f"F1-score: {metrics_map['F1-score']:.3f}")
                        st.write(f"Decision threshold: {artifacts['best_threshold']:.2f}")

                    with st.expander("See top feature contributions"):
                        st.dataframe(contrib_df.round(4), use_container_width=True)

                    with st.expander("See explanation graph"):
                        st.pyplot(make_shap_waterfall_figure(shap_single, max_display=10))

                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.session_state.current_stage = "done"

        except Exception as e:
            bot_reply = (
                "I could not read that input. Use this format:\n\n"
                "`RiskScore=72, TotalDebtToIncomeRatio=0.35, InterestRate=9.5, AnnualIncome=65000, LoanAmount=15000`\n\n"
                f"Error details: {e}"
            )
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    elif "loan" in prompt_lower or "approve" in prompt_lower or "customer" in prompt_lower:
        st.session_state.current_stage = "collect_features"
        bot_reply = (
            "Please enter the customer information in `feature=value` format, separated by commas.\n\n"
            + "Required features:\n"
            + "\n".join([f"- {field}" for field in required_fields])
            + "\n\nExample:\n"
            + "`RiskScore=72, TotalDebtToIncomeRatio=0.35, InterestRate=9.5, AnnualIncome=65000, LoanAmount=15000`"
        )

        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    else:
        bot_reply = "Start by saying you want to check a customer for a loan."
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})