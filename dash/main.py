"""ML Dashboard - Streamlit UI for model training and inference."""

import streamlit as st
import requests
import json
import os
from datetime import datetime

API_URL = "http://ml-api:80"
st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("ML Dashboard")

tab_datasets, tab_training, tab_inference = st.tabs(
    ["Datasets", "Training", "Inference"]
)

with tab_datasets:
    st.header("Dataset Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Available Datasets")
        try:
            res = requests.get(f"{API_URL}/datasets", timeout=5)
            if res.status_code == 200:
                datasets = res.json().get("datasets", [])
                if datasets:
                    for ds in datasets:
                        col_name, col_download, col_delete = st.columns([2, 1, 1])
                        with col_name:
                            st.write(f"• {ds}")
                        with col_download:
                            if st.button("Download", key=f"down_ds_{ds}"):
                                try:
                                    res = requests.get(
                                        f"{API_URL}/datasets/download/{ds}", timeout=10
                                    )
                                    if res.status_code == 200:
                                        st.download_button(
                                            label="Save",
                                            data=res.content,
                                            file_name=ds,
                                            mime="text/csv",
                                        )
                                    else:
                                        st.error(f"Error: {res.text}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        with col_delete:
                            if st.button("Delete", key=f"del_ds_{ds}"):
                                try:
                                    res = requests.delete(
                                        f"{API_URL}/datasets/{ds}", timeout=10
                                    )
                                    if res.status_code == 200:
                                        st.success(f"Deleted: {ds}")
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {res.text}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                else:
                    st.info("No datasets")
        except Exception as e:
            st.error(f"Error: {e}")

    with col2:
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader("CSV or JSON", type=["csv", "json"])

        if uploaded_file and st.button("Upload"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file)}
                res = requests.post(
                    f"{API_URL}/datasets/upload", files=files, timeout=30
                )
                if res.status_code == 200:
                    st.success(f"Uploaded: {uploaded_file.name}")
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Error: {e}")

with tab_training:
    st.header("Model Training")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Model Type")
        model_type = st.selectbox(
            "Select", ["linear_regression", "decision_tree", "random_forest"]
        )

    with col2:
        st.subheader("Dataset")
        try:
            res = requests.get(f"{API_URL}/datasets", timeout=5)
            datasets = res.json().get("datasets", [])
            dataset_path = st.selectbox(
                "Select", datasets if datasets else ["No datasets"]
            )
        except:
            dataset_path = st.text_input("Dataset filename")

    with col3:
        st.write("")
        if st.button("Train Model", key="train"):
            try:
                res = requests.post(
                    f"{API_URL}/models/train",
                    params={"model_type": model_type, "dataset_path": dataset_path},
                    timeout=120,
                )
                if res.status_code == 200:
                    result = res.json()
                    st.success("Training completed!")
                    st.json(result)
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.subheader("Trained Models")
    try:
        res = requests.get(f"{API_URL}/models", timeout=5)
        if res.status_code == 200:
            models_list = res.json().get("models", [])
            if models_list:
                for model in models_list:
                    model_id = (
                        model.get("model_id") if isinstance(model, dict) else model
                    )
                    model_type = (
                        model.get("type", "") if isinstance(model, dict) else ""
                    )
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"Model: {model_id}")
                    with col2:
                        if st.button("Delete", key=f"del_{model_id}"):
                            try:
                                res = requests.delete(
                                    f"{API_URL}/models/{model_id}", timeout=10
                                )
                                if res.status_code == 200:
                                    st.success(f"Deleted: {model_id}")
                                    st.rerun()
                                else:
                                    st.error(f"Delete error: {res.text}")
                            except Exception as e:
                                st.error(f"Error: {e}")
            else:
                st.info("No models trained")
    except Exception as e:
        st.error(f"Error: {e}")

with tab_inference:
    st.header("Prediction")

    try:
        res = requests.get(f"{API_URL}/models", timeout=5)
        models_data = res.json().get("models", [])
        models_list = [
            m.get("model_id") if isinstance(m, dict) else m for m in models_data
        ]
    except:
        models_list = []

    if not models_list:
        st.warning("No trained models available")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            model_id = st.selectbox("Model", models_list)

        with col2:
            st.write("")
            if st.button("Get Info"):
                try:
                    res = requests.get(f"{API_URL}/models/{model_id}/info", timeout=5)
                    if res.status_code == 200:
                        st.json(res.json())
                except:
                    st.info("Info not available")

        st.subheader("Features (JSON)")
        features_json = st.text_area(
            "Enter JSON", '{"feature_1": 1.0, "feature_2": 2.0}', height=80
        )

        if st.button("Predict"):
            try:
                features = json.loads(features_json)
                api_timeout = int(os.environ.get("API_REQUEST_TIMEOUT", "300"))
                res = requests.post(
                    f"{API_URL}/models/{model_id}/predict",
                    json=features,
                    timeout=api_timeout,
                )

                if res.status_code == 200:
                    prediction = res.json()
                    st.success("Prediction:")
                    st.json(prediction)
                else:
                    st.error(f"Error: {res.text}")
            except json.JSONDecodeError:
                st.error("Invalid JSON")
            except Exception as e:
                st.error(f"Error: {e}")
