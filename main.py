import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import data_analysis as analysis
import model_building as mb
from file_processing import save_to_json

if __name__ == "__main__":
    st.title("Multilayer Perceptron to predict patient's disease!")
    st.subheader("Upload your CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="train_dataset")

    if uploaded_file:
        if "disease_dataset" not in st.session_state:
            disease_dataset = analysis.load_dataset(uploaded_file)
            analysis.check_data_sanity(disease_dataset)
            disease_dataset = analysis.convert_features_to_float(disease_dataset)
            st.session_state["disease_dataset"] = disease_dataset
        else:
            disease_dataset = st.session_state["disease_dataset"]

        st.subheader("Data Preview")
        st.write(disease_dataset.head())

        st.subheader("Save Diseases Into JSON File")
        if st.button("Save"):
            save_to_json(disease_dataset["disease"].unique(), "disease.json")
            st.write("Successfully created JSON...")

        st.subheader("Count Total Symptoms Per Patient And Visualize")
        if st.button("Calculate"):
            disease_dataset = analysis.create_column_total_symptoms(disease_dataset)
            counted = disease_dataset["total_count"].value_counts()
            fig = plt.figure()
            plt.bar(counted.index, height=counted, color='cyan')
            plt.tight_layout()
            plt.xlabel("Symptoms")
            plt.ylabel("Patients")
            plt.title("Total Symptoms Count For Patients")
            plt.xlim(2, 18)
            plt.ylim(0, 900)
            plt.xticks(range(2, 18))
            plt.yticks(range(0, 901, 50))
            st.pyplot(fig)
            st.write(disease_dataset.head(), "\n")

        st.subheader("Transform And Create Diseases Into Integer Column")
        if st.button("Transform"):
            disease_dataset = analysis.create_target_to_int_column(disease_dataset)
            st.write(disease_dataset.head())

        st.subheader("Choose Which Columns Are Necessary To Train The Model")
        # if "selected_columns" not in st.session_state:
        #     st.session_state.selected_columns = list(disease_dataset.columns)
        # valid_selected = [col for col in st.session_state.selected_columns if col in disease_dataset.columns]

        columns = st.multiselect("Choose Columns for the model",
                                 options=disease_dataset.columns,
                                 default=disease_dataset.columns)
        disease_dataset = disease_dataset[columns]

        st.subheader("Split Data and Train Model")
        if st.button("Train model"):
            (
                train_features,
                train_target,
                val_features,
                val_target,
                test_features,
                test_target
            ) = mb.train_val_test_split(disease_dataset)
            st.write("Model is training...")
            mlp_model, history = mb.create_model(
                train_features, train_target,
                val_features, val_target,
                "disease_mlp_model.keras"
            )

            st.write("Model finished training!")
            st.session_state["mlp_model"] = mlp_model
            st.session_state["history"] = history
            st.session_state["test_features"] = test_features
            st.session_state["test_target"] = test_target

            st.pyplot(mb.plot_train_val_loss_functions(history))


        st.subheader("Predict And Execute Functions")
        options = ("How Many Are Correctly Predicted?",
                   "Plot Actual vs. Predicted Values",
                   "Extract Classification Report",
                   "Show Comparison Dataframe")
        option = st.selectbox("Choose operation", options, key="training_box")

        if all(k in st.session_state for k in ["mlp_model", "test_features", "test_target"]):
            mlp_model = st.session_state["mlp_model"]
            test_features = st.session_state["test_features"]
            test_target = st.session_state["test_target"]
            predictions = mb.make_prediction(st.session_state["mlp_model"], test_features)
            test_target = np.array(test_target, dtype=np.int64).flatten()

            if option == options[0]:
                total_data, predicted_true, predicted_false = mb.compare_actual_predicted_values(
                    test_target, predictions)
                st.write(f"Total Values in Data: {total_data}")
                st.write(f"Model Predicted Correctly: {predicted_true}")
                st.write(f"Model Predicted Incorrectly: {predicted_false}")
            elif option == options[1]:
                st.pyplot(mb.plot_actual_predicted_values(test_target, predictions))
            elif option == options[2]:
                st.write(mb.extract_classification_report(test_target, predictions))
            elif option == options[3]:
                st.write(mb.create_comparison_dataframe(test_target, predictions))

        st.subheader("Try The Model With Your Data")
        test_file = st.file_uploader("Choose a CSV file", type="csv", key="test_dataset")
        if test_file:
            test_data = pd.read_csv(test_file)
            mlp_model = st.session_state["mlp_model"]
            test_data = analysis.convert_features_to_float(test_data)
            test_data = analysis.create_target_to_int_column(test_data)
            test_data.drop("disease", axis=1, inplace=True)
            test_data = mb.shuffle_data(test_data)
            features, target = mb.extract_features_target_columns(test_data)

            predictions = mb.make_prediction(mlp_model, features)

            st.write("Choose function")
            options = ("How Many Are Correctly Predicted?",
                       "Plot Actual vs. Predicted Values",
                       "Extract Classification Report",
                       "Show Comparison Dataframe")
            option = st.selectbox("Choose operation", options, key="testing_box")

            target = np.array(target, dtype=np.int64).flatten()

            if option == options[0]:
                total_data, predicted_true, predicted_false = mb.compare_actual_predicted_values(
                    target, predictions)
                st.write(f"Total Values in Data: {total_data}")
                st.write(f"Model Predicted Correctly: {predicted_true}")
                st.write(f"Model Predicted Incorrectly: {predicted_false}")
            elif option == options[1]:
                st.pyplot(mb.plot_actual_predicted_values(target, predictions))
            elif option == options[2]:
                st.write(mb.extract_classification_report(target, predictions))
            elif option == options[3]:
                st.write(mb.create_comparison_dataframe(target, predictions))
