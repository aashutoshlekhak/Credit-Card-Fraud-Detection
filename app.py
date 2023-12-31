import streamlit as st
import pandas as pd
import joblib
import pickle

# Load the trained RandomForestClassifier model
# model = joblib.load("model/rf_model.joblib")


# Function to preprocess input dataa
def preprocess_input_with_pipeline(data, pipeline_path="model/model_pipeline.pkl"):
    # Load the preprocessed pipeline from the pickle file
    with open(pipeline_path, "rb") as file:
        preprocessor_pipeline = pickle.load(file)

    # Apply the preprocessor pipeline to the input data
    prediction = preprocessor_pipeline.predict(data)

    return prediction


# Streamlit app
def main():
    st.title("Fraud Detection App")

    # Get user input for prediction
    cc_num = st.text_input("Enter Credit Card Number:")
    first = st.text_input("Enter First Name:")
    last = st.text_input("Enter Last Name:")
    merchant = st.text_input("Enter Merchant:")
    zip_code = st.text_input("Enter ZIP Code:")
    gender = st.selectbox("Select Gender", ["F", "M"])
    state = st.text_input("Enter State:")
    amt = st.number_input("Enter Transaction Amount:")
    lat = st.number_input("Enter Latitude:")
    long = st.number_input("Enter Longitude:")
    merch_lat = st.number_input("Enter Merchant Latitude:")
    merch_long = st.number_input("Enter Merchant Longitude:")

    # Create a dictionary with user input
    input_data = {
        "cc_num": cc_num,
        "first": first,
        "last": last,
        "merchant": merchant,
        "zip": zip_code,
        "gender": gender,
        "state": state,
        "amt": amt,
        "lat": lat,
        "long": long,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
    }

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
    try:
        # Make predictions only when the predict button is clicked
        if st.button("Predict"):

            # Make predictions
            prediction = preprocess_input_with_pipeline(input_df)

            # Display the prediction
            st.subheader("Prediction:")
            if prediction[0] == 1:
                st.warning("Potential Fraud!")
            else:
                st.success("No Fraud Detected.")
    except ValueError: 
        st.warning("Potential Fraud") # If The Values aren't in database flag it as Fraud


# Run the app
if __name__ == "__main__":
    main()
