import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model("tourism_revenue_model.h5")

# Streamlit UI
st.title("Tourism Revenue Prediction")
st.write("Enter GDP per capita to predict tourism revenue for a specific year.")

# User input for GDP per capita
user_input = st.number_input("Enter GDP per capita (e.g., 500000):", value=0.0, step=1000.0)

# User input for the specific year (e.g., 2012)
year_input = st.number_input("Enter the year for prediction (e.g., 2012):", value=2012, step=1)

# Submit button to show the result
if st.button("Submit"):
    if user_input > 0:
        # Prepare data for prediction for the year
        input_data = np.array([[user_input]])  # Single GDP input
        
        # Perform prediction
        prediction = model.predict(input_data)
        
        # Display prediction
        st.write(f"Predicted Revenue for {year_input}: {prediction[0][0]:,.2f}")
        
        # Plot the result for the specific year
        plt.figure(figsize=(8, 6))
        plt.plot([year_input], [prediction[0][0]], marker='o', color='skyblue', linestyle='-', linewidth=2)
        
        # Label the point
        plt.text(year_input, prediction[0][0], f"${prediction[0][0]:,.2f}", ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Year')
        plt.ylabel('Predicted Revenue')
        plt.title(f'Tourism Revenue Prediction for {year_input}')
        plt.xticks([year_input])  # Set x-ticks to the specific year
        plt.grid(True)
        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(plt)
    else:
        st.write("Please enter a valid GDP per capita value greater than 0.")
