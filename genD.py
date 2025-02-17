import pandas as pd
import numpy as np

# Define the rules for generating synthetic data
def generate_synthetic_data(num_samples=1000):
    data = {
        'Age_Group': np.random.choice(['Adolescent', 'Young Adult', 'Elderly'], num_samples),
        'Sleep_Duration': np.zeros(num_samples),
        'Skin_Conductance': np.zeros(num_samples),
        'Blood_Pressure_Systolic': np.zeros(num_samples),
        'Blood_Pressure_Diastolic': np.zeros(num_samples),
        'Body_Temperature': np.zeros(num_samples),
        'Oxygen_Saturation': np.zeros(num_samples),
        'Stress_Level': np.random.choice(['Low', 'Medium', 'High'], num_samples),
        'Depression_Level': np.random.choice(['None', 'Mild', 'Severe'], num_samples)
    }

    for i in range(num_samples):
        age_group = data['Age_Group'][i]
        stress_level = data['Stress_Level'][i]
        depression_level = data['Depression_Level'][i]

        # Generate Sleep Duration based on age group and stress/depression
        if age_group == 'Adolescent':
            normal_sleep = np.random.uniform(8, 10)
        elif age_group == 'Young Adult':
            normal_sleep = np.random.uniform(7, 9)
        else:  # Elderly
            normal_sleep = np.random.uniform(7, 8)

        if stress_level == 'High':
            data['Sleep_Duration'][i] = np.random.uniform(5, 6)
        elif depression_level == 'Severe':
            data['Sleep_Duration'][i] = np.random.uniform(10, 12)
        else:
            data['Sleep_Duration'][i] = normal_sleep

        # Generate Skin Conductance (EDA)
        if stress_level == 'High':
            data['Skin_Conductance'][i] = np.random.uniform(10, 15)
        elif depression_level == 'Severe':
            data['Skin_Conductance'][i] = np.random.uniform(0.5, 5)
        else:
            data['Skin_Conductance'][i] = np.random.uniform(0.5, 5)

        # Generate Blood Pressure
        normal_systolic = 120
        normal_diastolic = 80
        if stress_level == 'High':
            data['Blood_Pressure_Systolic'][i] = normal_systolic + np.random.uniform(10, 30)
            data['Blood_Pressure_Diastolic'][i] = normal_diastolic + np.random.uniform(5, 15)
        elif depression_level == 'Severe':
            data['Blood_Pressure_Systolic'][i] = normal_systolic - np.random.uniform(5, 10)
            data['Blood_Pressure_Diastolic'][i] = normal_diastolic - np.random.uniform(5, 10)
        else:
            data['Blood_Pressure_Systolic'][i] = normal_systolic
            data['Blood_Pressure_Diastolic'][i] = normal_diastolic

        # Generate Body Temperature
        if stress_level == 'High':
            data['Body_Temperature'][i] = np.random.uniform(37.5, 41)
        elif depression_level == 'Severe':
            data['Body_Temperature'][i] = np.random.uniform(35.5, 36.5)
        else:
            data['Body_Temperature'][i] = np.random.uniform(36.1, 37.8)

        # Generate Oxygen Saturation
        if stress_level == 'High':
            data['Oxygen_Saturation'][i] = np.random.uniform(99, 100)
        elif depression_level == 'Severe':
            data['Oxygen_Saturation'][i] = np.random.uniform(94, 95)
        else:
            data['Oxygen_Saturation'][i] = np.random.uniform(95, 100)

    return pd.DataFrame(data)

# Generate the dataset
num_samples = 20000  # Number of samples to generate
dataset = generate_synthetic_data(num_samples)

# Save the dataset to a CSV file
dataset.to_csv('synthetic_health_data.csv', index=False)

print("Synthetic dataset generated and saved to 'synthetic_health_data.csv'.")
print(dataset.head())
