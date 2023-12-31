from sklearn.preprocessing import StandardScaler, LabelEncoder

CATEGORICAL_COLUMNS = ['merchant', 'category', 'gender', 'job']
NUMERICAL_COLUMNS = ['cc_num', 'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']


def process_input(user_input): 
    """Performs the Preprocessing on the Input by the Users to make it ready for Predictions

    Args:
        user_input (dict): user input

    Returns:
        dict: preprocessed input which is ready to be fed to the Model 
    """
    for col in CATEGORICAL_COLUMNS:
        user_input[col] = LabelEncoder.transform(user_input[col])

        # Normalize numerical features
        user_input[NUMERICAL_COLUMNS] = StandardScaler.transform(user_input[NUMERICAL_COLUMNS])
    return user_input 


