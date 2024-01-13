import pandas as pd

# Load the dataset
file_path = 'E:\codsoft\Data Science\DataSets/IMDb Movies India.csv'

# Attempting to load the dataset with different encoding if needed
try:
    movies_data = pd.read_csv(file_path, encoding='ISO-8859-1')
except UnicodeDecodeError:
    movies_data = pd.read_csv(file_path, encoding='latin1')

# Preprocessing Step 1: Handling Missing Values
# Dropping rows where 'Rating' is missing
movies_data_cleaned = movies_data.dropna(subset=['Rating'])

# Preprocessing Step 2: Convert Data Types
# Extracting year from 'Year' column
movies_data_cleaned['Year'] = movies_data_cleaned['Year'].str.extract('(\d{4})').astype(float)

# Converting 'Duration' to numeric by removing 'min' and converting to integer
movies_data_cleaned['Duration'] = movies_data_cleaned['Duration'].str.extract('(\d+)').astype(float)

# Converting 'Votes' to numeric
movies_data_cleaned['Votes'] = movies_data_cleaned['Votes'].str.replace(',', '').astype(float)

# Further Handling of Missing Values
# Imputing 'Duration' with the median
duration_median = movies_data_cleaned['Duration'].median()
movies_data_cleaned['Duration'].fillna(duration_median, inplace=True)

# Imputing categorial columns with the mode
for column in ['Genre', 'Director', 'Actor 1', 'Actor 3']:
    mode_value = movies_data_cleaned[column].mode()[0]
    movies_data_cleaned[column].fillna(mode_value, inplace=True)