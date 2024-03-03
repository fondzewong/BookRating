import pandas as pd # Import pandas library for data manipulation and analysis
import numpy as np  # Import numpy library for numerical operations
from sklearn.model_selection import train_test_split  # Import train_test_split function from sklearn to split data into training and testing sets
from sklearn.ensemble import RandomForestRegressor   # Import RandomForestRegressor from sklearn to use the Random Forest algorithm for regression probles
from sklearn.metrics import mean_squared_error    # Import mean_squared_error from sklearn to evaluate the performance of the regression model
from sklearn.impute import SimpleImputer    # Import SimpleImputer from sklearn to handle missing values in the dataset
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder from sklearn to convert categorical text data into model-understandable numerical data
import warnings # Import warnings library to control warning messages
warnings.filterwarnings('ignore')  # Suppress all warnings to ensure cleaner output during execution
import matplotlib.pyplot as plt  # Import the matplotlib.pyplot module to enable plotting and visualization. This module provides a MATLAB-like interface for creating a wide variety of plots and charts.
import seaborn as sns   # Import the seaborn module, a statistical data visualization library built on top of matplotlib that offers a higher-level interface for drawing attractive and informative statistical graphics.


# Loading the dataset using a relative path
file_path = "Books.csv" 
books_df = pd.read_csv(file_path, sep=',')

# Display the first 5 rows of the books dataframe to get an overview of the data
books_df.head(5)

# Display the last 5 rows of the books dataframe to get an overview of the data
books_df.tail(5)

#Assessing the extent of missing data in your dataset. 
# Calculate the sum of null (missing) values in each column of the books dataframe

books_df.isnull().sum()

#There are trailing space characters in front of the num_pages column. So we remove them 
books_df.rename(columns={'  num_pages': 'num_pages'}, inplace=True)

# Display data types of each column to verify consistency
books_df.dtypes

# Calculate and display the number of missing values in each column of the 'books_df' DataFrame. This helps in identifying which columns have missing data and may require cleaning or imputation before further analysis or modeling.

books_df.isna().sum()

# We check if there are any duplicated rows in the 'books_df' DataFrame. 
#This returns True if there are duplicates, and False otherwise, indicating whether data deduplication may be necessary.

books_df.duplicated().any()

# Summary statistics for numerical columns to check ranges and constraints
books_df.describe()

# Visualize outliers using box plots for numerical column 'num_pages'
books_df.boxplot(column=['num_pages']) 
plt.show()

# Visualize outliers using box plots for numerical column 'ratings_count'
books_df.boxplot(column=['ratings_count']) 
plt.show()

# Visualize outliers using box plots for numerical column 'text_reviews_count'

books_df.boxplot(column=['text_reviews_count']) 
plt.show()

# Calculate the number of authors per book by splitting the 'authors' column and counting the elements
# Calculate the number of publishers per book by splitting the 'publisher' column and counting the elements
# Convert 'publication_date' column to datetime format, coercing any errors

# Initial data preprocessing
books_df['num_authors'] = books_df['authors'].apply(lambda x: len(x.split('/')))
books_df['num_publishers'] = books_df['publisher'].apply(lambda x: len(x.split('/')))

missing_pub_date_book_ids = books_df[books_df.publication_date.isna()]['bookID']
print(missing_pub_date_book_ids)


#Assuming books_df['publication_date'] is already converted to datetime with 'coerce' for errors
books_df['publication_date'] = pd.to_datetime(books_df['publication_date'], errors='coerce')

# Additional feature engineering
books_df['author_popularity'] = books_df['authors'].str.split('/').explode().map(books_df['authors'].str.split('/').explode().value_counts()).groupby(level=0).sum()
books_df['publisher_popularity'] = books_df['publisher'].str.split('/').explode().map(books_df['publisher'].str.split('/').explode().value_counts()).groupby(level=0).sum()
books_df['title_occurrences'] = books_df['title'].map(books_df['title'].value_counts())
books_df = pd.concat([books_df, pd.get_dummies(books_df['language_code'], prefix='lang')], axis=1)
current_year = pd.to_datetime('now').year
books_df['years_since_publication'] = current_year - books_df['publication_date'].dt.year

# Define bins for publication period categorization
bins = pd.to_datetime(['1900-01-01', '1999-12-31', '2009-12-31', '2019-12-31', '2023-12-31'])
labels = ['Before 2000', '2000-2009', '2010-2019', '2020-2023']
books_df['publication_period'] = pd.cut(books_df['publication_date'], bins=bins, labels=labels)

#Create 3 new columns for the year, the month and the day of publication
books_df['year'] = books_df['publication_date'].dt.year
books_df['month'] = books_df['publication_date'].dt.month
books_df['day'] = books_df['publication_date'].dt.day

books_df.isnull().sum()

numerical_cols = ['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count', 'num_authors', 'num_publishers', 'author_popularity', 'publisher_popularity', 'title_occurrences', 'years_since_publication']
numerical_df = books_df[numerical_cols]
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(10, 8))  # Adjust the size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.show()

# Select features and target
features = ['ratings_count', 'num_pages', 'years_since_publication', 'publisher_popularity', 'author_popularity', 'text_reviews_count'] + [col for col in books_df.columns if col.startswith('lang_')]
target = 'average_rating'
X = books_df[features]
y = books_df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_imputed, y_train)

# Evaluate the model
y_pred = model.predict(X_test_imputed)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print results
print(f'RMSE: {rmse}')

modelling_df=books_df

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Selecting a subset of variables for the model (example)
features = ['num_pages', 'ratings_count', 'text_reviews_count', 'num_authors', 'publisher_popularity']
target = 'average_rating'

X = modelling_df[features]
y = modelling_df[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Calculate and print the R^2 and Mean Squared Error (MSE)
print("R^2 (Coefficient of Determination):", r2_score(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))




