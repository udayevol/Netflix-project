Netflix dataset containing information on movies and TV shows. The goal is to uncover insights and patterns within the data, examining factors such as release year, country, ratings, and genres. Through visualizations and statistical analysis, we aim to gain a comprehensive understanding of the content distribution and trends on the Netflix platform.


IMPORTING NECESSARY LIBRARIES


<!-- Now we are reading the data in the form of CSV -->

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

<!-- Reading the data -->
netflix_data = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv') 
netflix_data.head(5) #Showing the top5

<!-- Exploring the dataset and there shapes, size and data type. -->


netflix_data.info() #Exploring the dataset.

<!-- We are loading the data to see there basic statistics. -->


netflix_data.describe() #Now we are reading the basic statistics of the dataset.

<!-- Converting the data_added column to date time datatype. -->


# Converting the  data types.
netflix_data['date_added'] = pd.to_datetime(netflix_data['date_added'], errors='coerce')

<!-- Converting the duration column to flot datatype. -->


# Converting the  data types
netflix_data['duration'] = pd.to_numeric(netflix_data['duration'].str.extract('(\d+)', expand=False), errors='coerce')

<!-- Checking for the missing values in the dataset -->


netflix_data.isnull().sum() # checking for the missing values.

<!-- Filling the missing values in the 'country' column with most common country. -->


<!-- Filling missing values with most common country. -->
most_common_country = netflix_data['country'].mode()[0]  
netflix_data['country'].fillna(most_common_country, inplace=True)

<!-- Filling missing values in 'cast' and 'director' columns with 'Not Available'. -->


#Filling the missing values with not available.
netflix_data['cast'].fillna('Not Available', inplace=True)
netflix_data['director'].fillna('Not Available', inplace=True)

<!-- Checking for the missing values in the rating column. -->


#Checking for the missing values
missing_values = netflix_data['rating'].isnull()
(netflix_data[missing_values])

<!-- Filling the missing values manually by checking from the website. -->


# Fill missing values in the 'title' column
netflix_data.loc[netflix_data['title'] == '13TH: A Conversation with Oprah Winfrey & Ava DuVernay', 'rating'] = 'U/A 13+'
netflix_data.loc[netflix_data['title'] == 'Gargantia on the Verdurous Planet', 'rating'] = 'TV-14'
netflix_data.loc[netflix_data['title'] == 'Little Lunch', 'rating'] = 'TV-G'
netflix_data.loc[netflix_data['title'] == 'My Honor Was Loyalty', 'rating'] = 'PG-13'

Creating new columns.


movies = netflix_data[netflix_data['type'] == 'Movie']
tv_shows = netflix_data[netflix_data['type'] == 'TV Show']

Question 1: which rating has the highest number of counts on netflix?

# Creating a Bar chart for rating distribution
plt.figure(figsize=(10, 6))
sns.countplot(x ='rating', data = netflix_data, order = netflix_data['rating'].value_counts().index) #Creating a plot for rating column
plt.title('Distribution of Ratings')  #labelling the title.
plt.xlabel('Rating') #labelling the xlabel.
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count') #labelling the ylabel.
plt.show()
​

The bargraph shows distribution of rating, where TV-MA has highest count.


Question 2: What is percentage of movies and tv shows in netflix?


# Creating a Bar chart for the number of movies and TV shows
movies_shows = netflix_data.groupby('type').type.count()
plt.figure(figsize=(8, 5))
plt.pie(movies_shows,labels=movies_shows.index,  autopct='%1.1f%%', colors = ['green', 'red']) #Creating a pie chart.
plt.title('Number of Movies and TV Shows') #labeling the title.
plt.show()
​
​

The pie chart shows percentage of movie and tv shows, where movie has highest number of percentage compare to tv shows.


Question 3: Which country has the highest number of content on netflix?


# Count the top countries
top_countries = netflix_data['country'].value_counts().head(10)
​
plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='Blues_r') #Creating a plot for top countries.
plt.title('Top Countries Producing Content on Netflix') #labeling the title
plt.xlabel('Number of Titles') #labeling the xlabel
plt.ylabel('Country') #labeling the ylabel
plt.show()
​

The barplot shows top 10 contries with number of titles, where united states has highest number of titles.


Question 4: What is the trend in the number of releases for movies and TV shows on Netflix over the years?


#Creating a release count of movies and tv shows.
release_count = netflix_data.groupby(['release_year', 'type']).size().reset_index(name='count')
plt.figure(figsize=(12, 6))
sns.lineplot(x='release_year', y='count', hue='type', marker='X', data=release_count, palette='Set2') #Creating a plot.
plt.title('Trend of Movies and TV Shows Over the Years on Netflix') #labeling the title.
plt.xlabel('Release Year') #labeling the xlabel.
plt.ylabel('Number of Releases') #labeling the ylabel.
plt.legend(title='Type') #labeling the legend.
plt.show()
​
#The line plot shows the trend of movie and TV show releases on Netflix over the years. It shows the fluctuation in the number of releases for each type.

# Remove rows with missing values in the 'duration' column
netflix_data.dropna(subset=['duration'], inplace=True)
​
# Resetting the index after removing rows
netflix_data.reset_index(drop=True, inplace=True)
​
# Checking for missing values after removal
print(netflix_data.isnull().sum())
​
add Codeadd Markdown
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
​
# Prepare features and target variable
X = netflix_data[['release_year', 'duration']]
y = netflix_data['duration']
​
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)
​
# Predict on the testing set
y_pred = model.predict(X_test)
​
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
​
import matplotlib.pyplot as plt
​
# Plotting train dataset
plt.figure(figsize=(12, 6))
​
plt.subplot(1, 2, 1)
plt.scatter(X_train['release_year'], y_train, color='blue', label='Train Data')
plt.title('Train Data: Release Year vs Duration')
plt.xlabel('Release Year')
plt.ylabel('Duration')
plt.legend()
​
plt.subplot(1, 2, 2)
plt.scatter(X_train['duration'], y_train, color='green', label='Train Data')
plt.title('Train Data: Duration vs Duration')
plt.xlabel('Duration')
plt.ylabel('Duration')
plt.legend()
​
plt.tight_layout()
plt.show()
​
# Plotting test dataset
plt.figure(figsize=(12, 6))
​
plt.subplot(1, 2, 1)
plt.scatter(X_test['release_year'], y_test, color='red', label='Test Data')
plt.title('Test Data: Release Year vs Duration')
plt.xlabel('Release Year')
plt.ylabel('Duration')
plt.legend()
​
plt.subplot(1, 2, 2)
plt.scatter(X_test['duration'], y_test, color='orange', label='Test Data')
plt.title('Test Data: Duration vs Duration')
plt.xlabel('Duration')
plt.ylabel('Duration')
plt.legend()
​
plt.tight_layout()
plt.show()
​
add Codeadd Markdown
import matplotlib.pyplot as plt
​
# Filter out rows where 'type' is 'Movie'
#movies_data = netflix_data[netflix_data['type'] == 'Movie']
​
# Filter out rows where 'type' is 'TV Show'
#tv_shows_data = netflix_data[netflix_data['type'] == 'TV Show']
​
# Plotting scatter plot for movies
plt.figure(figsize=(12, 6))
plt.scatter(netflix_data['release_year'], netflix_data['duration'], color='blue', label='Movies')
plt.title('Release Year vs Duration for Movies')
plt.xlabel('Release Year')
plt.ylabel('Duration')
plt.legend()
plt.grid(True)
plt.show()
​
# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
​
# Preprocessing the data
# Dropping columns not relevant for prediction
data = netflix_data.drop(['show_id', 'title', 'director', 'cast', 'date_added', 'description'], axis=1)
​
# Converting categorical variables to numerical using Label Encoding
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])
data['country'] = le.fit_transform(data['country'])
data['rating'] = le.fit_transform(data['rating'])
data['listed_in'] = le.fit_transform(data['listed_in'])
​
# Filling missing values in 'duration' with mean value
data['duration'].fillna(data['duration'].mean(), inplace=True)
​
# Splitting the data into train and test sets
X = data.drop('type', axis=1)
y = data['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
# Building the Random Forest classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
​
# Making predictions
y_pred = model.predict(X_test)
​
# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
​
add Codeadd Markdown
# Inverse transforming predicted and true labels
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)
​
# Printing the converted labels
print("True labels (Test Set):", y_test_labels)
​
