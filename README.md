# **California House Price Prediction**
This dataset is frequently utilized in a well-regarded book on machine learning, written by Aurélien Géron. It is a great starting point for individuals looking to dive into the field as it requires minimal data preprocessing, features a straightforward set of variables, and strikes the ideal balance between simplicity and complexity.
The data presents information gathered from a census conducted in California in the year 1990. While it may not be applicable for forecasting current real estate values like other datasets such as Zillow Zestimate, it provides a user-friendly introduction to the fundamentals of machine learning.
### Get the Data.
**Download the Data**
<br>
To download and extract the housing data in a CSV file, you can either use your web browser and run a command or create a function to automate the process. This is helpful if the data changes frequently or needs to be installed on multiple machines.


<img src="https://github.com/SumitxThokar/California-House-Price-Prediction/blob/main/img/fetch.jpg">

### Data Exploration.
Let's take a quick look at the dataframe using **head()** method. The head() method is used to display the first n rows of a pandas DataFrame, where n is an optional parameter. By default, head() returns the first five rows of the DataFrame. This method is often used to quickly inspect the first few rows of a DataFrame to get a feel for the data, or to make sure that the data has been loaded correctly.


<img src="https://github.com/SumitxThokar/California-House-Price-Prediction/blob/main/img/quicklook.jpg">


The **info()** function is an essential tool for gaining a comprehensive overview of your data. It provides insightful information about the total number of rows in your dataset, as well as the data type and non-null count for each attribute, making it incredibly useful for data exploration.

<img src="https://github.com/SumitxThokar/California-House-Price-Prediction/blob/main/img/img3.jpg">

The **describe()** method is used to generate descriptive statistics of a pandas DataFrame. It provides a summary of the central tendency, dispersion, and shape of the distribution of a set of continuous variables, excluding NaN values.

<img src="https://github.com/SumitxThokar/California-House-Price-Prediction/blob/main/img/img5.jpg">

Now that we know how the data looks like. Let's visualize the summary of the data as Histogram using **hist()** method. The hist() method from Matplotlib will plot a histogram for each numerical attribute.
This is how it looks: <br>

<img src="https://github.com/SumitxThokar/California-House-Price-Prediction/blob/main/img/img6.jpg">

### Test-Train Set.
Creating a test and train set is important in machine learning to evaluate the performance of a model on previously unseen data. The idea is to split the data into two parts, one part is used to train the model and the other part is used to test the model. The model is trained on the training set and its performance is evaluated on the test set. This helps in avoiding overfitting, which is when a model performs well on the training set but poorly on new, unseen data. By evaluating the model on the test set, one can get a better estimate of its performance in real-world scenarios.
<br>
We split the dataset using **train_test_split** from sklearn.model_selection.


### Data Visualization.
Now we generate a scatter plot of the "median_house_value" based on "longitude" and "latitude" in the "housing" dataset. The size of each dot on the plot represents the population of the area divided by 100. The color of each dot represents the "median_house_value" and a color bar is added to help with interpretation. The plot size is set to (10,6) and a legend is added to explain the relationship between size and population.


<img src="https://github.com/SumitxThokar/California-House-Price-Prediction/blob/main/img/img7.jpg">


### Correlations

corr() method is used to find correlation between Features. To plot the relations we can use Pandas' scatter_matrix function, which plots every numerical attrubute against every other attribute.


<img src="https://github.com/SumitxThokar/California-House-Price-Prediction/blob/main/img/img7.jpg">


### Data Preparation.
**Data cleaning** involves preprocessing the data to make it suitable for use in machine learning algorithms. One common step in data cleaning is handling text and categorical attributes, which can be done using one-hot encoding. This technique converts categorical variables into numerical variables, allowing them to be used in algorithms that require numerical inputs.
<br>
Another step in data cleaning is **feature scaling**, which involves transforming the features to have a similar scale. There are two common techniques for feature scaling: min-max scaling and standardization. Min-max scaling scales the features to a specified range, while standardization transforms the features to have a mean of 0 and a standard deviation of 1.
<br>
**Transformation pipelines** are a convenient way to apply a sequence of transformations to the data in a consistent manner. The transformations can include data cleaning steps such as one-hot encoding, feature scaling, and others. The pipelines ensure that the same transformations are applied to both the training and test sets, making it easier to build and evaluate models.

### Train and Evaluate on the training set.
In the first step, the team used linear regression to train the model and predict using the test data. The accuracy of the model was measured and it was found that the model was underfitting.

Next, they tried using DecisionTreeRegressor() which resulted in worse performance compared to linear regression. Finally, they used RandomForestRegressor() and got much better results. Random Forest appeared to be a promising method, but there was a high risk of overfitting, so the team decided to regularize it.

To avoid overfitting and improve the performance of the model, the team used GridSearchCV to fine-tune their model. This allowed them to optimize the parameters and improve the accuracy of the model.
