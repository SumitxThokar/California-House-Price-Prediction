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
