#   Tesla Stock Analysis

This project is an interactive application developed in Python for analyzing historical Tesla stock data. The main objective is to provide financial insights through descriptive statistics, statistical tests, and graphical visualizations, as well as to simulate the evolution of an investment based on a user-selected purchase date.

##   🎯 Features

-   **Descriptive Statistics**:
    -   Mean, Median, Mode, Variance, Standard Deviation, and Mean Absolute Deviation of closing prices.
    -   Interactive histogram of prices.

-   **Student's t-test**:
    -   Comparison of the average closing prices with a hypothetical value to determine statistical significance.

-   **Prediction with Linear Regression**:
    -   Linear regression model to predict future stock prices.

-   **Investment Evolution**:
    -   Simulation of investment growth from a purchase date, considering accumulated returns over time.

-   **Total sales volume per year and month**:
    -   Example graph of sales over the year and per month.

-   **Buy and sell indicators**:
    -   Based on historical data, indicates when it was good to buy and when it was better to sell.

-   **Correlation between price and volume**
    -   Correlation graph between these two variables.

##   🛠️ Technologies Used

-   **Programming Language**: Python
-   **Main Libraries**:
    -   [Streamlit](https://streamlit.io/): For creating interactive dashboards.
    -   [Pandas](https://pandas.pydata.org/): Data manipulation and analysis.
    -   [Plotly](https://plotly.com/): Creation of interactive charts.
    -   [NumPy](https://numpy.org/): Mathematical and statistical calculations.
    -   [Scikit-learn](https://scikit-learn.org/): Predictive modeling with linear regression.

##   ⚙️ Project Structure

-   **app.py**: Main application code, including data processing, statistical calculations, and chart generation.
-   **Dataset/**: Directory containing historical Tesla stock data (CSV).
-   **README.md**: This file explaining the project.
-   **requirements.txt**: This file consists of the project dependencies.
