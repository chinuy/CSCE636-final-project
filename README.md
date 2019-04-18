![](figres/predict_5000.png)

Python Jupyter Notebook for 

The web traffic time series forecasting is a competition on [Kaggle](https://www.kaggle.com/c/web-traffic-time-series-forecasting).

Youtube link: https://youtu.be/G3Aeayh6--w

## Achievement summary

The project started using a python notebook from 2nd place author as the boilplate.
It's a simplified version of the author's solution. And the author also adopt xgboost to learn and produce final result.
My work is purely using deep neural network to produce the final result.

The SMAPE score of the cross-validation of the orignal notebook was 0.439, while after my work, the SMAPE score is 0.394.
The final submission result in Kaggle was 0.3820, which was 15th place out of 375 teams.

My work has 3 parts:
1. New neural network architecture
2. Using mean-absolute-error as loss function as SMAPE is not differciable when error > 0
3. Feature engineering

### Dataset
The training data has over 145k wikipedia pages associated with traffic of 793 days.
The prediction objective is given an user-interested page, predict the next 63 days traffic.
- Training data from 2015-07-01 to 2017-09-10
- Testing data from 2017-09-13 to 2017-11-10

The dataset contains traffic history of:
- 7 languages
- 3 access (all, desktop, mobile-web)
- 2 types (all-agents, spider)

### How to run
1. Clone the repo
2. Download the training/submission dataset from the [Kaggle compeition page](https://www.kaggle.com/c/web-traffic-time-series-forecasting/data) and unzip it to **data** folder
3. Run the **cross_validation.ipynb** to get the cross-validation result and the train the model
4. Run the **submission.ipynb** to load the trained model and predict the next 63 days result. The predictioin result will be placed at **submission** folder.

## Execution of GUI

Simply execute the python program

```python3 gui.py```


## The GUI functionality
Following is a screenshot of my GUI.
![](screen1.png)

The GUI has two sections.
- Top section is display section, which shows the network traffic of the user-interested page, including both the historical and predicted traffic.
- The bottome section is input section, which allows users to input the traffic profiles, and predict the future traffic using my pre-trained model, and finally select interested page to display.

A simple step-by-step flow is:
1. An user open a file (.csv) as input
2. The pre-trained model generates prediction result based on the input, and display page names in the selection area (radio button).
3. The user can see the historical traffic in blue, the real traffic result in green and prediction result in red.
