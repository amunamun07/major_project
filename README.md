# Major Project 4: Insights and Prediction of Growing Suitable Crop

Agriculture makes the people less dependent on foreign countries as it provides food and also provides income to the farmers and revenue to the government. 
This project is initiated to help these farmers with suggestions about the best crop that can be grown in their land.

The project objectives are:
- From the given dataset, we need to analyze the different conditions where crops are grown.
- We need to group the list of crops based on the similar condition that they grow in.
- We also need to predict which crop can be grown with which given conditions.
- Creating a monitoring system that can monitor the often updating data to create a better model.


💫 **Version 1.0**
## 📖 Documentation

| Documentation              |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| 📚 **[Project Details]**   | Document on Objective, System Workflow, Outputs and limitations                             | |
| 📚 **[Project Slides]**   | Final Workflow and Demo Presentation 
[Project Details]: https://docs.google.com/document/d/16azcveBOR0jl3mDAY437wPh1UlI_ezPo6L5IIcT9SMM/edit?usp=sharing
[Project Slides]: https://docs.google.com/presentation/d/1OHigyURp7DdXYgNisa0RaMyR0baNTVJPtHdw88X0MGQ/edit?usp=sharing

## Features

- Data Understanding
- Exploratory Data Analysis on the Data
- Derivation of High Level Statistics
- Visualization of various plots
- K Means Clustering
- Principle Component Analysis and TSNE Analysis
- Flask API
- Streamlit
- Html / CSS
- Predictive Modeling using Logistic Regression
- Evaluation using Various Metrics


## 📦 Setting up the project
```bash
git clone git@github.com:amunamun07/major_project.git
```

You need to set up the environment.

- **Operating system**: macOS / OS X · Linux · Windows (Cygwin, MinGW, Visual
  Studio)
- **Python version**: Python 3.9 (only 64 bit)
- **Package managers**: [pipenv]

[pipenv]: https://pypi.org/project/pipenv/

### pipenv

Using pip, we can install the pipenv package manager, make sure that
your `pip`, `setuptools` and `wheel` are up to date. If you already have 
pipenv installed, you can skip the first command.

```bash
pip install pipenv
```
To create a pipenv environment in a folder.
```bash
pipenv shell
```
To get the project requirements.
```bash
pipenv sync
```

### Getting the dataset

To download the dataset into the folder, use:
```bash
wget https://drive.google.com/file/d/1WCpTGIJrHudvWEbRe09dDIAYKJo2GiHo/view?usp=sharing crop_data.csv
```

### Getting the model

To download the model for ease, use:
```bash
wget https://drive.google.com/file/d/1q5wvhOCLYTSJxp6R8iRgsAfBnsNay44n/view?usp=sharing model.plk
```
(Alternatively): Copy those links and download them manually.

## ⚒ Run the project

We need to activate both the servers streamlit and flask at once. So one of the way is to run streamlit
in the background and then running the flask app.
Alternatively you can use two subprocesses and manage running both the servers.

```bash
# Run the Streamlit Dashboard in the background
streamlit run dashboard/main.py &

# (Optional) -> run the Streamlit Dashboard only
streamlit run dashboard/main.py

# Run the Flask Server
python app.py
```

## 🚦 Run tests

You can run the test files using the command below

```bash
pytest -v
