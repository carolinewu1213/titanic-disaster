# Titanic Disaster: Survival Prediction

This project predicts whether a Titanic passenger survived, using logistic regression. I built the model twice, once in Python and once in R, and each version runs in its own Docker container.

## What you need

* Git
* Docker Desktop, installed and running

You don't need Python or R on your computer. Everything runs inside Docker.

## 1. Clone the repository

```
git clone https://github.com/carolinewu1213/titanic-disaster.git
cd titanic-disaster
```

Run every command below from this folder.

## 2. Get the data

The data isn't in this repo, so you'll need to download it:

1. Go to https://www.kaggle.com/competitions/titanic/data.
2. Download the files.
3. Put `train.csv` and `test.csv` directly inside `src/data/`.

Check that both files are there:

```
ls src/data/
```

You should see `test.csv` and `train.csv`. The data folder is listed in `.gitignore`, so the files won't be committed. Both containers read the data through a mount at run time, so it never gets copied into an image.

## 3. Run the Python model

Make sure Docker Desktop is running, then build the image:

```
docker build -t titanic-python -f src/python/Dockerfile src/python
```

Then run it:

```
docker run --rm -v "$(pwd)/src/data:/app/data:ro" titanic-python
```

The output shows the data shapes, missing-value counts, preprocessing steps, and accuracy results. On my run, the numbers were:

```
Training accuracy: 0.8076
Held-out test accuracy: 0.8045
```

The image uses Python 3.12 with pandas 2.2.2, numpy 1.26.4, and scikit-learn 1.5.2, which are pinned in `src/python/requirements.txt`.

## 4. Run the R model

Build the image:

```
docker build -t titanic-r -f src/r/Dockerfile src/r
```

Then run it:

```
docker run --rm -v "$(pwd)/src/data:/app/data:ro" titanic-r
```

The output looks like the Python run. On my run, the numbers were:

```
Training accuracy: 0.8022
Held-out test accuracy: 0.8146
```

The R image uses `rocker/r-ver:4.6.1`. The `install_packages.R` file installs `readr` and `dplyr`.

## How the model works

Both versions use the same features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.

* Missing `Age` and `Fare` values are filled with the median from the training data.
* Missing `Embarked` values are filled with the most common value in the training data.
* The Python version scales the numbers and one-hot encodes the categories before running logistic regression. The R version turns the categories into factors and uses `glm()`.
* Since Kaggle's `test.csv` has no answers, I hold out 20% of `train.csv` to measure accuracy. The random seed is 42.
* After measuring accuracy, each model is refit on all of `train.csv` and predicts survival for the test passengers.

The two versions split the data slightly differently, so their numbers aren't identical.

## Output

Each script writes a `predictions.csv` file with `PassengerId` and `Survived` columns. A `1` means the passenger is predicted to survive, and a `0` means they aren't. The terminal output shows the predicted survival rate and the first few rows.

## Folder structure

```
titanic-disaster/
├── README.md
├── .gitignore
└── src/
    ├── data/                  # Put train.csv and test.csv here
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic_model.R
```

## Troubleshooting

* **`docker: command not found`**: Install Docker Desktop, then reopen Terminal.
* **Cannot connect to the Docker daemon**: Open Docker Desktop and wait until it says it's running.
* **`Could not find train.csv`**: Make sure both CSV files are directly inside `src/data/`, and that you're running the commands from the repo folder.
