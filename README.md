# College Station Weather Prediction Mini Project

A machine-learning project that uses historical weather observations from College Station, Texas to predict the **next day's maximum temperature**.

The project currently uses a custom implementation of **K-Nearest Neighbors (KNN) Regression** and compares its predictions against a simple persistence baseline.

The long-term goal is to combine the historical ML model with live College Station weather data and provide a real next-day weather prediction through a simple application.

---

## Project Goal

The current model answers the following question:

> Given today's weather and the recent seven-day weather pattern in College Station, what will tomorrow's maximum temperature be?

For example:

```text
Recent College Station Weather
            ↓
     Feature Engineering
            ↓
       KNN Regression
            ↓
Predicted Tomorrow High: 93.4°F
```

This is a **regression problem** because the model predicts a continuous numerical value: temperature in degrees Fahrenheit.

---

## Project Architecture

The project currently contains two main components.

```text
                HISTORICAL WEATHER
             Iowa State Mesonet ASOS
                       |
                       v
                weather_model.py
                       |
              Data Cleaning
                       |
              Daily Aggregation
                       |
              Feature Engineering
                       |
              KNN Regression
                       |
             Model Evaluation
                 /          \
                v            v
 weather_predictions.csv   college_station_weather_model.json


                   LIVE WEATHER
                    Open-Meteo
                        |
                        v
                    weather.py
                        |
                        v
             Current College Station
                Weather Conditions
```

The historical model and live-weather script are currently separate.

A future step will connect live/recent observations to the saved ML model so the project can generate a real next-day prediction.

---

# Project Files

## `weather_model.py`

This is the main machine-learning pipeline.

It performs the following steps:

1. Downloads historical College Station weather observations
2. Cleans invalid or missing measurements
3. Converts multiple observations per day into daily statistics
4. Creates machine-learning features
5. Creates the prediction target
6. Splits the data chronologically
7. Standardizes the features
8. Runs K-Nearest Neighbors regression
9. Compares the model against a baseline
10. Calculates evaluation metrics
11. Saves predictions
12. Saves the trained KNN data

---

## `weather.py`

This script retrieves the current weather for College Station using the Open-Meteo API.

It first converts:

```text
College Station
```

into latitude and longitude coordinates.

It then retrieves current conditions including:

* Temperature
* Relative humidity
* Apparent temperature
* Precipitation
* Wind speed

Example output:

```text
{'name': 'College Station',
 'latitude': 30.62798,
 'longitude': -96.33441}

{'temperature_2m': 91.9,
 'relative_humidity_2m': 53,
 'apparent_temperature': 104.1,
 'precipitation': 0.0,
 'wind_speed_10m': 1.1}
```

This script currently provides live weather information but does not yet send those values into the ML model.

---

## `college_station_weather_model.json`

This file stores the trained KNN model information.

Because KNN is an instance-based learning algorithm, the model must retain the historical training examples used to find similar weather situations.

The saved file contains:

* Model type
* Number of neighbors (`k`)
* Feature names
* Training-set feature means
* Training-set feature standard deviations
* Standardized training observations
* Training target temperatures

Current model:

```text
K-Nearest Neighbors Regression
k = 15
```

---

## `weather_predictions.csv`

This file contains predictions made on the historical test dataset.

Columns include:

```text
date
today_high
actual_tomorrow_high
predicted_tomorrow_high
absolute_error
```

Example:

```text
date        today   actual tomorrow   predicted tomorrow   error
2025-01-04  70°F    74°F              73.94°F              0.06°F
```

This file allows us to inspect both good and bad predictions and understand where the model performs well or poorly.

---

# Historical Weather Data

Historical weather observations are downloaded for the College Station area using the Iowa State Mesonet ASOS archive.

The station used by the project is:

```text
Station: CLL
College Station / Easterwood Field
```

The current historical range is:

```text
2005 through 2026
```

The downloaded observations include:

| Variable | Description          |
| -------- | -------------------- |
| `tmpf`   | Air temperature      |
| `dwpf`   | Dew point            |
| `relh`   | Relative humidity    |
| `sknt`   | Wind speed           |
| `p01i`   | Precipitation        |
| `alti`   | Atmospheric pressure |
| `drct`   | Wind direction       |

Wind direction is downloaded but is not currently used as an ML feature.

---

# Data Cleaning

Weather stations produce many observations throughout each day.

The program groups these measurements by calendar day and filters obviously invalid readings.

Examples of accepted ranges include:

```text
Temperature:    -20°F to 130°F
Humidity:         0% to 100%
Wind:             0 to 100
Pressure:        20 to 35 inHg
Precipitation:   >= 0
```

A day must contain at least 12 valid temperature observations before it is included.

Days must also contain usable:

* Dew point
* Humidity
* Wind
* Pressure

This helps prevent incomplete days from becoming misleading training examples.

---

# Daily Weather Representation

After cleaning the raw observations, the program summarizes each day.

Each daily record contains:

```text
Maximum temperature
Minimum temperature
Average temperature

Average dew point
Maximum dew point

Average humidity

Average wind
Maximum wind

Average pressure

Total precipitation
```

These daily values become the foundation for the machine-learning features.

---

# Feature Engineering

The model currently uses **25 features**.

### Current-day weather

```text
Maximum temperature
Minimum temperature
Average temperature
Average dew point
Maximum dew point
Average humidity
Average wind
Maximum wind
Average pressure
Precipitation
```

### Temperature lag features

The model also considers the maximum temperature from:

```text
1 day ago
2 days ago
3 days ago
5 days ago
7 days ago
```

These features help describe recent temperature trends.

### Rolling averages

The model calculates:

```text
3-day average maximum temperature
7-day average maximum temperature

3-day average dew point
3-day average humidity
3-day average pressure
```

### Weather changes

The model also measures:

```text
1-day temperature change
1-day pressure change
1-day dew-point change
```

These features can help identify changing weather patterns.

### Seasonal features

Weather depends strongly on the time of year.

Rather than using the day number directly, the model converts the day of the year into two cyclical features:

```text
day_sin
day_cos
```

using:

```text
sin(2π × day_of_year / 365.25)
cos(2π × day_of_year / 365.25)
```

This allows the model to understand that December and January are close together in the annual weather cycle.

---

# Prediction Target

For every usable day, the target is:

```text
Tomorrow's maximum temperature
```

Example:

```text
Features from July 10
        ↓
Target = July 11 maximum temperature
```

This means the model learns relationships such as:

```text
Current temperature
+
Recent temperature trend
+
Humidity
+
Dew point
+
Pressure
+
Wind
+
Precipitation
+
Season
        ↓
Tomorrow's maximum temperature
```

---

# Train / Validation / Test Split

Because weather is time-dependent, the project does **not** randomly mix historical observations.

Instead, the data is split chronologically.

```text
Before January 1, 2023
        ↓
Training Set

2023 – 2024
        ↓
Validation Set

2025 onward
        ↓
Test Set
```

This creates a more realistic ML evaluation:

```text
Learn from the past
        ↓
Predict the future
```

The validation set is created but is not yet used for hyperparameter tuning. Adding validation-based model selection is one of the planned improvements.

---

# Feature Standardization

KNN determines similarity by measuring distance between feature vectors.

Because weather variables have different numerical scales, the features are standardized before calculating distances.

For every feature:

```text
standardized value =
(value - training mean) / training standard deviation
```

For example:

```text
Historical mean high = 80°F
Standard deviation    = 15°F
Current high          = 95°F

(95 - 80) / 15 = 1.0
```

The model therefore sees the temperature as:

```text
1 standard deviation above average
```

rather than simply seeing `95`.

This prevents variables with large numerical values from unfairly dominating the KNN distance calculation.

---

# How the KNN Model Works

KNN stands for:

```text
K-Nearest Neighbors
```

The basic idea is:

> Find historical weather situations that look most similar to today's weather and see what happened the following day.

The project currently uses:

```text
k = 15
```

so the model finds the **15 most similar historical days**.

---

## Step 1: Represent today as features

A day becomes a vector containing the 25 standardized features.

Conceptually:

```text
[
 temperature,
 humidity,
 dew point,
 wind,
 pressure,
 precipitation,
 recent temperatures,
 rolling averages,
 weather changes,
 seasonal values
]
```

---

## Step 2: Compare today against historical days

The model calculates the Euclidean distance between today's feature vector and every training example.

The distance is:

```text
sqrt(
    (feature1 difference)^2 +
    (feature2 difference)^2 +
    ...
    (feature25 difference)^2
)
```

Smaller distance means:

```text
More similar historical weather
```

Larger distance means:

```text
Less similar historical weather
```

---

## Step 3: Select the 15 nearest days

After calculating all distances, the model sorts them and keeps the nearest 15 historical examples.

Example:

| Historical Day | Distance | Following Day High |
| -------------- | -------: | -----------------: |
| Day A          |     0.31 |               94°F |
| Day B          |     0.40 |               92°F |
| Day C          |     0.53 |               95°F |
| ...            |      ... |                ... |
| Day O          |     1.12 |               90°F |

---

## Step 4: Distance-weighted prediction

The 15 neighbors are not treated equally.

Closer neighbors receive greater weight.

The project uses:

```text
weight = 1 / distance
```

with a very small value added to prevent division by zero.

The final prediction is approximately:

```text
weighted sum of neighbor temperatures
-------------------------------------
sum of neighbor weights
```

For example:

```text
Most similar historical weather patterns
                ↓
Following-day highs:
94, 92, 95, 93, 96, ...
                ↓
Distance-weighted average
                ↓
Prediction = 94.1°F
```

---

# Persistence Baseline

The project also evaluates a very simple prediction:

> Tomorrow's high will equal today's high.

Example:

```text
Today's high = 92°F

Baseline prediction:
Tomorrow = 92°F
```

This is called a **persistence baseline**.

The KNN model should ideally outperform this simple approach.

If a complicated ML model cannot beat the baseline, then it may not be adding useful predictive value.

---

# Model Evaluation

The model currently uses three regression metrics.

## Mean Absolute Error — MAE

MAE measures the average absolute difference between the model prediction and the real temperature.

Example:

```text
Actual:     95°F
Prediction: 92°F
Error:       3°F
```

Lower MAE is better.

---

## Root Mean Squared Error — RMSE

RMSE also measures prediction error but penalizes very large mistakes more strongly.

This makes it useful for identifying models that occasionally make severe prediction errors.

Lower RMSE is better.

---

## R² Score

R² measures how much of the variation in tomorrow's temperature is explained by the model.

In general:

```text
Higher R² = better fit
```

The project compares these metrics for:

```text
Persistence Baseline
vs.
KNN Regression
```

---

# Worst Predictions

The program also identifies the ten test days with the largest absolute prediction errors.

This is useful because average metrics alone do not explain **when the model fails**.

Future analysis can investigate whether large errors are associated with weather events such as:

* Rapid temperature changes
* Cold fronts
* Strong pressure changes
* Severe storms
* Unusual seasonal events

These causes have not yet been formally analyzed in the current project.

---

# Current Live Weather System

`weather.py` uses Open-Meteo to retrieve live College Station conditions.

Current inputs include:

```text
Current temperature
Relative humidity
Apparent temperature
Precipitation
Wind speed
```

The live script currently works independently from the historical KNN model.

The next major development step is to retrieve enough recent weather history to build the same 25 features used during model training.

---

# Current Project Status

| Component                    | Status |
| ---------------------------- | ------ |
| Historical weather download  | ✅      |
| Data cleaning                | ✅      |
| Daily aggregation            | ✅      |
| Feature engineering          | ✅      |
| Seasonal features            | ✅      |
| Lag features                 | ✅      |
| Rolling features             | ✅      |
| Chronological data split     | ✅      |
| Feature standardization      | ✅      |
| KNN regression               | ✅      |
| Persistence baseline         | ✅      |
| MAE / RMSE / R² evaluation   | ✅      |
| Prediction CSV output        | ✅      |
| Saved KNN model              | ✅      |
| Live College Station weather | ✅      |
| Validation-based tuning      | 🚧     |
| Live ML prediction           | 🚧     |
| Multiple-model comparison    | 🚧     |
| Graphical interface          | 🚧     |

---

# Setup

## Requirements

Python 3.12 is recommended for development.

Create a virtual environment:

```bash
python3.12 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the currently required external dependency:

```bash
pip install requests
```

The current KNN implementation is written manually using Python's standard library, so scikit-learn is not required to run the existing model.

Future versions of this project may use packages such as:

```bash
pip install numpy pandas scikit-learn matplotlib joblib jupyter
```

for model comparison, visualization, and experimentation.

---

# Running the Project

## Test live College Station weather

Run:

```bash
python weather.py
```

Expected output includes:

```text
College Station coordinates
Current temperature
Current humidity
Apparent temperature
Current precipitation
Current wind
```

---

## Train and evaluate the historical model

Run:

```bash
python weather_model.py
```

The script will:

```text
Download historical weather
        ↓
Clean observations
        ↓
Create daily data
        ↓
Engineer features
        ↓
Split data
        ↓
Standardize features
        ↓
Evaluate persistence baseline
        ↓
Run KNN predictions
        ↓
Calculate MAE / RMSE / R²
        ↓
Display worst predictions
        ↓
Save prediction results
        ↓
Save trained model
```

---

# Generated Files

Running `weather_model.py` generates:

```text
weather_predictions.csv
college_station_weather_model.json
```

### `weather_predictions.csv`

Contains the test-set predictions and errors.

### `college_station_weather_model.json`

Contains the information required to reproduce the current KNN model, including stored training examples and standardization values.

---

# macOS SSL Note

Some macOS Python installations may encounter an SSL certificate error when `weather_model.py` attempts to download historical Mesonet data.

Example:

```text
CERTIFICATE_VERIFY_FAILED
```

For Python installed from Python.org, certificate installation can usually be repaired with:

```bash
open "/Applications/Python 3.12/Install Certificates.command"
```

A future project update may replace the current `urllib` historical download code with `requests` for consistency with `weather.py`.

---

# Planned Improvements

The next phases of the project are:

1. **Tune KNN using the validation dataset**

   Compare values such as:

   ```text
   k = 3
   k = 5
   k = 7
   k = 10
   k = 15
   k = 20
   k = 30
   ```

2. **Compare multiple regression models**

   Potential models include:

   ```text
   Linear Regression
   KNN Regression
   Decision Tree Regression
   Random Forest Regression
   Gradient Boosting
   ```

3. **Connect recent/live observations to the saved model**

   Retrieve the previous seven days of College Station weather and construct the same feature vector used during training.

4. **Generate a true current next-day prediction**

   Example:

   ```text
   College Station
   September 8

   Predicted September 9 High:
   92.7°F
   ```

5. **Predict additional weather variables**

   Possible future targets:

   ```text
   Tomorrow's low temperature
   Rain probability
   Precipitation amount
   Weather condition
   ```

6. **Build a graphical interface**

   The final application could display:

   ```text
   COLLEGE STATION WEATHER PREDICTOR

   Current Temperature: 92°F
   Current Humidity:    53%

   Tomorrow's Predicted High:
   94°F

   Model:
   KNN Regression
   ```

---

# Project Summary

This project demonstrates a complete introductory machine-learning workflow using real weather data:

```text
Real-world data collection
        ↓
Data cleaning
        ↓
Feature engineering
        ↓
Time-aware dataset splitting
        ↓
Feature scaling
        ↓
Machine-learning regression
        ↓
Baseline comparison
        ↓
Model evaluation
        ↓
Saved predictions and model
```

The current model uses historical weather similarity to predict the following day's maximum temperature in College Station.
