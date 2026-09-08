import csv
import json
import math
import urllib.request

from collections import defaultdict
from datetime import datetime, timedelta
from io import StringIO


# ============================================================
# CONFIG
# ============================================================

START_YEAR = 2005
END_YEAR = 2026

K_NEIGHBORS = 15

MODEL_FILE = "college_station_weather_model.json"
PREDICTIONS_FILE = "weather_predictions.csv"


# ============================================================
# DOWNLOAD WEATHER DATA
# ============================================================

def download_weather_data():
    print("\nDownloading College Station weather data...")

    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        "station=CLL"
        "&data=tmpf"
        "&data=dwpf"
        "&data=relh"
        "&data=drct"
        "&data=sknt"
        "&data=p01i"
        "&data=alti"
        f"&year1={START_YEAR}"
        "&month1=1"
        "&day1=1"
        f"&year2={END_YEAR}"
        "&month2=9"
        "&day2=8"
        "&tz=America/Chicago"
        "&format=onlycomma"
        "&latlon=no"
        "&elev=no"
        "&missing=empty"
        "&trace=0.0001"
        "&direct=no"
        "&report_type=3"
    )

    with urllib.request.urlopen(url, timeout=120) as response:
        text = response.read().decode("utf-8")

    rows = list(csv.DictReader(StringIO(text)))

    print(f"Downloaded {len(rows):,} observations.")

    return rows


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def mean(values):
    if not values:
        return None

    return sum(values) / len(values)


# ============================================================
# CLEAN AND CONVERT TO DAILY DATA
# ============================================================

def create_daily_data(rows):
    print("\nCleaning and converting observations to daily data...")

    grouped = defaultdict(
        lambda: {
            "temps": [],
            "dewpoints": [],
            "humidity": [],
            "wind": [],
            "pressure": [],
            "precipitation": []
        }
    )

    for row in rows:

        try:
            timestamp = datetime.strptime(
                row["valid"],
                "%Y-%m-%d %H:%M"
            )
        except (ValueError, KeyError):
            continue

        date = timestamp.date()

        temp = safe_float(row.get("tmpf"))
        dew = safe_float(row.get("dwpf"))
        humidity = safe_float(row.get("relh"))
        wind = safe_float(row.get("sknt"))
        pressure = safe_float(row.get("alti"))
        precip = safe_float(row.get("p01i"))

        # Remove obviously invalid readings
        if temp is not None and -20 <= temp <= 130:
            grouped[date]["temps"].append(temp)

        if dew is not None and -40 <= dew <= 100:
            grouped[date]["dewpoints"].append(dew)

        if humidity is not None and 0 <= humidity <= 100:
            grouped[date]["humidity"].append(humidity)

        if wind is not None and 0 <= wind <= 100:
            grouped[date]["wind"].append(wind)

        if pressure is not None and 20 <= pressure <= 35:
            grouped[date]["pressure"].append(pressure)

        if precip is not None and precip >= 0:
            grouped[date]["precipitation"].append(precip)

    daily = {}

    for date, data in grouped.items():

        # Require at least 12 temperature observations
        if len(data["temps"]) < 12:
            continue

        if (
            not data["dewpoints"]
            or not data["humidity"]
            or not data["wind"]
            or not data["pressure"]
        ):
            continue

        temps = data["temps"]

        daily[date] = {
            "date": date,

            "max_temp": max(temps),
            "min_temp": min(temps),
            "avg_temp": mean(temps),

            "avg_dewpoint": mean(data["dewpoints"]),
            "max_dewpoint": max(data["dewpoints"]),

            "avg_humidity": mean(data["humidity"]),

            "avg_wind": mean(data["wind"]),
            "max_wind": max(data["wind"]),

            "avg_pressure": mean(data["pressure"]),

            "precipitation": (
                sum(data["precipitation"])
                if data["precipitation"]
                else 0.0
            )
        }

    print(f"Usable daily observations: {len(daily):,}")

    return daily


# ============================================================
# FEATURE ENGINEERING
# ============================================================

FEATURE_NAMES = [
    "max_temp",
    "min_temp",
    "avg_temp",
    "avg_dewpoint",
    "max_dewpoint",
    "avg_humidity",
    "avg_wind",
    "max_wind",
    "avg_pressure",
    "precipitation",

    "max_temp_lag1",
    "max_temp_lag2",
    "max_temp_lag3",
    "max_temp_lag5",
    "max_temp_lag7",

    "temp_3day_avg",
    "temp_7day_avg",

    "dewpoint_3day_avg",
    "humidity_3day_avg",
    "pressure_3day_avg",

    "temp_change_1d",
    "pressure_change_1d",
    "dewpoint_change_1d",

    "day_sin",
    "day_cos"
]


def create_feature_row(date, daily, require_target=True):

    current = daily.get(date)

    if current is None:
        return None

    # Need previous 7 calendar days
    previous_days = []

    for lag in range(1, 8):
        past_date = date - timedelta(days=lag)

        if past_date not in daily:
            return None

        previous_days.append(
            daily[past_date]
        )

    tomorrow = daily.get(
        date + timedelta(days=1)
    )

    if require_target and tomorrow is None:
        return None

    day_of_year = date.timetuple().tm_yday

    day_sin = math.sin(
        2 * math.pi * day_of_year / 365.25
    )

    day_cos = math.cos(
        2 * math.pi * day_of_year / 365.25
    )

    max_temp_3day = mean([
        current["max_temp"],
        previous_days[0]["max_temp"],
        previous_days[1]["max_temp"]
    ])

    max_temp_7day = mean([
        current["max_temp"],
        previous_days[0]["max_temp"],
        previous_days[1]["max_temp"],
        previous_days[2]["max_temp"],
        previous_days[3]["max_temp"],
        previous_days[4]["max_temp"],
        previous_days[5]["max_temp"]
    ])

    dewpoint_3day = mean([
        current["avg_dewpoint"],
        previous_days[0]["avg_dewpoint"],
        previous_days[1]["avg_dewpoint"]
    ])

    humidity_3day = mean([
        current["avg_humidity"],
        previous_days[0]["avg_humidity"],
        previous_days[1]["avg_humidity"]
    ])

    pressure_3day = mean([
        current["avg_pressure"],
        previous_days[0]["avg_pressure"],
        previous_days[1]["avg_pressure"]
    ])

    features = [
        current["max_temp"],
        current["min_temp"],
        current["avg_temp"],
        current["avg_dewpoint"],
        current["max_dewpoint"],
        current["avg_humidity"],
        current["avg_wind"],
        current["max_wind"],
        current["avg_pressure"],
        current["precipitation"],

        previous_days[0]["max_temp"],
        previous_days[1]["max_temp"],
        previous_days[2]["max_temp"],
        previous_days[4]["max_temp"],
        previous_days[6]["max_temp"],

        max_temp_3day,
        max_temp_7day,

        dewpoint_3day,
        humidity_3day,
        pressure_3day,

        (
            current["max_temp"]
            - previous_days[0]["max_temp"]
        ),

        (
            current["avg_pressure"]
            - previous_days[0]["avg_pressure"]
        ),

        (
            current["avg_dewpoint"]
            - previous_days[0]["avg_dewpoint"]
        ),

        day_sin,
        day_cos
    ]

    return {
        "date": date,
        "features": features,
        "target": (
            tomorrow["max_temp"]
            if tomorrow
            else None
        ),
        "today_max": current["max_temp"]
    }


def create_ml_dataset(daily):
    print("\nEngineering ML features...")

    dataset = []

    for date in sorted(daily.keys()):

        row = create_feature_row(
            date,
            daily,
            require_target=True
        )

        if row is not None:
            dataset.append(row)

    print(f"Final ML samples: {len(dataset):,}")

    return dataset


# ============================================================
# DATA SPLIT
# ============================================================

def split_dataset(dataset):

    train = []
    validation = []
    test = []

    for row in dataset:

        date = row["date"]

        if date < datetime(2023, 1, 1).date():
            train.append(row)

        elif date < datetime(2025, 1, 1).date():
            validation.append(row)

        else:
            test.append(row)

    print("\nDataset Split")
    print("-" * 40)

    print(f"Training:   {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Testing:    {len(test):,}")

    return train, validation, test


# ============================================================
# STANDARDIZATION
# ============================================================

def calculate_scaler(train):

    feature_count = len(
        train[0]["features"]
    )

    means = []
    stds = []

    for feature_index in range(feature_count):

        values = [
            row["features"][feature_index]
            for row in train
        ]

        feature_mean = mean(values)

        variance = mean([
            (value - feature_mean) ** 2
            for value in values
        ])

        std = math.sqrt(variance)

        if std == 0:
            std = 1.0

        means.append(feature_mean)
        stds.append(std)

    return means, stds


def standardize(features, means, stds):

    return [
        (value - means[i]) / stds[i]
        for i, value in enumerate(features)
    ]


# ============================================================
# K-NEAREST NEIGHBORS MODEL
# ============================================================

def distance(a, b):

    total = 0.0

    for x, y in zip(a, b):
        total += (x - y) ** 2

    return math.sqrt(total)


def knn_predict(
    input_features,
    train_features,
    train_targets,
    k
):

    distances = []

    for features, target in zip(
        train_features,
        train_targets
    ):

        d = distance(
            input_features,
            features
        )

        distances.append(
            (d, target)
        )

    distances.sort(
        key=lambda x: x[0]
    )

    nearest = distances[:k]

    # Distance-weighted prediction
    numerator = 0.0
    denominator = 0.0

    for d, target in nearest:

        weight = 1 / (d + 0.000001)

        numerator += weight * target
        denominator += weight

    return numerator / denominator


# ============================================================
# EVALUATION
# ============================================================

def calculate_metrics(actual, predicted):

    errors = [
        actual[i] - predicted[i]
        for i in range(len(actual))
    ]

    absolute_errors = [
        abs(error)
        for error in errors
    ]

    squared_errors = [
        error ** 2
        for error in errors
    ]

    mae = mean(absolute_errors)

    rmse = math.sqrt(
        mean(squared_errors)
    )

    actual_mean = mean(actual)

    ss_total = sum(
        (value - actual_mean) ** 2
        for value in actual
    )

    ss_residual = sum(
        (actual[i] - predicted[i]) ** 2
        for i in range(len(actual))
    )

    if ss_total == 0:
        r2 = 0
    else:
        r2 = 1 - (
            ss_residual / ss_total
        )

    return mae, rmse, r2


def print_metrics(name, actual, predicted):

    mae, rmse, r2 = calculate_metrics(
        actual,
        predicted
    )

    print(f"\n{name}")
    print("-" * 40)

    print(f"MAE:  {mae:.2f} °F")
    print(f"RMSE: {rmse:.2f} °F")
    print(f"R²:   {r2:.3f}")

    return mae, rmse, r2


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    test,
    predictions
):

    with open(
        PREDICTIONS_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "date",
            "today_high",
            "actual_tomorrow_high",
            "predicted_tomorrow_high",
            "absolute_error"
        ])

        for row, prediction in zip(
            test,
            predictions
        ):

            error = abs(
                row["target"]
                - prediction
            )

            writer.writerow([
                row["date"].isoformat(),
                round(row["today_max"], 2),
                round(row["target"], 2),
                round(prediction, 2),
                round(error, 2)
            ])

    print(
        f"\nPredictions saved to "
        f"{PREDICTIONS_FILE}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    train_features,
    train_targets,
    means,
    stds
):

    model_data = {
        "model_type":
            "K-Nearest Neighbors Regression",

        "k":
            K_NEIGHBORS,

        "feature_names":
            FEATURE_NAMES,

        "means":
            means,

        "stds":
            stds,

        "train_features":
            train_features,

        "train_targets":
            train_targets
    }

    with open(
        MODEL_FILE,
        "w"
    ) as file:

        json.dump(
            model_data,
            file
        )

    print(
        f"Model saved as "
        f"{MODEL_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("COLLEGE STATION WEATHER ML MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------

    rows = download_weather_data()

    # --------------------------------------------------------
    # DAILY DATA
    # --------------------------------------------------------

    daily = create_daily_data(rows)

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    dataset = create_ml_dataset(daily)

    if len(dataset) == 0:
        print(
            "No usable ML samples were generated."
        )
        return

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train, validation, test = (
        split_dataset(dataset)
    )

    if not train or not test:
        print(
            "Not enough data for training/testing."
        )
        return

    # --------------------------------------------------------
    # STANDARDIZE FEATURES
    # --------------------------------------------------------

    means, stds = calculate_scaler(
        train
    )

    train_features = [
        standardize(
            row["features"],
            means,
            stds
        )
        for row in train
    ]

    train_targets = [
        row["target"]
        for row in train
    ]

    test_features = [
        standardize(
            row["features"],
            means,
            stds
        )
        for row in test
    ]

    test_targets = [
        row["target"]
        for row in test
    ]

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_predictions = [
        row["today_max"]
        for row in test
    ]

    baseline_metrics = print_metrics(
        "Persistence Baseline",
        test_targets,
        baseline_predictions
    )

    # --------------------------------------------------------
    # KNN MODEL
    # --------------------------------------------------------

    print(
        "\nRunning K-Nearest Neighbors predictions..."
    )

    knn_predictions = []

    total = len(test_features)

    for i, features in enumerate(
        test_features
    ):

        prediction = knn_predict(
            features,
            train_features,
            train_targets,
            K_NEIGHBORS
        )

        knn_predictions.append(
            prediction
        )

        if (
            (i + 1) % 100 == 0
            or i + 1 == total
        ):
            print(
                f"Predicted "
                f"{i + 1}/{total} test days"
            )

    knn_metrics = print_metrics(
        f"KNN Regression (k={K_NEIGHBORS})",
        test_targets,
        knn_predictions
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    print("\nMODEL COMPARISON")
    print("=" * 60)

    print(
        f"{'Model':<30}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'R²':>10}"
    )

    print("-" * 60)

    print(
        f"{'Persistence Baseline':<30}"
        f"{baseline_metrics[0]:>10.2f}"
        f"{baseline_metrics[1]:>10.2f}"
        f"{baseline_metrics[2]:>10.3f}"
    )

    print(
        f"{'KNN Regression':<30}"
        f"{knn_metrics[0]:>10.2f}"
        f"{knn_metrics[1]:>10.2f}"
        f"{knn_metrics[2]:>10.3f}"
    )

    # --------------------------------------------------------
    # WORST PREDICTIONS
    # --------------------------------------------------------

    results = []

    for row, prediction in zip(
        test,
        knn_predictions
    ):

        results.append({
            "date": row["date"],
            "actual": row["target"],
            "prediction": prediction,
            "error": abs(
                row["target"] - prediction
            )
        })

    results.sort(
        key=lambda x: x["error"],
        reverse=True
    )

    print("\nWORST 10 PREDICTIONS")
    print("=" * 70)

    print(
        f"{'Date':<15}"
        f"{'Actual':>12}"
        f"{'Predicted':>15}"
        f"{'Error':>12}"
    )

    print("-" * 70)

    for row in results[:10]:

        print(
            f"{row['date'].isoformat():<15}"
            f"{row['actual']:>12.1f}"
            f"{row['prediction']:>15.1f}"
            f"{row['error']:>12.1f}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_predictions(
        test,
        knn_predictions
    )

    save_model(
        train_features,
        train_targets,
        means,
        stds
    )

    # --------------------------------------------------------
    # LATEST HISTORICAL PREDICTION
    # --------------------------------------------------------

    latest_date = max(
        daily.keys()
    )

    # Latest day might not have 7 previous valid days,
    # so search backwards.
    latest_feature_row = None

    current_date = latest_date

    for _ in range(30):

        candidate = create_feature_row(
            current_date,
            daily,
            require_target=False
        )

        if candidate is not None:
            latest_feature_row = candidate
            break

        current_date -= timedelta(days=1)

    if latest_feature_row:

        latest_standardized = (
            standardize(
                latest_feature_row["features"],
                means,
                stds
            )
        )

        latest_prediction = (
            knn_predict(
                latest_standardized,
                train_features,
                train_targets,
                K_NEIGHBORS
            )
        )

        print("\nLATEST DATASET PREDICTION")
        print("=" * 60)

        print(
            "Weather input date:",
            latest_feature_row[
                "date"
            ].isoformat()
        )

        print(
            "Today's observed high:",
            f"{latest_feature_row['today_max']:.1f} °F"
        )

        print(
            "Predicted next-day high:",
            f"{latest_prediction:.1f} °F"
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        "\nNOTE: This currently evaluates a historical "
        "machine-learning model."
    )

    print(
        "The next step will be connecting it to today's "
        "live weather observations."
    )


if __name__ == "__main__":
    main()