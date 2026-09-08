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

K_VALUES = [3, 5, 7, 10, 15, 20, 30, 40, 50]

MODEL_FILE = "college_station_weather_model.json"
PREDICTIONS_FILE = "weather_predictions.csv"


# ============================================================
# HELPERS
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


def circular_mean_degrees(values):
    if not values:
        return None

    sin_sum = 0.0
    cos_sum = 0.0

    for value in values:
        radians = math.radians(value)

        sin_sum += math.sin(radians)
        cos_sum += math.cos(radians)

    angle = math.degrees(
        math.atan2(
            sin_sum / len(values),
            cos_sum / len(values)
        )
    )

    if angle < 0:
        angle += 360

    return angle


def first_mean(values, count=3):
    if not values:
        return None

    return mean(values[:count])


def last_mean(values, count=3):
    if not values:
        return None

    return mean(values[-count:])


# ============================================================
# DOWNLOAD DATA
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

    rows = list(
        csv.DictReader(
            StringIO(text)
        )
    )

    print(
        f"Downloaded {len(rows):,} observations."
    )

    return rows


# ============================================================
# CREATE DAILY DATA
# ============================================================

def create_daily_data(rows):
    print(
        "\nCleaning and converting observations to daily data..."
    )

    grouped = defaultdict(
        lambda: {
            "temps": [],
            "dewpoints": [],
            "humidity": [],
            "wind": [],
            "direction": [],
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
        direction = safe_float(row.get("drct"))
        pressure = safe_float(row.get("alti"))
        precip = safe_float(row.get("p01i"))

        if temp is not None and -20 <= temp <= 130:
            grouped[date]["temps"].append(temp)

        if dew is not None and -40 <= dew <= 100:
            grouped[date]["dewpoints"].append(dew)

        if humidity is not None and 0 <= humidity <= 100:
            grouped[date]["humidity"].append(humidity)

        if wind is not None and 0 <= wind <= 100:
            grouped[date]["wind"].append(wind)

        if direction is not None and 0 <= direction <= 360:
            grouped[date]["direction"].append(direction)

        if pressure is not None and 20 <= pressure <= 35:
            grouped[date]["pressure"].append(pressure)

        if precip is not None and precip >= 0:
            grouped[date]["precipitation"].append(precip)

    daily = {}

    for date, data in grouped.items():

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
        dewpoints = data["dewpoints"]
        pressures = data["pressure"]

        wind_direction = circular_mean_degrees(
            data["direction"]
        )

        if wind_direction is None:
            wind_direction = 0.0

        temp_intraday_change = (
            last_mean(temps)
            - first_mean(temps)
        )

        pressure_intraday_change = (
            last_mean(pressures)
            - first_mean(pressures)
        )

        dewpoint_intraday_change = (
            last_mean(dewpoints)
            - first_mean(dewpoints)
        )

        daily[date] = {
            "date": date,

            "max_temp": max(temps),
            "min_temp": min(temps),
            "avg_temp": mean(temps),

            "avg_dewpoint": mean(dewpoints),
            "max_dewpoint": max(dewpoints),

            "avg_humidity": mean(data["humidity"]),

            "avg_wind": mean(data["wind"]),
            "max_wind": max(data["wind"]),

            "wind_direction": wind_direction,

            "avg_pressure": mean(pressures),

            "precipitation": (
                sum(data["precipitation"])
                if data["precipitation"]
                else 0.0
            ),

            "temp_intraday_change":
                temp_intraday_change,

            "pressure_intraday_change":
                pressure_intraday_change,

            "dewpoint_intraday_change":
                dewpoint_intraday_change
        }

    print(
        f"Usable daily observations: {len(daily):,}"
    )

    return daily


# ============================================================
# FEATURES
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

    "wind_direction_sin",
    "wind_direction_cos",

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
    "temp_change_2d",

    "pressure_change_1d",
    "pressure_change_2d",

    "dewpoint_change_1d",
    "dewpoint_change_2d",

    "temp_intraday_change",
    "pressure_intraday_change",
    "dewpoint_intraday_change",

    "day_sin",
    "day_cos"
]


def create_feature_row(
    date,
    daily,
    require_target=True
):

    current = daily.get(date)

    if current is None:
        return None

    previous_days = []

    for lag in range(1, 8):
        previous_date = (
            date
            - timedelta(days=lag)
        )

        if previous_date not in daily:
            return None

        previous_days.append(
            daily[previous_date]
        )

    tomorrow = daily.get(
        date + timedelta(days=1)
    )

    if require_target and tomorrow is None:
        return None

    # Seasonal features
    day_of_year = date.timetuple().tm_yday

    day_sin = math.sin(
        2
        * math.pi
        * day_of_year
        / 365.25
    )

    day_cos = math.cos(
        2
        * math.pi
        * day_of_year
        / 365.25
    )

    # Wind direction
    wind_rad = math.radians(
        current["wind_direction"]
    )

    wind_sin = math.sin(wind_rad)
    wind_cos = math.cos(wind_rad)

    # Rolling averages
    temp_3day_avg = mean([
        current["max_temp"],
        previous_days[0]["max_temp"],
        previous_days[1]["max_temp"]
    ])

    temp_7day_avg = mean([
        current["max_temp"],
        previous_days[0]["max_temp"],
        previous_days[1]["max_temp"],
        previous_days[2]["max_temp"],
        previous_days[3]["max_temp"],
        previous_days[4]["max_temp"],
        previous_days[5]["max_temp"]
    ])

    dewpoint_3day_avg = mean([
        current["avg_dewpoint"],
        previous_days[0]["avg_dewpoint"],
        previous_days[1]["avg_dewpoint"]
    ])

    humidity_3day_avg = mean([
        current["avg_humidity"],
        previous_days[0]["avg_humidity"],
        previous_days[1]["avg_humidity"]
    ])

    pressure_3day_avg = mean([
        current["avg_pressure"],
        previous_days[0]["avg_pressure"],
        previous_days[1]["avg_pressure"]
    ])

    # Day-to-day changes
    temp_change_1d = (
        current["max_temp"]
        - previous_days[0]["max_temp"]
    )

    temp_change_2d = (
        current["max_temp"]
        - previous_days[1]["max_temp"]
    )

    pressure_change_1d = (
        current["avg_pressure"]
        - previous_days[0]["avg_pressure"]
    )

    pressure_change_2d = (
        current["avg_pressure"]
        - previous_days[1]["avg_pressure"]
    )

    dewpoint_change_1d = (
        current["avg_dewpoint"]
        - previous_days[0]["avg_dewpoint"]
    )

    dewpoint_change_2d = (
        current["avg_dewpoint"]
        - previous_days[1]["avg_dewpoint"]
    )

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

        wind_sin,
        wind_cos,

        previous_days[0]["max_temp"],
        previous_days[1]["max_temp"],
        previous_days[2]["max_temp"],
        previous_days[4]["max_temp"],
        previous_days[6]["max_temp"],

        temp_3day_avg,
        temp_7day_avg,

        dewpoint_3day_avg,
        humidity_3day_avg,
        pressure_3day_avg,

        temp_change_1d,
        temp_change_2d,

        pressure_change_1d,
        pressure_change_2d,

        dewpoint_change_1d,
        dewpoint_change_2d,

        current["temp_intraday_change"],
        current["pressure_intraday_change"],
        current["dewpoint_intraday_change"],

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
    print(
        "\nEngineering ML features..."
    )

    dataset = []

    for date in sorted(
        daily.keys()
    ):
        row = create_feature_row(
            date,
            daily,
            require_target=True
        )

        if row is not None:
            dataset.append(row)

    print(
        f"Final ML samples: {len(dataset):,}"
    )

    return dataset


# ============================================================
# SPLIT
# ============================================================

def split_dataset(dataset):
    train = []
    validation = []
    test = []

    for row in dataset:

        if row["date"] < datetime(
            2023, 1, 1
        ).date():
            train.append(row)

        elif row["date"] < datetime(
            2025, 1, 1
        ).date():
            validation.append(row)

        else:
            test.append(row)

    print("\nDataset Split")
    print("-" * 40)

    print(
        f"Training:   {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Testing:    {len(test):,}"
    )

    return train, validation, test


# ============================================================
# STANDARDIZATION
# ============================================================

def calculate_scaler(dataset):
    feature_count = len(
        dataset[0]["features"]
    )

    means = []
    stds = []

    for feature_index in range(
        feature_count
    ):

        values = [
            row["features"][
                feature_index
            ]
            for row in dataset
        ]

        feature_mean = mean(values)

        variance = mean([
            (
                value
                - feature_mean
            ) ** 2
            for value in values
        ])

        std = math.sqrt(
            variance
        )

        if std == 0:
            std = 1.0

        means.append(
            feature_mean
        )

        stds.append(
            std
        )

    return means, stds


def standardize(
    features,
    means,
    stds
):
    return [
        (
            value
            - means[i]
        )
        / stds[i]

        for i, value
        in enumerate(features)
    ]


# ============================================================
# KNN MODEL
# ============================================================

def squared_distance(a, b):
    total = 0.0

    for x, y in zip(a, b):
        difference = x - y

        total += (
            difference
            * difference
        )

    return total


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
        d = squared_distance(
            input_features,
            features
        )

        distances.append(
            (d, target)
        )

    distances.sort(
        key=lambda item:
            item[0]
    )

    nearest = distances[:k]

    numerator = 0.0
    denominator = 0.0

    for distance_value, target in nearest:

        weight = (
            1.0
            /
            (
                math.sqrt(
                    distance_value
                )
                + 0.000001
            )
        )

        numerator += (
            weight
            * target
        )

        denominator += weight

    return (
        numerator
        / denominator
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):
    absolute_errors = []
    squared_errors = []

    for actual_value, predicted_value in zip(
        actual,
        predicted
    ):
        error = (
            actual_value
            - predicted_value
        )

        absolute_errors.append(
            abs(error)
        )

        squared_errors.append(
            error ** 2
        )

    mae = mean(
        absolute_errors
    )

    rmse = math.sqrt(
        mean(
            squared_errors
        )
    )

    actual_mean = mean(
        actual
    )

    ss_total = sum(
        (
            value
            - actual_mean
        ) ** 2

        for value in actual
    )

    ss_residual = sum(
        (
            actual[i]
            - predicted[i]
        ) ** 2

        for i in range(
            len(actual)
        )
    )

    if ss_total == 0:
        r2 = 0
    else:
        r2 = (
            1
            - ss_residual
            / ss_total
        )

    return mae, rmse, r2


def print_metrics(
    name,
    actual,
    predicted
):
    metrics = calculate_metrics(
        actual,
        predicted
    )

    print(
        f"\n{name}"
    )

    print("-" * 40)

    print(
        f"MAE:  {metrics[0]:.2f} °F"
    )

    print(
        f"RMSE: {metrics[1]:.2f} °F"
    )

    print(
        f"R²:   {metrics[2]:.3f}"
    )

    return metrics


# ============================================================
# TUNE K
# ============================================================

def tune_k(
    train,
    validation,
    means,
    stds
):
    print(
        "\nTUNING K USING VALIDATION DATA"
    )

    print("=" * 50)

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

    validation_features = [
        standardize(
            row["features"],
            means,
            stds
        )
        for row in validation
    ]

    validation_targets = [
        row["target"]
        for row in validation
    ]

    best_k = None
    best_mae = float("inf")

    print(
        f"{'K':<10}"
        f"{'Validation MAE':>20}"
    )

    print("-" * 30)

    for k in K_VALUES:
        predictions = []

        for features in validation_features:

            prediction = knn_predict(
                features,
                train_features,
                train_targets,
                k
            )

            predictions.append(
                prediction
            )

        mae, _, _ = calculate_metrics(
            validation_targets,
            predictions
        )

        print(
            f"{k:<10}"
            f"{mae:>17.2f} °F"
        )

        if mae < best_mae:
            best_mae = mae
            best_k = k

    print(
        "\nBest K:",
        best_k
    )

    print(
        "Best Validation MAE:",
        f"{best_mae:.2f} °F"
    )

    return best_k


# ============================================================
# SAVE RESULTS
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

        writer = csv.writer(
            file
        )

        writer.writerow([
            "input_date",
            "forecast_date",
            "today_high",
            "actual_tomorrow_high",
            "predicted_tomorrow_high",
            "absolute_error"
        ])

        for row, prediction in zip(
            test,
            predictions
        ):

            writer.writerow([
                row["date"].isoformat(),

                (
                    row["date"]
                    + timedelta(days=1)
                ).isoformat(),

                round(
                    row["today_max"],
                    2
                ),

                round(
                    row["target"],
                    2
                ),

                round(
                    prediction,
                    2
                ),

                round(
                    abs(
                        row["target"]
                        - prediction
                    ),
                    2
                )
            ])

    print(
        "\nPredictions saved to:",
        PREDICTIONS_FILE
    )


def save_model(
    train_features,
    train_targets,
    means,
    stds,
    best_k
):
    model_data = {
        "model_type":
            "Distance Weighted K-Nearest Neighbors Regression",

        "k":
            best_k,

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
        "Model saved to:",
        MODEL_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(
        "COLLEGE STATION WEATHER ML MODEL V2"
    )
    print("=" * 60)

    rows = download_weather_data()

    daily = create_daily_data(
        rows
    )

    dataset = create_ml_dataset(
        daily
    )

    train, validation, test = split_dataset(
        dataset
    )

    # Use only training data for selecting K
    train_means, train_stds = (
        calculate_scaler(
            train
        )
    )

    best_k = tune_k(
        train,
        validation,
        train_means,
        train_stds
    )

    # After K is chosen, combine training + validation
    final_training = (
        train
        + validation
    )

    means, stds = calculate_scaler(
        final_training
    )

    train_features = [
        standardize(
            row["features"],
            means,
            stds
        )
        for row in final_training
    ]

    train_targets = [
        row["target"]
        for row in final_training
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

    # ========================================================
    # BASELINE
    # ========================================================

    baseline_predictions = [
        row["today_max"]
        for row in test
    ]

    baseline_metrics = print_metrics(
        "Persistence Baseline",
        test_targets,
        baseline_predictions
    )

    # ========================================================
    # FINAL KNN MODEL
    # ========================================================

    print(
        "\nRunning final KNN model "
        f"with k={best_k}..."
    )

    predictions = []

    total = len(
        test_features
    )

    for index, features in enumerate(
        test_features
    ):

        prediction = knn_predict(
            features,
            train_features,
            train_targets,
            best_k
        )

        predictions.append(
            prediction
        )

        if (
            (index + 1) % 100 == 0
            or index + 1 == total
        ):
            print(
                f"Predicted "
                f"{index + 1}/"
                f"{total} test days"
            )

    knn_metrics = print_metrics(
        f"Improved KNN (k={best_k})",
        test_targets,
        predictions
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    print(
        "\nFINAL MODEL COMPARISON"
    )

    print("=" * 65)

    print(
        f"{'Model':<30}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'R²':>10}"
    )

    print("-" * 65)

    print(
        f"{'Persistence Baseline':<30}"
        f"{baseline_metrics[0]:>10.2f}"
        f"{baseline_metrics[1]:>10.2f}"
        f"{baseline_metrics[2]:>10.3f}"
    )

    print(
        f"{'Improved KNN':<30}"
        f"{knn_metrics[0]:>10.2f}"
        f"{knn_metrics[1]:>10.2f}"
        f"{knn_metrics[2]:>10.3f}"
    )

    improvement = (
        (
            baseline_metrics[0]
            - knn_metrics[0]
        )
        / baseline_metrics[0]
        * 100
    )

    print(
        "\nMAE improvement over baseline:",
        f"{improvement:.1f}%"
    )

    # ========================================================
    # WORST PREDICTIONS
    # ========================================================

    results = []

    for row, prediction in zip(
        test,
        predictions
    ):

        results.append({
            "input_date":
                row["date"],

            "forecast_date":
                row["date"]
                + timedelta(days=1),

            "actual":
                row["target"],

            "prediction":
                prediction,

            "error":
                abs(
                    row["target"]
                    - prediction
                )
        })

    results.sort(
        key=lambda item:
            item["error"],
        reverse=True
    )

    print(
        "\nWORST 10 PREDICTIONS"
    )

    print("=" * 85)

    print(
        f"{'Input Date':<15}"
        f"{'Forecast Date':<15}"
        f"{'Actual':>12}"
        f"{'Predicted':>15}"
        f"{'Error':>12}"
    )

    print("-" * 85)

    for result in results[:10]:

        print(
            f"{result['input_date'].isoformat():<15}"
            f"{result['forecast_date'].isoformat():<15}"
            f"{result['actual']:>12.1f}"
            f"{result['prediction']:>15.1f}"
            f"{result['error']:>12.1f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    save_predictions(
        test,
        predictions
    )

    save_model(
        train_features,
        train_targets,
        means,
        stds,
        best_k
    )

    # ========================================================
    # LATEST PREDICTION
    # ========================================================

    latest_date = max(
        daily.keys()
    )

    latest_row = None
    current_date = latest_date

    for _ in range(30):

        candidate = create_feature_row(
            current_date,
            daily,
            require_target=False
        )

        if candidate is not None:
            latest_row = candidate
            break

        current_date -= timedelta(
            days=1
        )

    if latest_row:

        features = standardize(
            latest_row["features"],
            means,
            stds
        )

        prediction = knn_predict(
            features,
            train_features,
            train_targets,
            best_k
        )

        print(
            "\nLATEST DATASET PREDICTION"
        )

        print("=" * 60)

        print(
            "Weather input date:",
            latest_row[
                "date"
            ].isoformat()
        )

        print(
            "Observed high:",
            f"{latest_row['today_max']:.1f} °F"
        )

        print(
            "Predicted next-day high:",
            f"{prediction:.1f} °F"
        )

    print(
        "\n" + "=" * 60
    )

    print("DONE")

    print("=" * 60)


if __name__ == "__main__":
    main()