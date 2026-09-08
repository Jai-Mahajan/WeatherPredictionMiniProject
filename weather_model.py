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

K_VALUES = [
    3,
    5,
    7,
    10,
    15,
    20,
    30,
    40,
    50
]

MODEL_FILE = "college_station_weather_model.json"
PREDICTIONS_FILE = "weather_predictions.csv"


# ============================================================
# WEATHER STATIONS
# ============================================================

STATIONS = {
    "CLL": "College Station",
    "ACT": "Waco",
    "AUS": "Austin",
    "DFW": "Dallas Fort Worth",
    "IAH": "Houston"
}


# ============================================================
# BASIC HELPERS
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
# DOWNLOAD ONE STATION
# ============================================================

def download_station_data(
    station,
    station_name
):

    print(
        f"\nDownloading {station_name} "
        f"({station}) weather data..."
    )

    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={station}"
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

    try:

        with urllib.request.urlopen(
            url,
            timeout=180
        ) as response:

            text = (
                response
                .read()
                .decode("utf-8")
            )

    except Exception as error:

        print(
            f"ERROR downloading "
            f"{station_name}: {error}"
        )

        return []

    rows = list(
        csv.DictReader(
            StringIO(text)
        )
    )

    print(
        f"{station_name}: "
        f"{len(rows):,} observations downloaded."
    )

    return rows


# ============================================================
# DOWNLOAD ALL STATIONS
# ============================================================

def download_all_weather_data():

    all_station_rows = {}

    print("\n" + "=" * 60)
    print("DOWNLOADING TEXAS WEATHER NETWORK")
    print("=" * 60)

    for station, name in STATIONS.items():

        rows = download_station_data(
            station,
            name
        )

        all_station_rows[
            station
        ] = rows

    return all_station_rows


# ============================================================
# CREATE DAILY WEATHER FOR ONE STATION
# ============================================================

def create_daily_station_data(
    rows,
    station_name
):

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

        except (
            ValueError,
            KeyError
        ):

            continue

        date = timestamp.date()

        temp = safe_float(
            row.get("tmpf")
        )

        dew = safe_float(
            row.get("dwpf")
        )

        humidity = safe_float(
            row.get("relh")
        )

        wind = safe_float(
            row.get("sknt")
        )

        direction = safe_float(
            row.get("drct")
        )

        pressure = safe_float(
            row.get("alti")
        )

        precip = safe_float(
            row.get("p01i")
        )

        # ----------------------------------------------------
        # BASIC QUALITY CONTROL
        # ----------------------------------------------------

        if (
            temp is not None
            and -30 <= temp <= 135
        ):
            grouped[
                date
            ]["temps"].append(
                temp
            )

        if (
            dew is not None
            and -50 <= dew <= 105
        ):
            grouped[
                date
            ]["dewpoints"].append(
                dew
            )

        if (
            humidity is not None
            and 0 <= humidity <= 100
        ):
            grouped[
                date
            ]["humidity"].append(
                humidity
            )

        if (
            wind is not None
            and 0 <= wind <= 120
        ):
            grouped[
                date
            ]["wind"].append(
                wind
            )

        if (
            direction is not None
            and 0 <= direction <= 360
        ):
            grouped[
                date
            ]["direction"].append(
                direction
            )

        if (
            pressure is not None
            and 20 <= pressure <= 35
        ):
            grouped[
                date
            ]["pressure"].append(
                pressure
            )

        if (
            precip is not None
            and precip >= 0
        ):
            grouped[
                date
            ]["precipitation"].append(
                precip
            )

    daily = {}

    for date, data in grouped.items():

        # Require at least 12 temperature
        # observations for the day.
        if len(
            data["temps"]
        ) < 12:
            continue

        if (
            not data["dewpoints"]
            or not data["humidity"]
            or not data["wind"]
            or not data["pressure"]
        ):
            continue

        temps = data["temps"]

        dewpoints = (
            data["dewpoints"]
        )

        pressures = (
            data["pressure"]
        )

        wind_direction = (
            circular_mean_degrees(
                data["direction"]
            )
        )

        if wind_direction is None:
            wind_direction = 0.0

        daily[date] = {

            "max_temp":
                max(temps),

            "min_temp":
                min(temps),

            "avg_temp":
                mean(temps),

            "avg_dewpoint":
                mean(dewpoints),

            "avg_humidity":
                mean(
                    data["humidity"]
                ),

            "avg_wind":
                mean(
                    data["wind"]
                ),

            "max_wind":
                max(
                    data["wind"]
                ),

            "wind_direction":
                wind_direction,

            "avg_pressure":
                mean(pressures),

            "precipitation":
                sum(
                    data[
                        "precipitation"
                    ]
                )
                if data[
                    "precipitation"
                ]
                else 0.0,

            # -----------------------------------------------
            # INTRADAY CHANGE FEATURES
            # -----------------------------------------------

            "temp_intraday_change":
                last_mean(temps)
                -
                first_mean(temps),

            "pressure_intraday_change":
                last_mean(
                    pressures
                )
                -
                first_mean(
                    pressures
                ),

            "dewpoint_intraday_change":
                last_mean(
                    dewpoints
                )
                -
                first_mean(
                    dewpoints
                )
        }

    print(
        f"{station_name}: "
        f"{len(daily):,} usable days"
    )

    return daily


# ============================================================
# CREATE DAILY DATA FOR ALL STATIONS
# ============================================================

def create_all_daily_data(
    all_station_rows
):

    print("\n" + "=" * 60)
    print("CREATING DAILY WEATHER DATA")
    print("=" * 60)

    daily_data = {}

    for station, rows in (
        all_station_rows.items()
    ):

        daily_data[
            station
        ] = create_daily_station_data(
            rows,
            STATIONS[station]
        )

    return daily_data


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [

    # ========================================================
    # COLLEGE STATION CURRENT WEATHER
    # ========================================================

    "cll_max_temp",
    "cll_min_temp",
    "cll_avg_temp",
    "cll_dewpoint",
    "cll_humidity",
    "cll_wind",
    "cll_max_wind",
    "cll_pressure",
    "cll_precip",

    "cll_wind_sin",
    "cll_wind_cos",

    "cll_temp_intraday",
    "cll_pressure_intraday",
    "cll_dewpoint_intraday",

    # ========================================================
    # COLLEGE STATION HISTORY
    # ========================================================

    "cll_high_lag1",
    "cll_high_lag2",
    "cll_high_lag3",
    "cll_high_lag5",
    "cll_high_lag7",

    "cll_temp_3day_avg",
    "cll_temp_7day_avg",

    "cll_temp_change_1d",
    "cll_pressure_change_1d",
    "cll_dewpoint_change_1d",

    # ========================================================
    # WACO
    # ========================================================

    "waco_temp",
    "waco_dewpoint",
    "waco_pressure",
    "waco_wind",
    "waco_wind_sin",
    "waco_wind_cos",

    "waco_temp_difference",
    "waco_pressure_difference",

    # ========================================================
    # DALLAS
    # ========================================================

    "dfw_temp",
    "dfw_dewpoint",
    "dfw_pressure",
    "dfw_wind",
    "dfw_wind_sin",
    "dfw_wind_cos",

    "dfw_temp_difference",
    "dfw_pressure_difference",

    # ========================================================
    # AUSTIN
    # ========================================================

    "austin_temp",
    "austin_dewpoint",
    "austin_pressure",
    "austin_wind",

    "austin_temp_difference",

    # ========================================================
    # HOUSTON
    # ========================================================

    "houston_temp",
    "houston_dewpoint",
    "houston_pressure",
    "houston_wind",

    "houston_temp_difference",

    # ========================================================
    # NORTH-SOUTH GRADIENTS
    # ========================================================

    "dfw_cll_temp_gradient",
    "waco_cll_temp_gradient",

    "dfw_cll_pressure_gradient",
    "waco_cll_pressure_gradient",

    # ========================================================
    # SEASON
    # ========================================================

    "day_sin",
    "day_cos"
]


# ============================================================
# CREATE FEATURE ROW
# ============================================================

def create_feature_row(
    date,
    daily_data,
    require_target=True
):

    cll_data = daily_data["CLL"]

    current = cll_data.get(
        date
    )

    if current is None:
        return None

    # --------------------------------------------------------
    # NEED 7 PREVIOUS CLL DAYS
    # --------------------------------------------------------

    previous = []

    for lag in range(
        1,
        8
    ):

        previous_date = (
            date
            - timedelta(days=lag)
        )

        if (
            previous_date
            not in cll_data
        ):
            return None

        previous.append(
            cll_data[
                previous_date
            ]
        )

    tomorrow = cll_data.get(
        date
        + timedelta(days=1)
    )

    if (
        require_target
        and tomorrow is None
    ):
        return None

    # --------------------------------------------------------
    # REQUIRE NEARBY STATIONS
    # --------------------------------------------------------

    waco = daily_data[
        "ACT"
    ].get(date)

    austin = daily_data[
        "AUS"
    ].get(date)

    dfw = daily_data[
        "DFW"
    ].get(date)

    houston = daily_data[
        "IAH"
    ].get(date)

    if (
        waco is None
        or austin is None
        or dfw is None
        or houston is None
    ):
        return None

    # --------------------------------------------------------
    # SEASONAL FEATURES
    # --------------------------------------------------------

    day_of_year = (
        date
        .timetuple()
        .tm_yday
    )

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

    # --------------------------------------------------------
    # WIND DIRECTION FEATURES
    # --------------------------------------------------------

    cll_rad = math.radians(
        current[
            "wind_direction"
        ]
    )

    waco_rad = math.radians(
        waco[
            "wind_direction"
        ]
    )

    dfw_rad = math.radians(
        dfw[
            "wind_direction"
        ]
    )

    cll_wind_sin = math.sin(
        cll_rad
    )

    cll_wind_cos = math.cos(
        cll_rad
    )

    waco_wind_sin = math.sin(
        waco_rad
    )

    waco_wind_cos = math.cos(
        waco_rad
    )

    dfw_wind_sin = math.sin(
        dfw_rad
    )

    dfw_wind_cos = math.cos(
        dfw_rad
    )

    # --------------------------------------------------------
    # ROLLING TEMPERATURE FEATURES
    # --------------------------------------------------------

    temp_3day_avg = mean([
        current["max_temp"],
        previous[0]["max_temp"],
        previous[1]["max_temp"]
    ])

    temp_7day_avg = mean([
        current["max_temp"],
        previous[0]["max_temp"],
        previous[1]["max_temp"],
        previous[2]["max_temp"],
        previous[3]["max_temp"],
        previous[4]["max_temp"],
        previous[5]["max_temp"]
    ])

    # --------------------------------------------------------
    # LOCAL WEATHER CHANGE
    # --------------------------------------------------------

    temp_change_1d = (
        current["max_temp"]
        -
        previous[0]["max_temp"]
    )

    pressure_change_1d = (
        current["avg_pressure"]
        -
        previous[0]["avg_pressure"]
    )

    dewpoint_change_1d = (
        current["avg_dewpoint"]
        -
        previous[0]["avg_dewpoint"]
    )

    # --------------------------------------------------------
    # DIFFERENCES BETWEEN CITIES
    #
    # These are especially important for detecting
    # fronts approaching College Station.
    #
    # Example:
    #
    # DFW 35°F
    # CLL 65°F
    #
    # gradient = -30°F
    #
    # That may indicate much colder air to the north.
    # --------------------------------------------------------

    waco_temp_difference = (
        waco["avg_temp"]
        -
        current["avg_temp"]
    )

    dfw_temp_difference = (
        dfw["avg_temp"]
        -
        current["avg_temp"]
    )

    austin_temp_difference = (
        austin["avg_temp"]
        -
        current["avg_temp"]
    )

    houston_temp_difference = (
        houston["avg_temp"]
        -
        current["avg_temp"]
    )

    waco_pressure_difference = (
        waco["avg_pressure"]
        -
        current["avg_pressure"]
    )

    dfw_pressure_difference = (
        dfw["avg_pressure"]
        -
        current["avg_pressure"]
    )

    # --------------------------------------------------------
    # FINAL FEATURE VECTOR
    # --------------------------------------------------------

    features = [

        # CLL current weather
        current["max_temp"],
        current["min_temp"],
        current["avg_temp"],
        current["avg_dewpoint"],
        current["avg_humidity"],
        current["avg_wind"],
        current["max_wind"],
        current["avg_pressure"],
        current["precipitation"],

        cll_wind_sin,
        cll_wind_cos,

        current[
            "temp_intraday_change"
        ],

        current[
            "pressure_intraday_change"
        ],

        current[
            "dewpoint_intraday_change"
        ],

        # CLL history
        previous[0]["max_temp"],
        previous[1]["max_temp"],
        previous[2]["max_temp"],
        previous[4]["max_temp"],
        previous[6]["max_temp"],

        temp_3day_avg,
        temp_7day_avg,

        temp_change_1d,
        pressure_change_1d,
        dewpoint_change_1d,

        # Waco
        waco["avg_temp"],
        waco["avg_dewpoint"],
        waco["avg_pressure"],
        waco["avg_wind"],

        waco_wind_sin,
        waco_wind_cos,

        waco_temp_difference,
        waco_pressure_difference,

        # Dallas
        dfw["avg_temp"],
        dfw["avg_dewpoint"],
        dfw["avg_pressure"],
        dfw["avg_wind"],

        dfw_wind_sin,
        dfw_wind_cos,

        dfw_temp_difference,
        dfw_pressure_difference,

        # Austin
        austin["avg_temp"],
        austin["avg_dewpoint"],
        austin["avg_pressure"],
        austin["avg_wind"],

        austin_temp_difference,

        # Houston
        houston["avg_temp"],
        houston["avg_dewpoint"],
        houston["avg_pressure"],
        houston["avg_wind"],

        houston_temp_difference,

        # Explicit north-to-CLL gradients
        dfw["avg_temp"]
        -
        current["avg_temp"],

        waco["avg_temp"]
        -
        current["avg_temp"],

        dfw["avg_pressure"]
        -
        current["avg_pressure"],

        waco["avg_pressure"]
        -
        current["avg_pressure"],

        # Season
        day_sin,
        day_cos
    ]

    return {

        "date":
            date,

        "features":
            features,

        "target":
            tomorrow["max_temp"]
            if tomorrow
            else None,

        "today_max":
            current["max_temp"]
    }


# ============================================================
# BUILD ML DATASET
# ============================================================

def create_ml_dataset(
    daily_data
):

    print("\n" + "=" * 60)
    print("ENGINEERING MULTI-CITY ML FEATURES")
    print("=" * 60)

    dataset = []

    for date in sorted(
        daily_data[
            "CLL"
        ].keys()
    ):

        row = create_feature_row(
            date,
            daily_data,
            require_target=True
        )

        if row is not None:
            dataset.append(row)

    print(
        f"Final multi-city ML samples: "
        f"{len(dataset):,}"
    )

    return dataset


# ============================================================
# SPLIT DATA
# ============================================================

def split_dataset(
    dataset
):

    train = []
    validation = []
    test = []

    for row in dataset:

        if (
            row["date"]
            <
            datetime(
                2023,
                1,
                1
            ).date()
        ):

            train.append(row)

        elif (
            row["date"]
            <
            datetime(
                2025,
                1,
                1
            ).date()
        ):

            validation.append(
                row
            )

        else:

            test.append(
                row
            )

    print("\nDataset Split")
    print("-" * 40)

    print(
        f"Training:   "
        f"{len(train):,}"
    )

    print(
        f"Validation: "
        f"{len(validation):,}"
    )

    print(
        f"Testing:    "
        f"{len(test):,}"
    )

    return (
        train,
        validation,
        test
    )


# ============================================================
# STANDARDIZATION
# ============================================================

def calculate_scaler(
    dataset
):

    feature_count = len(
        dataset[0][
            "features"
        ]
    )

    means = []
    stds = []

    for feature_index in range(
        feature_count
    ):

        values = [

            row[
                "features"
            ][
                feature_index
            ]

            for row in dataset
        ]

        feature_mean = mean(
            values
        )

        variance = mean([

            (
                value
                -
                feature_mean
            ) ** 2

            for value
            in values
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

    return (
        means,
        stds
    )


def standardize(
    features,
    means,
    stds
):

    return [

        (
            value
            -
            means[index]
        )
        /
        stds[index]

        for index, value
        in enumerate(
            features
        )
    ]


# ============================================================
# KNN
# ============================================================

def squared_distance(
    a,
    b
):

    total = 0.0

    for x, y in zip(
        a,
        b
    ):

        difference = (
            x - y
        )

        total += (
            difference
            *
            difference
        )

    return total


def knn_predict(
    input_features,
    train_features,
    train_targets,
    k
):

    distances = []

    for (
        features,
        target
    ) in zip(
        train_features,
        train_targets
    ):

        distance = (
            squared_distance(
                input_features,
                features
            )
        )

        distances.append(
            (
                distance,
                target
            )
        )

    distances.sort(
        key=lambda item:
            item[0]
    )

    nearest = (
        distances[:k]
    )

    numerator = 0.0
    denominator = 0.0

    for (
        distance,
        target
    ) in nearest:

        weight = (
            1.0
            /
            (
                math.sqrt(
                    distance
                )
                +
                0.000001
            )
        )

        numerator += (
            weight
            *
            target
        )

        denominator += (
            weight
        )

    return (
        numerator
        /
        denominator
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

    for (
        actual_value,
        predicted_value
    ) in zip(
        actual,
        predicted
    ):

        error = (
            actual_value
            -
            predicted_value
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
            -
            actual_mean
        ) ** 2

        for value
        in actual
    )

    ss_residual = sum(

        (
            actual[index]
            -
            predicted[index]
        ) ** 2

        for index
        in range(
            len(actual)
        )
    )

    if ss_total == 0:

        r2 = 0.0

    else:

        r2 = (
            1
            -
            (
                ss_residual
                /
                ss_total
            )
        )

    return (
        mae,
        rmse,
        r2
    )


def print_metrics(
    name,
    actual,
    predicted
):

    mae, rmse, r2 = (
        calculate_metrics(
            actual,
            predicted
        )
    )

    print(
        f"\n{name}"
    )

    print(
        "-" * 40
    )

    print(
        f"MAE:  "
        f"{mae:.2f} °F"
    )

    print(
        f"RMSE: "
        f"{rmse:.2f} °F"
    )

    print(
        f"R²:   "
        f"{r2:.3f}"
    )

    return (
        mae,
        rmse,
        r2
    )


# ============================================================
# K TUNING
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

    print(
        "=" * 50
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

    validation_features = [

        standardize(
            row["features"],
            means,
            stds
        )

        for row
        in validation
    ]

    validation_targets = [

        row["target"]

        for row
        in validation
    ]

    best_k = None

    best_mae = float(
        "inf"
    )

    print(
        f"{'K':<10}"
        f"{'Validation MAE':>20}"
    )

    print(
        "-" * 30
    )

    for k in K_VALUES:

        predictions = []

        for index, features in enumerate(
            validation_features
        ):

            prediction = (
                knn_predict(
                    features,
                    train_features,
                    train_targets,
                    k
                )
            )

            predictions.append(
                prediction
            )

        mae, _, _ = (
            calculate_metrics(
                validation_targets,
                predictions
            )
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

        for (
            row,
            prediction
        ) in zip(
            test,
            predictions
        ):

            writer.writerow([

                row[
                    "date"
                ].isoformat(),

                (
                    row["date"]
                    +
                    timedelta(days=1)
                ).isoformat(),

                round(
                    row[
                        "today_max"
                    ],
                    2
                ),

                round(
                    row[
                        "target"
                    ],
                    2
                ),

                round(
                    prediction,
                    2
                ),

                round(
                    abs(
                        row["target"]
                        -
                        prediction
                    ),
                    2
                )
            ])

    print(
        "\nPredictions saved to:",
        PREDICTIONS_FILE
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    train_features,
    train_targets,
    means,
    stds,
    best_k
):

    model_data = {

        "model_type":
            "Multi-City Distance Weighted KNN",

        "stations":
            STATIONS,

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

    print(
        "=" * 65
    )

    print(
        "COLLEGE STATION WEATHER ML MODEL V3"
    )

    print(
        "MULTI-CITY WEATHER NETWORK"
    )

    print(
        "=" * 65
    )

    # ========================================================
    # DOWNLOAD WEATHER
    # ========================================================

    all_station_rows = (
        download_all_weather_data()
    )

    # ========================================================
    # DAILY WEATHER
    # ========================================================

    daily_data = (
        create_all_daily_data(
            all_station_rows
        )
    )

    # ========================================================
    # ML DATASET
    # ========================================================

    dataset = (
        create_ml_dataset(
            daily_data
        )
    )

    if not dataset:

        print(
            "\nERROR: No ML samples generated."
        )

        return

    # ========================================================
    # SPLIT
    # ========================================================

    (
        train,
        validation,
        test
    ) = split_dataset(
        dataset
    )

    # ========================================================
    # SCALE TRAINING DATA
    # ========================================================

    (
        train_means,
        train_stds
    ) = calculate_scaler(
        train
    )

    # ========================================================
    # FIND BEST K
    # ========================================================

    best_k = tune_k(

        train,

        validation,

        train_means,

        train_stds
    )

    # ========================================================
    # COMBINE TRAIN + VALIDATION
    # ========================================================

    final_training = (
        train
        +
        validation
    )

    (
        means,
        stds
    ) = calculate_scaler(
        final_training
    )

    train_features = [

        standardize(
            row["features"],
            means,
            stds
        )

        for row
        in final_training
    ]

    train_targets = [

        row["target"]

        for row
        in final_training
    ]

    test_features = [

        standardize(
            row["features"],
            means,
            stds
        )

        for row
        in test
    ]

    test_targets = [

        row["target"]

        for row
        in test
    ]

    # ========================================================
    # BASELINE
    # ========================================================

    baseline_predictions = [

        row[
            "today_max"
        ]

        for row
        in test
    ]

    baseline_metrics = (
        print_metrics(
            "Persistence Baseline",
            test_targets,
            baseline_predictions
        )
    )

    # ========================================================
    # RUN FINAL MODEL
    # ========================================================

    print(
        "\nRunning multi-city KNN "
        f"with k={best_k}..."
    )

    predictions = []

    total = len(
        test_features
    )

    for index, features in enumerate(
        test_features
    ):

        prediction = (
            knn_predict(
                features,
                train_features,
                train_targets,
                best_k
            )
        )

        predictions.append(
            prediction
        )

        if (
            (index + 1) % 100 == 0
            or
            index + 1 == total
        ):

            print(
                f"Predicted "
                f"{index + 1}/"
                f"{total} test days"
            )

    model_metrics = (
        print_metrics(
            f"Multi-City KNN (k={best_k})",
            test_targets,
            predictions
        )
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    print(
        "\nFINAL MODEL COMPARISON"
    )

    print(
        "=" * 65
    )

    print(
        f"{'Model':<30}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'R²':>10}"
    )

    print(
        "-" * 65
    )

    print(
        f"{'Persistence Baseline':<30}"
        f"{baseline_metrics[0]:>10.2f}"
        f"{baseline_metrics[1]:>10.2f}"
        f"{baseline_metrics[2]:>10.3f}"
    )

    print(
        f"{'Multi-City KNN':<30}"
        f"{model_metrics[0]:>10.2f}"
        f"{model_metrics[1]:>10.2f}"
        f"{model_metrics[2]:>10.3f}"
    )

    improvement = (

        (
            baseline_metrics[0]
            -
            model_metrics[0]
        )

        /
        baseline_metrics[0]

        *
        100
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
                +
                timedelta(
                    days=1
                ),

            "actual":
                row["target"],

            "prediction":
                prediction,

            "error":
                abs(
                    row["target"]
                    -
                    prediction
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

    print(
        "=" * 85
    )

    print(
        f"{'Input Date':<15}"
        f"{'Forecast Date':<15}"
        f"{'Actual':>12}"
        f"{'Predicted':>15}"
        f"{'Error':>12}"
    )

    print(
        "-" * 85
    )

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
    # LATEST FORECAST
    # ========================================================

    latest_date = max(
        daily_data[
            "CLL"
        ].keys()
    )

    latest_row = None

    current_date = (
        latest_date
    )

    for _ in range(
        30
    ):

        candidate = (
            create_feature_row(
                current_date,
                daily_data,
                require_target=False
            )
        )

        if candidate is not None:

            latest_row = (
                candidate
            )

            break

        current_date -= (
            timedelta(
                days=1
            )
        )

    if latest_row:

        latest_features = (
            standardize(
                latest_row[
                    "features"
                ],
                means,
                stds
            )
        )

        latest_prediction = (
            knn_predict(
                latest_features,
                train_features,
                train_targets,
                best_k
            )
        )

        print(
            "\nLATEST MULTI-CITY PREDICTION"
        )

        print(
            "=" * 60
        )

        print(
            "Weather input date:",
            latest_row[
                "date"
            ].isoformat()
        )

        print(
            "College Station observed high:",
            f"{latest_row['today_max']:.1f} °F"
        )

        print(
            "Predicted next-day high:",
            f"{latest_prediction:.1f} °F"
        )

    print(
        "\n" + "=" * 65
    )

    print("DONE")

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()