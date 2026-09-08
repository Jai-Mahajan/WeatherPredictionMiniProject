import requests


def get_coordinates(city):
    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    if "results" not in data:
        return None

    result = data["results"][0]

    return {
        "name": result["name"],
        "latitude": result["latitude"],
        "longitude": result["longitude"]
    }


def get_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "wind_speed_10m"
        ],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    location = get_coordinates("College Station")

    if location is None:
        print("Location not found")
    else:
        print(location)

        weather = get_weather(
            location["latitude"],
            location["longitude"]
        )

        print(weather["current"])