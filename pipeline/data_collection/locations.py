"""
Location configuration.

DEFAULT_LOCATION  - the monitored city: hourly pipeline, stored history, accuracy tracking.
TRAINING_CITIES   - training corpus for the GLOBAL model: diverse climates on every
                    continent, so the model can forecast any place on Earth.
"""

DEFAULT_LOCATION = {"name": "Bengaluru", "latitude": 12.97, "longitude": 77.59,
                    "country": "India", "timezone": "Asia/Kolkata"}

# (name, latitude, longitude)
TRAINING_CITIES = [
    # India - tropical, coastal, desert, north-east, mountain
    ("Bengaluru", 12.97, 77.59), ("Chennai", 13.08, 80.27), ("Mumbai", 19.08, 72.88),
    ("Delhi", 28.61, 77.21), ("Kolkata", 22.57, 88.36), ("Hyderabad", 17.39, 78.49),
    ("Jaipur", 26.91, 75.79), ("Guwahati", 26.14, 91.74), ("Kochi", 9.93, 76.27),
    ("Srinagar", 34.08, 74.80), ("Shimla", 31.10, 77.17),
    # Asia & Middle East
    ("Singapore", 1.35, 103.82), ("Tokyo", 35.68, 139.69), ("Beijing", 39.90, 116.40),
    ("Dubai", 25.20, 55.27),
    # Europe
    ("London", 51.51, -0.13), ("Madrid", 40.42, -3.70), ("Moscow", 55.76, 37.62),
    ("Oslo", 59.91, 10.75),
    # Americas
    ("New York", 40.71, -74.01), ("Chicago", 41.88, -87.63), ("Phoenix", 33.45, -112.07),
    ("Mexico City", 19.43, -99.13), ("Lima", -12.05, -77.04), ("Sao Paulo", -23.55, -46.63),
    ("Buenos Aires", -34.60, -58.38),
    # Africa
    ("Cairo", 30.04, 31.24), ("Lagos", 6.52, 3.38), ("Nairobi", -1.29, 36.82),
    ("Johannesburg", -26.20, 28.05),
    # Oceania
    ("Sydney", -33.87, 151.21), ("Auckland", -36.85, 174.76),
]
