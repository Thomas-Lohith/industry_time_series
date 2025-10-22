# sensor_data.py- to load the sensor_map and it can be updated according to the bridge used

import csv

def load_sensor_map(csv_file="sensor_map.csv"):
    """
    Load sensor mapping from a CSV file into a dictionary:
    sensor_id -> (vertical, flexural, torsional)
    """
    sensor_dict = {}
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sensor_id = int(row["sensor_id"])
            sensor_dict[sensor_id] = (
                row["vertical"],
                row["flexural"],
                row["torsional"],
            )
    return sensor_dict

# Load immediately when imported
sensor_dict_combined = load_sensor_map()