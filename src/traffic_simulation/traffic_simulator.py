import argparse
import csv
import json
import math 
import os 
import random

from src.shared.bridge_model import load_bridge
from src.shared.config import position_csv, threshold_csv, delimiter

#method to read the config file
def load_config(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise SystemExit(f"Error loading config: {e}, PyYaml not avialable or config unavialble"
                 "Please install PyYaml and try again.")

#method to load the sensors from the config file
def load_sensors(cfg):
    s = cfg['sensors']  
    cols = s['columns']
    sensors = []
    with open(s['csv_path']) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sensors.append({
                'sensor_id': row[cols['id']],
                'sensor_vertical': row[cols['vertical']],
                'dist_m': float(row[cols['distance']]),
                'span': row.get(cols['span'], '')
            })
    # oredering the sensors by position so the detections are postional