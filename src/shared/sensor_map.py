"""
sensormap.py

Parametric script for bridge monitoring sensor mapping.
Run with command-line arguments to get sensor information.

Usage:
    # Get vertical axis for sensor 1
    python sensormap.py --sensor_id 1 --direction vertical

    # Get all axes for sensor 1
    python sensormap.py --sensor_id 1 --all_axes
    
    # Get all vertical sensors
    python sensormap.py --direction vertical --all_sensors

    # List available sensors
    python sensormap.py --list_sensors

"""
# sensormap.py
import argparse
from sensor_data import sensor_dict_combined

def get_sensor_direction(sensor_id: int, direction: str) -> str:
    directions = {'vertical': 0, 'flexural': 1, 'torsional': 2}
    if direction not in directions:
        raise ValueError(f"Invalid direction: {direction}")
    return sensor_dict_combined[sensor_id][directions[direction]]

def get_all_sensors_for_direction(direction: str):
    directions = {'vertical': 0, 'flexural': 1, 'torsional': 2}
    if direction not in directions:
        raise ValueError(f"Invalid direction: {direction}")
    idx = directions[direction]
    return {sid: axes[idx] for sid, axes in sensor_dict_combined.items()}

def main():
    parser = argparse.ArgumentParser(description='Bridge Sensor Mapping Tool')
    parser.add_argument('--sensor_id', type=int, help='Sensor ID (1-106)')
    parser.add_argument('--direction', type=str, choices=['vertical', 'flexural', 'torsional'])
    parser.add_argument('--all_axes', action='store_true')
    parser.add_argument('--all_sensors', action='store_true')
    parser.add_argument('--list_sensors', action='store_true')
    args = parser.parse_args()

    if args.list_sensors:
        print("Available sensor IDs:", list(sensor_dict_combined.keys()))
        return

    if args.sensor_id and args.all_axes:
        print(f"Sensor {args.sensor_id}: {sensor_dict_combined[args.sensor_id]}")
        return

    if args.sensor_id and args.direction:
        print(get_sensor_direction(args.sensor_id, args.direction))
        return

    if args.direction and args.all_sensors:
        print(get_all_sensors_for_direction(args.direction))
        return

    parser.print_help()

if __name__ == "__main__":
    main()