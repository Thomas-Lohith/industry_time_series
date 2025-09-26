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


import argparse

# Sensor mapping: numeric ID -> (vertical_axis, flexural_axis, torsional_axis)
sensor_dict_combined = {
    1: ("03091011_x", "03091011_y", "03091011_z"),
    2: ("030911F1_x", "030911F1_y", "030911F1_z"),
    3: ("030910F5_x", "030910F5_y", "030910F5_z"),
    4: ("0309120A_z", "0309120A_y", "0309120A_x"),
    5: ("03091043_x", "03091043_y", "03091043_z"),
    6: ("0309102D_x", "0309102D_y", "0309102D_z"),
    7: ("030911EE_x", "030911EE_y", "030911EE_z"),
    8: ("03091042_x", "03091042_y", "03091042_z"),
    9: ("03091126_x", "03091126_y", "03091126_z"),
    10: ("03091028_z", "03091028_y", "03091028_x"),
    11: ("03091024_x", "03091024_y", "03091024_z"),
    12: ("03091114_x", "03091114_y", "03091114_z"),
    13: ("03091205_x", "03091205_y", "03091205_z"),
    14: ("03091047_z", "03091047_y", "03091047_x"),
    15: ("030911E0_x", "030911E0_y", "030911E0_z"),
    16: ("030911F2_x", "030911F2_y", "030911F2_z"),
    17: ("030911F3_x", "030911F3_y", "030911F3_z"),
    18: ("0309114D_x", "0309114D_y", "0309114D_z"),
    19: ("0309120B_x", "0309120B_y", "0309120B_z"),
    20: ("03091110_z", "03091110_y", "03091110_x"),
    21: ("03091208_x", "03091208_y", "03091208_z"),
    22: ("03091033_x", "03091033_y", "03091033_z"),
    23: ("0309120C_x", "0309120C_y", "0309120C_z"),
    24: ("030910FA_z", "030910FA_y", "030910FA_z"),
    25: ("03091041_x", "03091041_y", "03091041_z"),
    26: ("030911ED_x", "030911ED_y", "030911ED_z"),
    27: ("03091013_x", "03091013_y", "03091013_z"),
    28: ("03091202_x", "03091202_y", "03091202_z"),
    29: ("030911D3_x", "030911D3_y", "030911D3_z"),
    30: ("03091016_z", "03091016_y", "03091016_x"),
    31: ("030910F4_x", "030910F4_y", "030910F4_z"),
    32: ("030910F2_x", "030910F2_y", "030910F2_z"),
    33: ("03091021_x", "03091021_y", "03091021_z"),
    34: ("03091127_z", "03091127_y", "03091127_x"),
    35: ("030910F7_x", "030910F7_y", "030910F7_z"),
    36: ("03091132_x", "03091132_y", "03091132_z"),
    37: ("03091046_x", "03091046_y", "03091046_z"),
    38: ("03091115_x", "03091115_y", "03091115_z"),
    39: ("03091118_x", "03091118_y", "03091118_z"),
    40: ("03091117_z", "03091117_y", "03091117_x"),
    41: ("030910F9_x", "030910F9_y", "030910F9_z"),
    42: ("03091129_x", "03091129_y", "03091129_z"),
    43: ("0309120D_x", "0309120D_y", "0309120D_z"),
    44: ("03091035_z", "03091035_y", "03091035_x"),
    45: ("03091211_x", "03091211_y", "03091211_z"),
    46: ("0309120E_x", "0309120E_y", "0309120E_z"),
    47: ("03091210_x", "03091210_y", "03091210_z"),
    48: ("03091153_x", "03091153_y", "03091153_z"),
    49: ("03091010_x", "03091010_y", "03091010_z"),
    50: ("03091018_z", "03091018_y", "03091018_x"),
    51: ("0309101E_x", "0309101E_y", "0309101E_z"),
    52: ("030910F6_x", "030910F6_y", "030910F6_z"),
    53: ("0309100F_x", "0309100F_y", "0309100F_z"),
    54: ("03091003_x", "03091003_y", "03091003_z"),
    55: ("03091006_x", "03091006_y", "03091006_z"),
    56: ("03091034_x", "03091034_y", "03091034_z"),
    57: ("03091119_z", "03091119_y", "03091119_x"),
    58: ("030911EC_x", "030911EC_y", "030911EC_z"),
    59: ("03091007_x", "03091007_y", "03091007_z"),
    60: ("0309112A_x", "0309112A_y", "0309112A_z"),
    61: ("03091032_x", "03091032_y", "03091032_z"),
    62: ("0309112F_x", "0309112F_y", "0309112F_z"),
    63: ("0309100B_z", "0309100B_y", "0309100B_x"),
    64: ("0309102F_x", "0309102F_y", "0309102F_z"),
    65: ("03091031_x", "03091031_y", "03091031_z"),
    66: ("03091212_x", "03091212_y", "03091212_z"),
    67: ("03091111_z", "03091111_y", "03091111_x"),
    68: ("03091012_x", "03091012_y", "03091012_z"),
    69: ("03091128_x", "03091128_y", "03091128_z"),
    70: ("03091112_x", "03091112_y", "03091112_z"),
    71: ("03091002_x", "03091002_y", "03091002_z"),
    72: ("0309101D_x", "0309101D_y", "0309101D_z"),
    73: ("0309102E_z", "0309102E_y", "0309102E_x"),
    74: ("030911F4_x", "030911F4_y", "030911F4_z"),
    75: ("030911AE_x", "030911AE_y", "030911AE_z"),
    76: ("03091015_x", "03091015_y", "03091015_z"),
    77: ("03091045_z", "03091045_y", "03091045_x"),
    78: ("03091204_x", "03091204_y", "03091204_z"),
    79: ("0309114E_x", "0309114E_y", "0309114E_z"),
    80: ("0309120F_x", "0309120F_y", "0309120F_z"),
    81: ("03091201_x", "03091201_y", "03091201_z"),
    82: ("03091019_x", "03091019_y", "03091019_z"),
    83: ("0309123B_z", "0309123B_y", "0309123B_x"),
    84: ("03091030_x", "03091030_y", "03091030_z"),
    85: ("0309100A_x", "0309100A_y", "0309100A_z"),
    86: ("0309102A_x", "0309102A_y", "0309102A_z"),
    87: ("030911DF_z", "030911DF_y", "030911DF_x"),
    88: ("03091008_x", "03091008_y", "03091008_z"),
    89: ("03091113_x", "03091113_y", "03091113_z"),
    90: ("03091049_x", "03091049_y", "03091049_z"),
    91: ("03091014_x", "03091014_y", "03091014_z"),
    92: ("03091048_x", "03091048_y", "03091048_z"),
    93: ("0309113F_z", "0309113F_y", "0309113F_x"),
    94: ("030911E5_x", "030911E5_y", "030911E5_z"),
    95: ("03091131_x", "03091131_y", "03091131_z"),
    96: ("03091207_x", "03091207_y", "03091207_z"),
    97: ("03091017_z", "03091017_y", "03091017_x"),
    98: ("03091036_x", "03091036_y", "03091036_z"),
    99: ("030911D2_x", "030911D2_y", "030911D2_z"),
    100: ("03091005_x", "03091005_y", "03091005_z"),
    101: ("0309101F_x", "0309101F_y", "0309101F_z"),
    102: ("03091203_x", "03091203_y", "03091203_z"),
    103: ("03091155_z", "03091155_y", "03091155_x"),
    104: ("03091200_x", "03091200_y", "03091200_z"),
    105: ("030911EF_x", "030911EF_y", "030911EF_z"),
    106: ("030911FF_x", "030911FF_y", "030911FF_z"),
}

# Placeholder span_map: numeric sensor_id -> bridge span number
span_map = {}

def get_sensor_direction(sensor_id: int, direction: str) -> str:
    """Get the sensor column name for a given sensor and direction axis."""
    directions = {'vertical': 0, 'flexural': 1, 'torsional': 2}
    if direction not in directions:
        raise ValueError(f"Invalid direction: {direction}. Must be one of {list(directions.keys())}")
    return sensor_dict_combined[sensor_id][directions[direction]]

def get_all_sensors_for_direction(direction: str) -> dict[int, str]:
    """Get a dictionary of all sensor IDs mapped to their column names for specified direction."""
    directions = {'vertical': 0, 'flexural': 1, 'torsional': 2}
    if direction not in directions:
        raise ValueError(f"Invalid direction: {direction}. Must be one of {list(directions.keys())}")
    idx = directions[direction]
    return {sensor_id: axes[idx] for sensor_id, axes in sensor_dict_combined.items()}

def main():
    parser = argparse.ArgumentParser(description='Bridge Sensor Mapping Tool')
    parser.add_argument('--sensor_id', type=int, help='Sensor ID (1-106)')
    parser.add_argument('--direction', type=str, choices=['vertical', 'flexural', 'torsional'], 
                       help='Sensor direction')
    parser.add_argument('--all_axes', action='store_true', 
                       help='Show all axes for specified sensor')
    parser.add_argument('--all_sensors', action='store_true', 
                       help='Show all sensors for specified direction')
    parser.add_argument('--list_sensors', action='store_true', 
                       help='List all available sensor IDs')
    
    args = parser.parse_args()
    
    if args.list_sensors:
        print("Available sensor IDs: 1-106")
        print("Total sensors:", len(sensor_dict_combined))
        return
    
    if args.sensor_id and args.all_axes:
        vertical, flexural, torsional = sensor_dict_combined[args.sensor_id]
        print(f"Sensor {args.sensor_id}:")
        print(f"  Vertical: {vertical}")
        print(f"  Flexural: {flexural}")
        print(f"  Torsional: {torsional}")
        return
    
    if args.sensor_id and args.direction:
        column = get_sensor_direction(args.sensor_id, args.direction)
        print(f"Sensor {args.sensor_id} {args.direction} axis: {column}")
        return
    
    if args.direction and args.all_sensors:
        sensors = get_all_sensors_for_direction(args.direction)
        print(f"All {args.direction} sensors:")
        for sid, column in sensors.items():
            print(f"  Sensor {sid}: {column}")
        return
    
    parser.print_help()

if __name__ == "__main__":
    main()
