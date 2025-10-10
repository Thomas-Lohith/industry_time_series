import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import re
import argparse

def extract_time_from_datetime(datetime_str):
    """Extract start time from the datetime string"""
    match = re.search(r'(\d{1,2}:\d{2})\s*-\s*\d{1,2}:\d{2}', str(datetime_str))
    if match:
        return match.group(1)
    return None

def extract_date_from_datetime(datetime_str):
    """Extract date from the datetime string"""
    match = re.search(r'(\d{2}/\d{2}/\d{4})', str(datetime_str))
    if match:
        return match.group(1)
    return None

def time_to_minutes(time_str):
    """Convert time string (HH:MM) to minutes from midnight"""
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except:
        return None

def calculate_time_based_moving_average(df, time_window_minutes=60):
    """
    Calculate moving average based on time window
    time_window_minutes: the time period for moving average (default 60 minutes = 1 hour)
    """
    # Convert time to minutes from midnight
    df['TimeInMinutes'] = df['Time'].apply(time_to_minutes)
    
    # Calculate moving average based on time window
    # Since data is in 15-minute intervals, calculate how many intervals fit in the window
    intervals_in_window = time_window_minutes / 15
    window_size = max(1, int(intervals_in_window))
    
    # Calculate rolling average
    df['MovingAverage'] = df['VehicleCount'].rolling(
        window=window_size, 
        center=True, 
        min_periods=1
    ).mean()
    
    return df, window_size

def load_and_process_data(file_path, sheet_name=0):
    """Load Excel data and process it"""
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    
    # Get the first two columns regardless of their names
    df = df.iloc[:, :2]
    df.columns = ['DateTime', 'VehicleCount']
    
    # Extract date and time
    df['Date'] = df['DateTime'].apply(extract_date_from_datetime)
    df['Time'] = df['DateTime'].apply(extract_time_from_datetime)
    
    # Remove rows with missing data
    df = df.dropna(subset=['Date', 'Time', 'VehicleCount'])
    
    # Convert VehicleCount to numeric
    df['VehicleCount'] = pd.to_numeric(df['VehicleCount'], errors='coerce')
    df = df.dropna(subset=['VehicleCount'])
    
    return df

def plot_moving_average_for_day(df, target_date, time_window_minutes=60, save_path=None):
    """
    Plot moving average for a specific day
    
    Parameters:
    - df: DataFrame with transit data
    - target_date: date to plot (format: 'DD/MM/YYYY')
    - time_window_minutes: time period for moving average in minutes (default 60)
    - save_path: path to save the plot (optional)
    """
    # Filter data for the target date
    day_data = df[df['Date'] == target_date].copy()
    
    if len(day_data) == 0:
        print(f"No data found for date: {target_date}")
        return
    
    # Reset index for proper plotting
    day_data = day_data.reset_index(drop=True)
    
    # Calculate time-based moving average
    day_data, window_size = calculate_time_based_moving_average(
        day_data, 
        time_window_minutes
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Plot original data
    ax.plot(day_data.index, day_data['VehicleCount'], 
            label='Original Data (15-min intervals)', 
            color='#87CEEB', linewidth=1.5, marker='o', markersize=4, alpha=0.7)
    
    # Plot moving average
    ax.plot(day_data.index, day_data['MovingAverage'], 
            label=f'Moving Average ({time_window_minutes} minutes window)', 
            color='#FF6347', linewidth=3, marker='s', markersize=5)
    
    # Customize the plot
    ax.set_xlabel('Time of Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Vehicles (per 15 minutes)', fontsize=13, fontweight='bold')
    ax.set_title(f'Vehicle Transit Moving Average Analysis\nDate: {target_date} | Time Window: {time_window_minutes} minutes', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    
    # Set x-axis labels to show times
    step = max(1, len(day_data) // 20)  # Show approximately 20 labels
    ax.set_xticks(day_data.index[::step])
    ax.set_xticklabels(day_data['Time'].iloc[::step], rotation=45, ha='right', fontsize=10)
    
    # Add statistics box
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {day_data["VehicleCount"].mean():.1f}\n'
    stats_text += f'Max: {day_data["VehicleCount"].max():.0f}\n'
    stats_text += f'Min: {day_data["VehicleCount"].min():.0f}\n'
    stats_text += f'Std Dev: {day_data["VehicleCount"].std():.1f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY FOR {target_date}")
    print(f"{'='*60}")
    print(f"Time Window for Moving Average: {time_window_minutes} minutes ({window_size} intervals)")
    print(f"Total 15-minute intervals: {len(day_data)}")
    print(f"Time range: {day_data['Time'].iloc[0]} to {day_data['Time'].iloc[-1]}")
    print(f"\nVehicle Count Statistics:")
    print(f"  Average vehicles per 15-min: {day_data['VehicleCount'].mean():.2f}")
    print(f"  Maximum vehicles: {day_data['VehicleCount'].max():.0f} (at {day_data.loc[day_data['VehicleCount'].idxmax(), 'Time']})")
    print(f"  Minimum vehicles: {day_data['VehicleCount'].min():.0f} (at {day_data.loc[day_data['VehicleCount'].idxmin(), 'Time']})")
    print(f"  Standard deviation: {day_data['VehicleCount'].std():.2f}")
    print(f"  Total vehicles for the day: {day_data['VehicleCount'].sum():.0f}")
    print(f"{'='*60}\n")
    
    return day_data

def plot_multiple_time_windows(df, target_date, time_windows=[30, 60, 120], save_path=None):
    """
    Plot moving averages with different time windows for comparison
    
    Parameters:
    - df: DataFrame with transit data
    - target_date: date to plot
    - time_windows: list of time windows in minutes to compare
    - save_path: path to save the plot
    """
    # Filter data for the target date
    day_data = df[df['Date'] == target_date].copy().reset_index(drop=True)
    
    if len(day_data) == 0:
        print(f"No data found for date: {target_date}")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot original data
    ax.plot(day_data.index, day_data['VehicleCount'], 
            label='Original Data', color='lightgray', 
            linewidth=1, marker='o', markersize=3, alpha=0.5)
    
    # Define colors for different windows
    colors = ['#FF6347', '#4169E1', '#32CD32', '#FF8C00', '#9370DB']
    
    # Plot moving averages for different time windows
    for idx, time_window in enumerate(time_windows):
        temp_data = day_data.copy()
        temp_data, _ = calculate_time_based_moving_average(temp_data, time_window)
        
        ax.plot(temp_data.index, temp_data['MovingAverage'], 
                label=f'{time_window} min MA', 
                color=colors[idx % len(colors)], 
                linewidth=2.5, marker='s', markersize=4)
    
    # Customize the plot
    ax.set_xlabel('Time of Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Vehicles', fontsize=13, fontweight='bold')
    ax.set_title(f'Comparison of Moving Average Time Windows\nDate: {target_date}', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis labels
    step = max(1, len(day_data) // 20)
    ax.set_xticks(day_data.index[::step])
    ax.set_xticklabels(day_data['Time'].iloc[::step], rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="moving average plots for a given day")
    parser.add_argument('--path', type=str, required=True, help="Path to file")
    parser.add_argument('--date', type=str, required=True, help="target date")
    args = parser.parse_args()

    file_path = args.path
    target_date = args.date

    """Main function to run the analysis"""
    print("="*60)
    print("VEHICLE TRANSIT MOVING AVERAGE ANALYSIS")
    print("="*60)
    
    # File path - UPDATE THIS WITH YOUR EXCEL FILE PATH
    #file_path = '/Users/thomas/Data/7_AID webcam data/Marzo/transiti.xlsx'  # Change this to your file path
    
    # Load and process data
    print("\nLoading data...")
    try:
        df = load_and_process_data(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        print("\nPlease ensure:")
        print("  1. The file path is correct")
        print("  2. The file exists")
        print("  3. The file has the correct format (2 columns)")
        return
    
    print(f"✓ Data loaded successfully!")
    print(f"\n Dataset Summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nAvailable dates:")
    for date in sorted(df['Date'].unique()):
        count = len(df[df['Date'] == date])
        print(f"    {date}: {count} records")
    
    # ========================
    # CONFIGURATION PARAMETERS
    # ========================
    
    # Set your target date here
    #target_date = '01/03/2025'  # Change to your desired date
    
    # Set your time window for moving average (in minutes)
    time_window_minutes = 60  # Change this value (e.g., 30, 45, 60, 90, 120)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS CONFIGURATION")
    print(f"{'='*60}")
    
    print(f"Target Date: {target_date}")
    print(f"Moving Average Time Window: {time_window_minutes} minutes")
    print(f"{'='*60}")
    
    # Plot for specific date with specific time window
    print(f"\n Generating moving average plot...")
    plot_moving_average_for_day(
        df, 
        target_date, 
        time_window_minutes=time_window_minutes,
        save_path=f'graphs/transit_ma_{target_date.replace("/", "_")}_{time_window_minutes}min.png'
    )
    
    # Plot comparison with multiple time windows
    print(f"\nGenerating comparison plot with multiple time windows...")
    plot_multiple_time_windows(
        df,
        target_date,
        time_windows=[30, 60, 120],  # Compare 30min, 60min, and 120min windows
        save_path=f'graphs/transit_ma_comparison_{target_date.replace("/", "_")}.png'
    )
    
    # Interactive mode
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print(f"{'='*60}")
    user_input = input("\nWould you like to analyze another date? (y/n): ").strip().lower()
    
    if user_input == 'y':
        user_date = input("Enter date (DD/MM/YYYY): ").strip()
        user_window = input("Enter time window in minutes (e.g., 30, 60, 120): ").strip()
        
        try:
            time_window = int(user_window)
            plot_moving_average_for_day(df, user_date, time_window_minutes=time_window)
        except ValueError:
            print("Invalid time window. Using default 60 minutes.")
            plot_moving_average_for_day(df, user_date, time_window_minutes=60)
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()