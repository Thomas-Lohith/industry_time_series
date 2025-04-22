import pandas as pd
import datetime
import argparse
def find_isolated_events(df, time_diff):
   
   df['time_diff'] = df['StartTimeStr'].diff().dt.total_seconds() / 60
   print(df['time_diff'].head(5))

   condition = ((df['time_diff']>=time_diff) & (df['time_diff']<= float(1000)))
   events = df[condition]
   print('the num of evemts with time _diff of  traffic is:', len(events))
   print('the events are following:\n', events)

   time_stamps = events[['StartTimeStr' , 'time_diff']]
   print(f'the events with time diff of {time_diff} mins:\n', time_stamps)

   with open('events_lesstraffic.txt', 'a') as e:
       e.writelines(f'\n events with less traffic :\n {str(time_stamps)}')

def main():
    parser = argparse.ArgumentParser('Findind the events with less traffic to analyse the intensity of vibrations')
    parser.add_argument('--path', type = str, required=True, help= 'path for the file')
    parser.add_argument('--time_difference', type= int, required=True, help= 'starting time frame interedted in')

    args = parser.parse_args()
    path = args.path
    time_diff = args.time_difference

    df = pd.read_excel(path)
    df['StartTimeStr'] = pd.to_datetime(df['StartTimeStr'], format='%d/%m/%Y %H:%M', errors="coerce", exact=False)
    df.sort_values(by='StartTimeStr')
    print(df['StartTimeStr'].head())

    find_isolated_events(df, time_diff)

if __name__ == "__main__":
    main()