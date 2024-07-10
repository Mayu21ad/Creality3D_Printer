from datetime import datetime, timedelta

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}hours:{int(minutes):02}minutes:{int(seconds):02}seconds"

def calculate_times(timestamps):
    
    cycle_times = []
    idle_times = []
    
    for i in range(1, len(timestamps) - 1, 2):
        prev_T2 = timestamps[i - 1]  
        curr_T1 = timestamps[i]      
        curr_T2 = timestamps[i + 1]  
        
        cycle_time = curr_T2 - prev_T2  
        idle_time = curr_T1 - prev_T2   

        cycle_times.append(cycle_time.total_seconds())
        idle_times.append(idle_time.total_seconds())
    
    total_cycle_time = sum(cycle_times)  
    total_idle_time = sum(idle_times)    

    return format_time(total_cycle_time), format_time(total_idle_time)

timestamps = [
    datetime(2024, 7, 9, 9, 15, 0),  
    datetime(2024, 7, 9, 9, 20, 0),  
    datetime(2024, 7, 9, 9, 35, 0),  
    datetime(2024, 7, 9, 9, 40, 0),  
    datetime(2024, 7, 9, 9, 55, 0),  
    datetime(2024, 7, 9, 10, 0, 0),  
    datetime(2024, 7, 9, 10, 15, 0)  

]

total_cycle_time, total_idle_time = calculate_times(timestamps)

print(f"Total Cycle Time: {total_cycle_time}")
print(f"Total Idle Time: {total_idle_time}")
