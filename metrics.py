def calculate_production_metrics(total_time, downtime, idle_time, downward_motion_time, pressing_time, cutting_time, steam_releasing_time, recooking_of_the_bottom_time, upwards_motion_time, total_count, good_count):
    # Calculate cycle time as the sum of the six times
    cycle_time = (downward_motion_time + pressing_time + cutting_time + steam_releasing_time + recooking_of_the_bottom_time + upwards_motion_time)
    
    # Calculate uptime
    uptime = total_time - downtime - idle_time
    
    # Calculate availability
    availability = uptime / total_time
    
    # Calculate performance
    run_time = uptime  # Assuming run_time is equivalent to uptime
    performance = (cycle_time * total_count) / run_time
    
    # Calculate quality
    quality = good_count / total_count
    
    # Calculate OEE
    OEE = availability * performance * quality
    
    return {
        "Uptime": uptime,
        "Downtime": downtime,
        "Idle Time": idle_time,
        "Cycle Time": cycle_time,
        "Total Time": total_time,
        "Availability": availability,
        "Performance": performance,
        "Quality": quality,
        "OEE": OEE
    }

# Example usage
total_time = 480  # Total time in minutes
downtime = 60     # Downtime in minutes
idle_time = 30    # Idle time in minutes

# Define the six times in minutes
downward_motion_time = 0.5
pressing_time = 1.5
cutting_time = 1
steam_releasing_time = 0.7
recooking_of_the_bottom_time = 1.2
upwards_motion_time = 0.6

total_count = 1000    # Total number of units produced
good_count = 950      # Total number of good units produced

metrics = calculate_production_metrics(total_time, downtime, idle_time, downward_motion_time, pressing_time, cutting_time, steam_releasing_time, recooking_of_the_bottom_time, upwards_motion_time, total_count, good_count)
for key, value in metrics.items():
    print(f"{key}: {value:.2f}")


