
# current_mA = 2  # in milliamperes
# voltage = 240  # in Volts
# time_hours_per_day = 6  # total work hours per day

# labour_cost_min = 350  # minimum cost per day for one labour in rupees
# labour_cost_max = 450  # maximum cost per day for one labour in rupees
# number_of_labours = 17

# number_of_raw_materials = 1000
# cost_per_raw_material = 6  # cost per raw material in rupees

# products_per_day = 1200

# energy_consumption_per_6hrs_kwh = 2.88  # energy consumption for 6 hours in kWh

# # Constants
# hours_per_day = 24
# days_per_month = 30

# # Converting current from milliamperes to amperes
# current = current_mA / 1000  # in Amperes

# # Calculations
# # Power consumption (P = V * I)
# power_consumption_per_hour = voltage * current  # in Watts
# power_consumption_per_day = power_consumption_per_hour * time_hours_per_day  # in Watt-hours
# power_consumption_per_month = power_consumption_per_day * days_per_month  # in Watt-hours

# # Total raw material cost for daily production
# total_raw_material_cost_per_day = cost_per_raw_material * products_per_day

# # Average labour cost per day
# average_labour_cost_per_day = (labour_cost_min + labour_cost_max) / 2

# # Pressing cost of machine (labour cost + energy cost)
# energy_cost_per_day = (energy_consumption_per_6hrs_kwh * 1000 / 6) * time_hours_per_day * current * voltage / 1000  # converting Wh to kWh
# total_labour_cost_per_day = average_labour_cost_per_day * number_of_labours
# pressing_cost_per_day = total_labour_cost_per_day + energy_cost_per_day
# pressing_cost_per_hour = pressing_cost_per_day / time_hours_per_day
# pressing_cost_per_month = pressing_cost_per_day * days_per_month

# # Inspection cost
# inspection_cost_per_day = average_labour_cost_per_day * number_of_labours
# inspection_cost_per_hour = inspection_cost_per_day / time_hours_per_day
# inspection_cost_per_month = inspection_cost_per_day * days_per_month

# # Total daily cost
# total_daily_cost = total_raw_material_cost_per_day + pressing_cost_per_day + inspection_cost_per_day

# # Cost per unit
# cost_per_unit = total_daily_cost / products_per_day

# # Outputs
# print(f"Power Consumption (in Watt-hours):")
# print(f"Per Hour: {power_consumption_per_hour:.2f} Wh")
# print(f"Per Day: {power_consumption_per_day:.2f} Wh")
# print(f"Per Month: {power_consumption_per_month:.2f} Wh")

# print("\nRaw Material Cost:")
# print(f"Total Raw Material Cost per Day: ₹{total_raw_material_cost_per_day:.2f}")

# print("\nPressing Cost of Machine:")
# print(f"Per Hour: ₹{pressing_cost_per_hour:.2f}")
# print(f"Per Day: ₹{pressing_cost_per_day:.2f}")
# print(f"Per Month: ₹{pressing_cost_per_month:.2f}")

# print("\nInspection Cost:")
# print(f"Per Hour: ₹{inspection_cost_per_hour:.2f}")
# print(f"Per Day: ₹{inspection_cost_per_day:.2f}")
# print(f"Per Month: ₹{inspection_cost_per_month:.2f}")

# print("\nTotal Daily Cost: ₹{total_daily_cost:.2f}")
# print(f"Cost per Unit for {products_per_day} products per day: ₹{cost_per_unit:.2f}")


# import random
# import pandas as pd
# import time

# def generate_random_current():
#     return random.uniform(1, 2)  # Generates a random current between 1 mA and 2 mA

# def calculate_costs_realtime():
#     # User inputs
#     voltage = float(input("Enter voltage in Volts (V): "))
#     time_hours_per_day = 6  # Setting working hours to 6
#     total_hours = 8  # Total simulation hours
    
#     labour_cost_min = float(input("Enter minimum labour cost per day (in rupees): "))
#     labour_cost_max = float(input("Enter maximum labour cost per day (in rupees): "))
#     number_of_labours = int(input("Enter number of labours: "))

#     number_of_raw_materials = int(input("Enter number of raw materials: "))
#     cost_per_raw_material = float(input("Enter cost per raw material (in rupees): "))

#     products_per_day = int(input("Enter total products produced per day: "))

#     # Constants
#     days_per_month = 30

#     # Initialize variables for non-changing calculations
#     total_raw_material_cost_per_day = cost_per_raw_material * products_per_day
#     average_labour_cost_per_day = (labour_cost_min + labour_cost_max) / 2

#     # Data storage
#     data = {
#         "Time": [],
#         "Current (mA)": [],
#         "Power Consumption (Wh)": [],
#         "Total Raw Material Cost per Day (₹)": [],
#         "Pressing Cost per Hour (₹)": [],
#         "Pressing Cost per Day (₹)": [],
#         "Pressing Cost per Month (₹)": [],
#         "Inspection Cost per Hour (₹)": [],
#         "Inspection Cost per Day (₹)": [],
#         "Inspection Cost per Month (₹)": [],
#         "Total Daily Cost (₹)": [],
#         "Cost per Unit (₹)": []
#     }

#     # Simulate in real-time
#     current_hour = 1
#     while current_hour <= total_hours:
#         for minute in range(0, 60, 5):
#             # Generate random current for each 5 minutes
#             current_mA = generate_random_current()
#             current = current_mA / 1000  # Convert to Amperes

#             # Power consumption (P = V * I)
#             power_consumption_per_hour = voltage * current  # in Watts
#             power_consumption_per_day = power_consumption_per_hour * time_hours_per_day  # in Watt-hours
#             power_consumption_per_month = power_consumption_per_day * days_per_month  # in Watt-hours

#             # Energy consumption
#             energy_consumption_per_6hrs_kwh = power_consumption_per_hour * time_hours_per_day

#             # Pressing cost of machine (labour cost + energy cost)
#             energy_cost_per_day = energy_consumption_per_6hrs_kwh * (time_hours_per_day / 6)  # in kWh
#             total_labour_cost_per_day = average_labour_cost_per_day * number_of_labours
#             pressing_cost_per_day = total_labour_cost_per_day + energy_cost_per_day
#             pressing_cost_per_hour = pressing_cost_per_day / time_hours_per_day
#             pressing_cost_per_month = pressing_cost_per_day * days_per_month

#             # Inspection cost
#             inspection_cost_per_day = average_labour_cost_per_day * number_of_labours
#             inspection_cost_per_hour = inspection_cost_per_day / time_hours_per_day
#             inspection_cost_per_month = inspection_cost_per_day * days_per_month

#             # Total daily cost
#             total_daily_cost = total_raw_material_cost_per_day + pressing_cost_per_day + inspection_cost_per_day

#             # Cost per unit
#             cost_per_unit = total_daily_cost / products_per_day

#             # Storing data
#             time_label = f"{current_hour:02d}:{minute:02d}"
#             data["Time"].append(time_label)
#             data["Current (mA)"].append(current_mA)
#             data["Power Consumption (Wh)"].append(power_consumption_per_hour)
#             data["Total Raw Material Cost per Day (₹)"].append(total_raw_material_cost_per_day)
#             data["Pressing Cost per Hour (₹)"].append(pressing_cost_per_hour)
#             data["Pressing Cost per Day (₹)"].append(pressing_cost_per_day)
#             data["Pressing Cost per Month (₹)"].append(pressing_cost_per_month)
#             data["Inspection Cost per Hour (₹)"].append(inspection_cost_per_hour)
#             data["Inspection Cost per Day (₹)"].append(inspection_cost_per_day)
#             data["Inspection Cost per Month (₹)"].append(inspection_cost_per_month)
#             data["Total Daily Cost (₹)"].append(total_daily_cost)
#             data["Cost per Unit (₹)"].append(cost_per_unit)

#             # Convert to DataFrame and display
#             df = pd.DataFrame(data)
#             print("\nCurrent Data at", time_label, "\n")
#             print(df)
#             print("-" * 50)

#             # Sleep for 5 seconds to simulate real-time
#             time.sleep(5)

#         current_hour += 1

# # Run the function
# calculate_costs_realtime()


import random
import pandas as pd
import time

def generate_random_current():
    return random.uniform(1, 2)  # Generates a random current between 1 mA and 2 mA

def calculate_costs_realtime():
    # User inputs
    voltage = float(input("Enter voltage in Volts (V): "))
    time_hours_per_day = 6  # Setting working hours to 6
    total_hours = 8  # Total simulation hours
    
    labour_cost_min = float(input("Enter minimum labour cost per day (in rupees): "))
    labour_cost_max = float(input("Enter maximum labour cost per day (in rupees): "))
    number_of_labours = int(input("Enter number of labours: "))

    number_of_raw_materials = int(input("Enter number of raw materials: "))
    cost_per_raw_material = float(input("Enter cost per raw material (in rupees): "))

    products_per_day = int(input("Enter total products produced per day: "))

    # Constants
    days_per_month = 30

    # Initialize variables for non-changing calculations
    total_raw_material_cost_per_day = cost_per_raw_material * products_per_day
    average_labour_cost_per_day = (labour_cost_min + labour_cost_max) / 2

    # Data storage
    data = {
        "Time": [],
        "Current (mA)": [],
        "Power Consumption (Wh)": [],
        "Total Raw Material Cost per Day (₹)": [],
        "Pressing Cost per Hour (₹)": [],
        "Pressing Cost per Day (₹)": [],
        "Pressing Cost per Month (₹)": [],
        "Inspection Cost per Hour (₹)": [],
        "Inspection Cost per Day (₹)": [],
        "Inspection Cost per Month (₹)": [],
        "Total Daily Cost (₹)": [],
        "Cost per Unit (₹)": []
    }

    # Simulate in real-time
    current_hour = 1
    while current_hour <= total_hours:
        for second in range(0, 3600):  # 3600 seconds in an hour
            # Generate random current for each second
            current_mA = generate_random_current()
            current = current_mA / 1000  # Convert to Amperes

            # Power consumption (P = V * I)
            power_consumption_per_hour = voltage * current  # in Watts
            power_consumption_per_day = power_consumption_per_hour * time_hours_per_day  # in Watt-hours
            power_consumption_per_month = power_consumption_per_day * days_per_month  # in Watt-hours

            # Energy consumption
            energy_consumption_per_6hrs_kwh = power_consumption_per_hour * time_hours_per_day

            # Pressing cost of machine (labour cost + energy cost)
            energy_cost_per_day = energy_consumption_per_6hrs_kwh * (time_hours_per_day / 6)  # in kWh
            total_labour_cost_per_day = average_labour_cost_per_day * number_of_labours
            pressing_cost_per_day = total_labour_cost_per_day + energy_cost_per_day
            pressing_cost_per_hour = pressing_cost_per_day / time_hours_per_day
            pressing_cost_per_month = pressing_cost_per_day * days_per_month

            # Inspection cost
            inspection_cost_per_day = average_labour_cost_per_day * number_of_labours
            inspection_cost_per_hour = inspection_cost_per_day / time_hours_per_day
            inspection_cost_per_month = inspection_cost_per_day * days_per_month

            # Total daily cost
            total_daily_cost = total_raw_material_cost_per_day + pressing_cost_per_day + inspection_cost_per_day

            # Cost per unit
            cost_per_unit = total_daily_cost / products_per_day

            # Storing data
            time_label = f"{current_hour:02d}:{second // 60:02d}:{second % 60:02d}"
            data["Time"].append(time_label)
            data["Current (mA)"].append(current_mA)
            data["Power Consumption (Wh)"].append(power_consumption_per_hour)
            data["Total Raw Material Cost per Day (₹)"].append(total_raw_material_cost_per_day)
            data["Pressing Cost per Hour (₹)"].append(pressing_cost_per_hour)
            data["Pressing Cost per Day (₹)"].append(pressing_cost_per_day)
            data["Pressing Cost per Month (₹)"].append(pressing_cost_per_month)
            data["Inspection Cost per Hour (₹)"].append(inspection_cost_per_hour)
            data["Inspection Cost per Day (₹)"].append(inspection_cost_per_day)
            data["Inspection Cost per Month (₹)"].append(inspection_cost_per_month)
            data["Total Daily Cost (₹)"].append(total_daily_cost)
            data["Cost per Unit (₹)"].append(cost_per_unit)

            # Convert to DataFrame and display
            df = pd.DataFrame(data)
            print("\nCurrent Data at", time_label, "\n")
            print(df)
            print("-" * 50)

            # Sleep for 1 second to simulate real-time
            time.sleep(1)

        current_hour += 1

# Run the function
calculate_costs_realtime()
