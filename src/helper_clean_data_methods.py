import pandas as pd

def extract_strike_data(row, column_name):
    """
    Extracts landed and thrown strikes from a given column in a row.

    Args:
    row (pd.Series): The row from which to extract data.
    column_name (str): The column name from which to extract strike data.

    Returns:
    tuple: Landed and thrown strikes as integers.
    """
    if pd.notnull(row[column_name]) and 'of' in row[column_name]:
        landed, thrown = row[column_name].split(' of ')
        return int(landed), int(thrown)
    else:
        return 0, 0


def categorize_method(method):
    method = method.lower()
    if 'dec' in method:
        return 'dec'
    elif 'sub' in method:
        return 'sub'
    elif 'ko/tko' in method or 'knockout' in method or 'tko' in method:
        return 'ko'
    elif 'cnc' in method:
        return 'ko'  # CNC indicates a stoppage due to an inability to continue, which is treated as a TKO.
    elif 'dq' in method or 'disqualification' in method:
        return 'dq'
    elif 'overturned' in method:
        return 'overturned'
    else:
        return 'other'


def clean_weight_class(wc):
    weight_classes = [
        'strawweight', 'flyweight', 'bantamweight', 'featherweight', 'lightweight',
        'welterweight', 'middleweight', 'light heavyweight', 'heavyweight'
    ]

    for wc_class in weight_classes:
        if wc_class in wc:
            return wc_class.replace(' ', '_')  
    return 'other'  # Return 'other' for classes not in the predefined list


def extract_first_value(s):
    if pd.isna(s) or s == '--':
        return None
    return int(s.split()[0])


def calculate_cumulative_metrics(row, fighter_data):
    # Filter the fighter's historical data before the current fight date
    historical_fights = fighter_data[
        (fighter_data['name'] == row['name']) &
        (fighter_data['weight_class'] == row['weight_class']) &
        (pd.to_datetime(fighter_data['date']) <= pd.to_datetime(row['date']))
        ]

    # Calculate cumulative metricsl
    cumulative_metrics = {
        'cumulative_knockdowns': historical_fights['knockdowns'].sum(),
        'cumulative_significant_strikes_landed': historical_fights['significant_strikes_landed'].sum(),
        'cumulative_significant_strikes_thrown': historical_fights['significant_strikes_thrown'].sum(),
        'cumulative_total_strikes_landed': historical_fights['total_strikes_landed'].sum(),
        'cumulative_total_strikes_thrown': historical_fights['total_strikes_thrown'].sum(),
        'cumulative_takedowns_landed': historical_fights['takedowns_landed'].sum(),
        'cumulative_takedowns_thrown': historical_fights['takedowns_thrown'].sum(),
        'cumulative_head_strikes_landed': historical_fights['head_strikes_landed'].sum(),
        'cumulative_head_strikes_thrown': historical_fights['head_strikes_thrown'].sum(),
        'cumulative_body_strikes_landed': historical_fights['body_strikes_landed'].sum(),
        'cumulative_body_strikes_thrown': historical_fights['body_strikes_thrown'].sum(),
        'cumulative_leg_strikes_landed': historical_fights['leg_strikes_landed'].sum(),
        'cumulative_leg_strikes_thrown': historical_fights['leg_strikes_thrown'].sum(),
        'cumulative_distance_strikes_landed': historical_fights['distance_strikes_landed'].sum(),
        'cumulative_distance_strikes_thrown': historical_fights['distance_strikes_thrown'].sum(),
        'cumulative_clinch_strikes_landed': historical_fights['clinch_strikes_landed'].sum(),
        'cumulative_clinch_strikes_thrown': historical_fights['clinch_strikes_thrown'].sum(),
        'cumulative_ground_strikes_landed': historical_fights['ground_strikes_landed'].sum(),
        'cumulative_ground_strikes_thrown': historical_fights['ground_strikes_thrown'].sum(),
        'cumulative_title_fights': historical_fights['total_title_fights'].sum(),
        'cumulative_rounds': historical_fights['total_rounds'].sum(),
        'cumulative_unique_events': historical_fights['total_unique_events'].sum(),
        'cumulative_wins': historical_fights['wins'].sum(),
        'cumulative_dec': historical_fights['total_method_dec'].sum(),
        'cumulative_dq': historical_fights['total_method_dq'].sum(),
        'cumulative_ko': historical_fights['total_method_ko'].sum(),
        'cumulative_overturned': historical_fights['total_method_overturned'].sum(),
        'cumulative_sub': historical_fights['total_method_sub'].sum()
    }

    return pd.Series(cumulative_metrics)


# Function to get the most recent cumulative data prior to each fight
def get_most_recent_cumulative(row, cumulative_data):
    # Filter cumulative data for the same fighter and weight class
    relevant_data = cumulative_data[(cumulative_data['name'] == row['name']) &
                                    (cumulative_data['weight_class'] == row['weight_class']) &
                                    (cumulative_data['date'] < row['date'])]
    # Return the most recent cumulative data (last row)
    if not relevant_data.empty:
        return relevant_data.iloc[-1]
    else:
        # If no prior fights, return a row of NaNs as default, with a specified dtype
        return pd.Series(index=cumulative_data.columns, dtype='float64')


def one_hot_encode_fight_details(user_inputs, all_weight_classes):
    # Extract user inputs
    weight_class = user_inputs[0]
    is_male_fight = user_inputs[1]

    # Replace spaces with underscores in the weight class name
    weight_class = weight_class.lower().replace(' ', '_')

    # Initialize the one-hot encoded list with 0s
    encoded_fight_details = [0] * (len(all_weight_classes) + 1)

    # Set the gender indicator
    encoded_fight_details[0] = int(is_male_fight)

    # Find the index of the weight class in the all_weight_classes list
    weight_class_index = all_weight_classes.index(f'weight_class_{weight_class}')

    # Set the corresponding index for the weight class to 1
    encoded_fight_details[weight_class_index + 1] = 1

    return encoded_fight_details



def extract_round_number(round_str):
    try:
        return int(round_str.split()[1])
    except (IndexError, ValueError):
        return None
