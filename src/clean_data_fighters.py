from datetime import datetime

import pandas as pd
from helper_clean_data_methods import categorize_method, extract_strike_data, clean_weight_class, extract_first_value, \
    extract_round_number, calculate_cumulative_metrics, get_most_recent_cumulative, one_hot_encode_fight_details
import numpy as np


def process_fighter_attributes(ufc_data):
    """
    Processes and modifies the attributes of fighters such as the fighter's age, weight class, height, reach, weight.

    Parameters:
    df (pd.DataFrame): The DataFrame containing fighter data to be processed.

    Returns:
    pd.DataFrame: A DataFrame with processed fighter attributes to be used for further feature engineering.
    """

    # Rename columns for easier manipulation
    ufc_data.rename(columns={
        'Sig. Str.': 'significant_strikes',
        'Total Str.': 'total_strikes',
        'TD': 'takedowns',
        'TD %': 'takedown_percentage',
        'Sub. Att': 'submission_attempts',
        'Rev.': 'reversals',
        'Ctrl': 'control_time',
        'Head': 'head_strikes',
        'Body': 'body_strikes',
        'Leg': 'leg_strikes',
        'Distance': 'distance_strikes',
        'Clinch': 'clinch_strikes',
        'Ground': 'ground_strikes',
        'Method': 'method'
    }, inplace=True)

    ufc_data['method'] = ufc_data['method'].apply(categorize_method)

    # Columns containing strike data
    strikes_col = [
        'significant_strikes', 'total_strikes', 'takedowns',
        'head_strikes', 'body_strikes', 'leg_strikes',
        'distance_strikes', 'clinch_strikes', 'ground_strikes'
    ]

    # Apply the extraction function to each relevant column
    for col in strikes_col:
        ufc_data[f'{col}_landed'], ufc_data[f'{col}_thrown'] = zip(
            *ufc_data.apply(lambda row: extract_strike_data(row, col), axis=1))

    cols_drop = [
        'significant_strikes',
        'total_strikes',
        'takedowns',
        'takedown_percentage',
        'submission_attempts',
        'reversals',
        'control_time',
        'head_strikes',
        'body_strikes',
        'leg_strikes',
        'distance_strikes',
        'clinch_strikes',
        'ground_strikes',
        'Sig. Str. %'
    ]

    ufc_data.drop(columns=cols_drop, inplace=True)

    # Cleaning Date data
    # Convert DOB and Date to datetime

    ufc_data = ufc_data[ufc_data['DOB'] != '--'].copy()
    # Convert DOB and Date to datetime
    ufc_data['DOB'] = pd.to_datetime(ufc_data['DOB'], format='%b %d, %Y')
    ufc_data['Date'] = pd.to_datetime(ufc_data['Date'], format='%b. %d, %Y')

    # Calculate the age at the time of the fight
    ufc_data['fight_age'] = (ufc_data['Date'] - ufc_data['DOB']).dt.days // 365
    ufc_data['current_age'] = (datetime.now() - ufc_data['DOB']).dt.days // 365

    # Convert to lowercase
    ufc_data['Weight Class'] = ufc_data['Weight Class'].str.lower()

    # Flag title fights
    ufc_data['is_title_fight'] = ufc_data['Weight Class'].apply(lambda x: 'title' in x)

    # Flag male fights
    ufc_data['is_male_fight'] = ~ufc_data['Weight Class'].apply(lambda x: 'women' in x)

    ufc_data['weight_class'] = ufc_data['Weight Class'].apply(clean_weight_class)
    ufc_data = ufc_data[ufc_data['weight_class'] != 'other']

    # Drop the old 'Weight Class' column
    ufc_data = ufc_data.drop(columns=['Weight Class'])

    # Filter out rows where 'Reach' is '--'
    ufc_data = ufc_data[ufc_data['Reach'] != '--'].reset_index(drop=True)

    # Apply the function to create new columns
    ufc_data['height_inches'] = ufc_data['Height'].apply(extract_first_value)
    ufc_data['weight_pounds'] = ufc_data['Weight'].apply(extract_first_value)
    ufc_data['reach_inches'] = ufc_data['Reach'].apply(extract_first_value)

    # Drop the original columns if no longer needed
    ufc_data.drop(columns=['Height', 'Weight', 'Reach'], inplace=True)

    ufc_data['round_number'] = ufc_data['Round'].apply(extract_round_number)
    ufc_data.drop(columns=['Round'], inplace=True)

    # Remove any results that were really old and fighters are no longer active

    ufc_data['Date'] = pd.to_datetime(ufc_data['Date'], format='%b. %d, %Y')

    # Rename columns for consistency
    ufc_data.rename(columns={
        'Event': 'event',
        'Name': 'name',
        'KD': 'knockdowns',
        'Wins': 'wins',
        'Losses': 'losses',
        'Draws': 'draws',
        'No Contests': 'nc',
        'Stance': 'stance',
        'Date': 'date',
        'Result': 'result',
        'Method': 'method',
    }, inplace=True)

    return ufc_data


def engineer_fight_stats(ufc_data, user_input):
    """
    Performs feature engineering on fight statistics, including aggregating and calculating
    cumulative fight performance metrics. This includes metrics like significant strikes,
    takedowns, and win ratios, as well as preparing the data for further analysis.

    Parameters:
    ufc_data (pd.DataFrame): The DataFrame containing raw UFC fight data.
    user_input (dict): A dictionary containing user-provided information about the fight,
                       including weight class.

    Returns:
    pd.DataFrame: A DataFrame with engineered features, including cumulative fight metrics
                  and win ratios.
    """
    ufc_fight_data = ufc_data.groupby(
        ['event', 'name', 'wins', 'losses', 'draws', 'nc', 'stance', 'DOB', 'date', 'result', 'method'
         # method
            , 'Fighter_1', 'Fighter_2', 'current_age', 'fight_age', 'is_title_fight', 'is_male_fight', 'weight_class',
         'height_inches', 'weight_pounds', 'reach_inches'
         ]).agg({
        'knockdowns': 'sum',
        'significant_strikes_landed': 'sum',
        'significant_strikes_thrown': 'sum',
        'total_strikes_landed': 'sum',
        'total_strikes_thrown': 'sum',
        'takedowns_landed': 'sum',
        'takedowns_thrown': 'sum',
        'head_strikes_landed': 'sum',
        'head_strikes_thrown': 'sum',
        'body_strikes_landed': 'sum',
        'body_strikes_thrown': 'sum',
        'leg_strikes_landed': 'sum',
        'leg_strikes_thrown': 'sum',
        'distance_strikes_landed': 'sum',
        'distance_strikes_thrown': 'sum',
        'clinch_strikes_landed': 'sum',
        'clinch_strikes_thrown': 'sum',
        'ground_strikes_landed': 'sum',
        'ground_strikes_thrown': 'sum',
        'round_number': 'max'
    }).reset_index()

    # One-hot encode the 'method' column

    method_dummies = pd.get_dummies(ufc_fight_data['method'], prefix='method')
    required_methods = ['method_ko', 'method_sub', 'method_dec', 'method_dq', 'method_overturned']
    for method in required_methods:
        if method not in method_dummies.columns:
            method_dummies[method] = 0
    method_dummies = method_dummies[required_methods]

    ufc_fight_data = ufc_fight_data.join(method_dummies)

    # Aggregating Data with methods
    aggregated_fighter_data = ufc_fight_data.groupby(['name', 'weight_class', 'date']).agg(
        knockdowns=('knockdowns', 'sum'),
        significant_strikes_landed=('significant_strikes_landed', 'sum'),
        significant_strikes_thrown=('significant_strikes_thrown', 'sum'),
        total_strikes_landed=('total_strikes_landed', 'sum'),
        total_strikes_thrown=('total_strikes_thrown', 'sum'),
        takedowns_landed=('takedowns_landed', 'sum'),
        takedowns_thrown=('takedowns_thrown', 'sum'),
        head_strikes_landed=('head_strikes_landed', 'sum'),
        head_strikes_thrown=('head_strikes_thrown', 'sum'),
        body_strikes_landed=('body_strikes_landed', 'sum'),
        body_strikes_thrown=('body_strikes_thrown', 'sum'),
        leg_strikes_landed=('leg_strikes_landed', 'sum'),
        leg_strikes_thrown=('leg_strikes_thrown', 'sum'),
        distance_strikes_landed=('distance_strikes_landed', 'sum'),
        distance_strikes_thrown=('distance_strikes_thrown', 'sum'),
        clinch_strikes_landed=('clinch_strikes_landed', 'sum'),
        clinch_strikes_thrown=('clinch_strikes_thrown', 'sum'),
        ground_strikes_landed=('ground_strikes_landed', 'sum'),
        ground_strikes_thrown=('ground_strikes_thrown', 'sum'),
        total_title_fights=('is_title_fight', 'sum'),
        wins=('result', lambda x: (x == 'win').sum()),
        total_rounds=('round_number', 'sum'),
        total_unique_events=('event', 'nunique'),
        **{f'total_{col}': (col, 'sum') for col in method_dummies.columns}
    ).reset_index()

    # Apply the function to each row
    cumulative_data = aggregated_fighter_data.apply(calculate_cumulative_metrics, axis=1,
                                                    fighter_data=aggregated_fighter_data)
    final_data_with_cumulative = pd.concat([aggregated_fighter_data, cumulative_data], axis=1)
    # Combine the cumulative data with the original data

    # Define the columns to keep
    columns_to_keep = [
        'name', 'weight_class', 'date',
        'cumulative_knockdowns', 'cumulative_significant_strikes_landed', 'cumulative_significant_strikes_thrown',
        'cumulative_total_strikes_landed', 'cumulative_total_strikes_thrown', 'cumulative_takedowns_landed',
        'cumulative_takedowns_thrown', 'cumulative_head_strikes_landed', 'cumulative_head_strikes_thrown',
        'cumulative_body_strikes_landed', 'cumulative_body_strikes_thrown', 'cumulative_leg_strikes_landed',
        'cumulative_leg_strikes_thrown', 'cumulative_distance_strikes_landed', 'cumulative_distance_strikes_thrown',
        'cumulative_clinch_strikes_landed', 'cumulative_clinch_strikes_thrown', 'cumulative_ground_strikes_landed',
        'cumulative_ground_strikes_thrown', 'cumulative_title_fights', 'cumulative_rounds', 'cumulative_unique_events',
        'cumulative_wins', 'cumulative_dec', 'cumulative_dq', 'cumulative_ko', 'cumulative_overturned', 'cumulative_sub'
    ]

    cumulative_columns = [
        'cumulative_knockdowns', 'cumulative_significant_strikes_landed', 'cumulative_significant_strikes_thrown',
        'cumulative_total_strikes_landed', 'cumulative_total_strikes_thrown', 'cumulative_takedowns_landed',
        'cumulative_takedowns_thrown', 'cumulative_head_strikes_landed', 'cumulative_head_strikes_thrown',
        'cumulative_body_strikes_landed', 'cumulative_body_strikes_thrown', 'cumulative_leg_strikes_landed',
        'cumulative_leg_strikes_thrown', 'cumulative_distance_strikes_landed', 'cumulative_distance_strikes_thrown',
        'cumulative_clinch_strikes_landed', 'cumulative_clinch_strikes_thrown', 'cumulative_ground_strikes_landed',
        'cumulative_ground_strikes_thrown', 'cumulative_title_fights', 'cumulative_rounds', 'cumulative_unique_events',
        'cumulative_wins', 'cumulative_dec', 'cumulative_dq', 'cumulative_ko', 'cumulative_overturned', 'cumulative_sub'
    ]

    for col in cumulative_columns:
        final_data_with_cumulative[col] = pd.to_numeric(final_data_with_cumulative[col], errors='coerce').fillna(0)

    # Select only the columns to keep
    final_data_with_cumulative = final_data_with_cumulative[columns_to_keep]

    # Strike Accuracy
    final_data_with_cumulative['strike_accuracy'] = np.where(
        final_data_with_cumulative['cumulative_total_strikes_thrown'] > 0,
        final_data_with_cumulative['cumulative_total_strikes_landed'] / final_data_with_cumulative[
            'cumulative_total_strikes_thrown'],
        0
    )

    final_data_with_cumulative['sig_strike_accuracy'] = np.where(
        final_data_with_cumulative['cumulative_significant_strikes_thrown'] > 0,
        final_data_with_cumulative['cumulative_significant_strikes_landed'] / final_data_with_cumulative[
            'cumulative_significant_strikes_thrown'],
        0
    )
    # Takedown Accuracy
    final_data_with_cumulative['takedown_accuracy'] = np.where(
        final_data_with_cumulative['cumulative_takedowns_thrown'] > 0,
        final_data_with_cumulative['cumulative_takedowns_landed'] / final_data_with_cumulative[
            'cumulative_takedowns_thrown'],
        0
    )
    # Strike Ratios
    final_data_with_cumulative['head_strike_ratio'] = np.where(
        final_data_with_cumulative['cumulative_total_strikes_landed'] > 0,
        final_data_with_cumulative['cumulative_head_strikes_landed'] / final_data_with_cumulative[
            'cumulative_total_strikes_landed'],
        0
    )
    final_data_with_cumulative['body_strike_ratio'] = np.where(
        final_data_with_cumulative['cumulative_total_strikes_landed'] > 0,
        final_data_with_cumulative['cumulative_body_strikes_landed'] / final_data_with_cumulative[
            'cumulative_total_strikes_landed'],
        0
    )
    final_data_with_cumulative['leg_strike_ratio'] = np.where(
        final_data_with_cumulative['cumulative_total_strikes_landed'] > 0,
        final_data_with_cumulative['cumulative_leg_strikes_landed'] / final_data_with_cumulative[
            'cumulative_total_strikes_landed'],
        0
    )

    # win ratio
    final_data_with_cumulative['fight_duration'] = np.where(
        final_data_with_cumulative['cumulative_unique_events'] > 0,
        final_data_with_cumulative['cumulative_rounds'] / final_data_with_cumulative['cumulative_unique_events'],
        0
    )

    # average fight rounds
    final_data_with_cumulative['win_rate'] = np.where(
        final_data_with_cumulative['cumulative_unique_events'] > 0,
        final_data_with_cumulative['cumulative_wins'] / final_data_with_cumulative['cumulative_unique_events'],
        0
    )

    # knockdown pct
    final_data_with_cumulative['knockdown_percentage'] = np.where(
        final_data_with_cumulative['cumulative_total_strikes_landed'] > 0,
        final_data_with_cumulative['cumulative_knockdowns'] / final_data_with_cumulative[
            'cumulative_total_strikes_landed'],
        0
    )

    # knock out rate
    final_data_with_cumulative['ko_rate'] = np.where(
        final_data_with_cumulative['cumulative_unique_events'] > 0,
        final_data_with_cumulative['cumulative_ko'] / final_data_with_cumulative['cumulative_unique_events'],
        0
    )

    # submission rate
    final_data_with_cumulative['submission_rate'] = np.where(
        final_data_with_cumulative['cumulative_unique_events'] > 0,
        final_data_with_cumulative['cumulative_ko'] / final_data_with_cumulative['cumulative_unique_events'],
        0
    )

    # finish rate
    final_data_with_cumulative['finish_rate'] = np.where(
        final_data_with_cumulative['cumulative_unique_events'] > 0,
        (final_data_with_cumulative['cumulative_sub'] + final_data_with_cumulative['cumulative_ko']) /
        final_data_with_cumulative['cumulative_unique_events'],
        0
    )

    final_data_with_cumulative = final_data_with_cumulative.sort_values(by=['name', 'weight_class', 'date'])

    # Create the next_fight_date column
    final_data_with_cumulative['next_fight_date'] = final_data_with_cumulative.groupby(['name', 'weight_class'])[
        'date'].shift(-1)

    cols_drop = [
        'cumulative_knockdowns', 'cumulative_significant_strikes_landed', 'cumulative_significant_strikes_thrown',
        'cumulative_total_strikes_landed', 'cumulative_total_strikes_thrown', 'cumulative_takedowns_landed',
        'cumulative_takedowns_thrown', 'cumulative_head_strikes_landed', 'cumulative_head_strikes_thrown',
        'cumulative_body_strikes_landed', 'cumulative_body_strikes_thrown', 'cumulative_leg_strikes_landed',
        'cumulative_leg_strikes_thrown', 'cumulative_distance_strikes_landed', 'cumulative_distance_strikes_thrown',
        'cumulative_clinch_strikes_landed', 'cumulative_clinch_strikes_thrown', 'cumulative_ground_strikes_landed',
        'cumulative_ground_strikes_thrown', 'cumulative_title_fights', 'cumulative_rounds', 'cumulative_unique_events',
        'cumulative_wins', 'cumulative_dec', 'cumulative_dq', 'cumulative_ko', 'cumulative_overturned', 'cumulative_sub'
    ]
    final_data_with_cumulative.drop(columns=cols_drop, inplace=True)

    final_data_with_cumulative = final_data_with_cumulative.copy()
    ufc_fight_data = ufc_fight_data.copy()

    # Convert 'date' columns to datetime format
    final_data_with_cumulative['date'] = pd.to_datetime(final_data_with_cumulative['date'], errors='coerce')
    ufc_fight_data['date'] = pd.to_datetime(ufc_fight_data['date'], errors='coerce')

    # Sort the cumulative data by name, weight_class, and date
    final_data_with_cumulative = final_data_with_cumulative.sort_values(['name', 'weight_class', 'date'])

    # Apply the function to get the most recent cumulative stats for each fight event
    most_recent_cumulative = ufc_fight_data.apply(
        lambda row: get_most_recent_cumulative(row, final_data_with_cumulative), axis=1)
    most_recent_cumulative.dropna(inplace=True)

    # Reset index for both DataFrames to align for concatenation
    most_recent_cumulative = most_recent_cumulative.reset_index(drop=True)

    ufc_fight_data = ufc_fight_data.reset_index(drop=True)

    # Concatenate the most recent cumulative data to the original fight data
    ufc_fight_data_with_cumulative = pd.concat([ufc_fight_data, most_recent_cumulative], axis=1)

    # Merge the cumulative data with the UFC fight event data
    ufc_fight_data_with_cumulative = pd.merge(
        ufc_fight_data,
        final_data_with_cumulative,
        left_on=['name', 'weight_class', 'date'],
        right_on=['name', 'weight_class', 'next_fight_date'],
        how='left',
        suffixes=('', '_cumulative')
    )

    cols = ['next_fight_date', 'date_cumulative']
    ufc_fight_data_with_cumulative.drop(columns=cols, inplace=True)

    ufc_fight_data_with_cumulative['is_first_fight'] = ufc_fight_data_with_cumulative['sig_strike_accuracy'].isnull()
    ufc_fight_data_filtered = ufc_fight_data_with_cumulative.copy()
    ufc_fight_data_filtered = ufc_fight_data_filtered[ufc_fight_data_filtered['is_first_fight'] == False]
    ufc_fight_data_filtered.drop(columns='is_first_fight', inplace=True)
    ufc_fight_data_filtered.reset_index(drop=True, inplace=True)

    fighter_agg = ufc_fight_data_filtered.copy()

    columns_to_drop = ['event', 'DOB', 'date', 'result', 'method',
                       'Fighter_1', 'Fighter_2', 'round_number', 'method_ko',
                       'method_sub', 'method_dec', 'method_dq', 'method_overturned',
                       'draws', 'nc', 'stance',
                       'fight_age', 'weight_pounds', 'is_title_fight', 'is_male_fight']

    fighter_agg_cleaned = fighter_agg.drop(columns=columns_to_drop)

    fighter_agg_cleaned = fighter_agg_cleaned.groupby(['name', 'weight_class']).mean(numeric_only=True).reset_index()

    # Extract the required values from the user input dictionary
    user_input_weightclass = user_input.get("weight_class")
    user_input_weightclass = user_input_weightclass.lower().replace(' ', '_')

    fighter_agg_cleaned = fighter_agg_cleaned[fighter_agg_cleaned['weight_class'] == user_input_weightclass]

    return fighter_agg_cleaned


def filter_weight_class_data(df, user_input):
    """
    Filters the aggregated fighter data to only include fighters from the specified weight class
    provided by the user. This step ensures that only relevant data is considered in further analysis.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the raw data where we store fighter's attributes
    user_input (dict): A dictionary containing user-provided information about the fight,
                       including weight class.

    Returns:
    pd.DataFrame: A DataFrame filtered to include only the fighters from the specified weight class.
    """

    cols = ['Name', 'Stance', 'Date']
    df = df[cols]
    df.sort_values(by=['Name', 'Date'], ascending=[True, False])
    ufc_data_attr = df.groupby('Name').first()

    stance_dummies = pd.get_dummies(ufc_data_attr['Stance'], prefix='stance')
    required_stances = ['stance_Orthodox', 'stance_Southpaw', 'stance_Switch']
    for stance in required_stances:
        if stance not in stance_dummies.columns:
            stance_dummies[stance] = 0
    stance_dummies = stance_dummies[required_stances]
    ufc_data_attr = ufc_data_attr.join(stance_dummies)

    ufc_data_attr.reset_index(inplace=True)
    ufc_data_attr.drop(columns=['Stance', 'Date'], inplace=True)

    ufc_data_attr.rename(columns={
        'Name': 'name'
    }, inplace=True)

    user_input_weightclass = user_input.get("weight_class")
    user_input_male_fight = user_input.get("is_male_fight")

    user_inputs = [
        user_input_weightclass,
        user_input_male_fight,
    ]

    all_weight_classes = [
        'weight_class_featherweight', 'weight_class_flyweight',
        'weight_class_heavyweight', 'weight_class_light_heavyweight',
        'weight_class_lightweight', 'weight_class_middleweight',
        'weight_class_strawweight', 'weight_class_welterweight',
        'weight_class_bantamweight'
    ]

    encoded_fight_details = one_hot_encode_fight_details(user_inputs, all_weight_classes)
    encoded_columns = ['is_male_fight_True'] + all_weight_classes
    encoded_df = pd.DataFrame([encoded_fight_details], columns=encoded_columns)

    encoded_df_repeated = pd.concat([encoded_df] * len(ufc_data_attr), ignore_index=True)

    ufc_data_attr = pd.concat([ufc_data_attr.reset_index(drop=True), encoded_df_repeated], axis=1)

    for column in encoded_columns:
        if column in ufc_data_attr.columns:
            ufc_data_attr[column] = encoded_df_repeated[column]
        else:
            ufc_data_attr[column] = encoded_df_repeated[column]

    return ufc_data_attr


def prepare_fight_data_pairs(agg_data, attr_data, user_input):
    """
    Prepares and pairs the data of two fighters for comparison, calculating differences
    in their attributes and fight performance metrics. The resulting DataFrame is formatted
    and ready for use in a machine learning model to predict fight outcomes.

    Parameters:
    agg_data (pd.DataFrame): The DataFrame containing aggregated fight performance data.
    attr_data (pd.DataFrame): The DataFrame containing processed fighter attributes.
    user_input (dict): A dictionary containing user-provided information about the fight,
                       including weight class.

    Returns:
    pd.DataFrame: A DataFrame with paired fighter data, including calculated differences
                  in performance metrics and relevant features for model input.
    """
    final_comp_df = attr_data.merge(agg_data, how='inner', on='name')
    fighter_1 = user_input.get("fighter_1")
    fighter_2 = user_input.get("fighter_2")

    # Split data into two DataFrames based on the role of Fighter_1 and Fighter_2
    fighter_A_df = final_comp_df.copy()
    fighter_A_df = fighter_A_df[fighter_A_df['name'] == fighter_1]

    fighter_B_df = final_comp_df.copy()
    fighter_B_df = fighter_B_df[fighter_B_df['name'] == fighter_2]

    # Rename columns to identify as fighter_A and fighter_B
    fighter_A_df = fighter_A_df.add_prefix('fighter_A_')
    fighter_B_df = fighter_B_df.add_prefix('fighter_B_')

    fighter_A_df.reset_index(drop=True, inplace=True)
    fighter_B_df.reset_index(drop=True, inplace=True)

    # Concatenating the DataFrames side by side
    combined_fighters = pd.concat([fighter_A_df, fighter_B_df], axis=1)
    final_model_df = combined_fighters.copy()

    numerical_columns = [
        'wins', 'losses', 'current_age', 'height_inches', 'reach_inches', 'knockdowns',
        'significant_strikes_landed', 'significant_strikes_thrown', 'total_strikes_landed',
        'total_strikes_thrown', 'takedowns_landed', 'takedowns_thrown', 'head_strikes_landed',
        'head_strikes_thrown', 'body_strikes_landed', 'body_strikes_thrown', 'leg_strikes_landed',
        'leg_strikes_thrown', 'distance_strikes_landed', 'distance_strikes_thrown',
        'clinch_strikes_landed', 'clinch_strikes_thrown', 'ground_strikes_landed',
        'ground_strikes_thrown', 'strike_accuracy', 'sig_strike_accuracy', 'takedown_accuracy',
        'head_strike_ratio', 'body_strike_ratio', 'leg_strike_ratio', 'fight_duration', 'win_rate',
        'knockdown_percentage', 'ko_rate', 'submission_rate', 'finish_rate'
    ]

    # Handle missing columns and fill NaN values with 0
    for col in numerical_columns:
        fighter_A_col = f'fighter_A_{col}'
        fighter_B_col = f'fighter_B_{col}'

        # Check if the columns exist, if not, create them filled with 0
        if fighter_A_col not in final_model_df.columns:
            final_model_df[fighter_A_col] = 0
        if fighter_B_col not in final_model_df.columns:
            final_model_df[fighter_B_col] = 0

        # Fill NaN values with 0
        final_model_df[fighter_A_col].fillna(0, inplace=True)
        final_model_df[fighter_B_col].fillna(0, inplace=True)

    # Calculate the differences
    for col in numerical_columns:
        fighter_A_col = f'fighter_A_{col}'
        fighter_B_col = f'fighter_B_{col}'
        final_model_df[f'diff_{col}'] = final_model_df[fighter_A_col] - final_model_df[fighter_B_col]

    # Identify columns that are both in numerical_columns and start with fighter_A or fighter_B
    columns_to_diff = [col for col in numerical_columns if
                       f'fighter_A_{col}' in final_model_df.columns and f'fighter_B_{col}' in final_model_df.columns]

    # Calculate differences
    for col in columns_to_diff:
        final_model_df[f'diff_{col}'] = final_model_df[f'fighter_A_{col}'] - final_model_df[f'fighter_B_{col}']

    columns_to_drop = []
    for col in columns_to_diff:
        columns_to_drop.append(f'fighter_A_{col}')
        columns_to_drop.append(f'fighter_B_{col}')
        
    # Drop original fighter_A and fighter_B columns for numerical metrics
    final_model_df.drop(columns=columns_to_drop, inplace=True)
    final_model_df.drop(columns=['fighter_A_name', 'fighter_B_name'], inplace=True)

    drop_cols = [
        'fighter_A_weight_class',
        'fighter_B_is_male_fight_True',
        'fighter_B_weight_class_featherweight',
        'fighter_B_weight_class_flyweight',
        'fighter_B_weight_class_heavyweight',
        'fighter_B_weight_class_light_heavyweight',
        'fighter_B_weight_class_lightweight',
        'fighter_B_weight_class_middleweight',
        'fighter_B_weight_class_strawweight',
        'fighter_B_weight_class_welterweight',
        'fighter_B_weight_class'
    ]

    user_input_is_title_fight = user_input.get("is_title_fight")

    final_model_df.drop(columns=drop_cols, inplace=True)
    final_model_df['is_title_fight_True'] = 1 if user_input_is_title_fight else 0

    final_model_df.rename(columns={
        'fighter_A_is_male_fight_True': 'is_male_fight_True',
        'fighter_A_weight_class_featherweight': 'weight_class_featherweight',
        'fighter_A_weight_class_flyweight': 'weight_class_flyweight',
        'fighter_A_weight_class_heavyweight': 'weight_class_heavyweight',
        'fighter_A_weight_class_light_heavyweight': 'weight_class_light_heavyweight',
        'fighter_A_weight_class_lightweight': 'weight_class_lightweight',
        'fighter_A_weight_class_middleweight': 'weight_class_middleweight',
        'fighter_A_weight_class_strawweight': 'weight_class_strawweight',
        'fighter_A_weight_class_welterweight': 'weight_class_welterweight',
        'diff_current_age': 'diff_fight_age'
    }, inplace=True)

    desired_columns = [
        'fighter_A_stance_Orthodox',
        'fighter_A_stance_Southpaw',
        'fighter_A_stance_Switch',
        'fighter_B_stance_Orthodox',
        'fighter_B_stance_Southpaw',
        'fighter_B_stance_Switch',
        'is_title_fight_True',
        'is_male_fight_True',
        'weight_class_featherweight',
        'weight_class_flyweight',
        'weight_class_heavyweight',
        'weight_class_light_heavyweight',
        'weight_class_lightweight',
        'weight_class_middleweight',
        'weight_class_strawweight',
        'weight_class_welterweight',
        'diff_wins',
        'diff_losses',
        'diff_fight_age',
        'diff_height_inches',
        'diff_reach_inches',
        'diff_knockdowns',
        'diff_significant_strikes_landed',
        'diff_significant_strikes_thrown',
        'diff_total_strikes_landed',
        'diff_total_strikes_thrown',
        'diff_takedowns_landed',
        'diff_takedowns_thrown',
        'diff_head_strikes_landed',
        'diff_head_strikes_thrown',
        'diff_body_strikes_landed',
        'diff_body_strikes_thrown',
        'diff_leg_strikes_landed',
        'diff_leg_strikes_thrown',
        'diff_distance_strikes_landed',
        'diff_distance_strikes_thrown',
        'diff_clinch_strikes_landed',
        'diff_clinch_strikes_thrown',
        'diff_ground_strikes_landed',
        'diff_ground_strikes_thrown',
        'diff_strike_accuracy',
        'diff_sig_strike_accuracy',
        'diff_takedown_accuracy',
        'diff_head_strike_ratio',
        'diff_body_strike_ratio',
        'diff_leg_strike_ratio',
        'diff_fight_duration',
        'diff_win_rate',
        'diff_knockdown_percentage',
        'diff_ko_rate',
        'diff_submission_rate',
        'diff_finish_rate'
    ]

    final_model_df = final_model_df[desired_columns]
    return final_model_df
