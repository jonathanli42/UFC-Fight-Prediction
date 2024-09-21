import pandas as pd
import fighter_comparison
from clean_data_fighters import process_fighter_attributes, engineer_fight_stats, filter_weight_class_data, \
    prepare_fight_data_pairs
from model_ufc_prediction import prediction_model


def extract_data(fighter_1, fighter_2):
    """
    Extract data for the two specified fighters.
    """
    print(f"Extracting data for fighters: {fighter_1} and {fighter_2}")
    fighter_names = [fighter_1, fighter_2]
    combined_fighter_data = fighter_comparison.fetch_specific_fighter_data(fighter_names)
    data = pd.DataFrame(combined_fighter_data)
    print("Data collection complete...")
    return data


def clean_fighter_data(raw_data):
    """
    Clean the raw data for a given fighter.
    """
    print("Cleaning raw data...")
    cleaned_data = process_fighter_attributes(raw_data)
    return cleaned_data


def process_and_filter_data(fighter_names, user_input):
    """
    Process and filter the data for the specified fighters.
    """
    # Extract the data
    raw_data = extract_data(fighter_names[0], fighter_names[1])

    # Filter the data based on weight class
    filtered_data = filter_weight_class_data(raw_data, user_input)

    return filtered_data


def merge_fighter_data(agg_data, attr_data, user_input):
    """
    Merge the data for the two fighters and prepare it for model input.
    """
    final_df = prepare_fight_data_pairs(agg_data, attr_data, user_input)
    return final_df


def engineer_fighter_stats(cleaned_data, weight_class):
    """
    Perform feature engineering on the cleaned data using the weight class.
    """
    engineered_data = engineer_fight_stats(cleaned_data, {"weight_class": weight_class})
    return engineered_data


def process_fighter_data(fights_data):
    """
    Process the data for all fights in the provided list.
    """
    all_fights_data = []

    for fight in fights_data:
        print(f"\nProcessing fight: {fight['fighter_1']} vs. {fight['fighter_2']}")

        # Extract and clean data
        raw_data = extract_data(fight['fighter_1'], fight['fighter_2'])
        cleaned_data = clean_fighter_data(raw_data)

        # Engineer fight stats
        weight_class = fight["weight_class"]
        engineered_data = engineer_fighter_stats(cleaned_data, weight_class)

        # Process and filter data
        filtered_data = process_and_filter_data([fight['fighter_1'], fight['fighter_2']],
                                                {"weight_class": fight["weight_class"],
                                                 "is_male_fight": fight["is_male_fight"]}
                                                )

        # Merge data and prepare for ML model
        final_df = prepare_fight_data_pairs(engineered_data, filtered_data, fight)

        # Use the ML model to predict the outcome
        prediction_result = ml_model(final_df)

        # Determine the predicted winner's name
        predicted_winner_name = fight['fighter_1'] if prediction_result['predicted_winner'] == 'fighter_1' else fight['fighter_2']

        # Append the prediction result to all_fights_data with the actual fighter name
        all_fights_data.append({
            "fight": f"{fight['fighter_1']} vs. {fight['fighter_2']}",
            "predicted_winner": predicted_winner_name,
            "fighter_A_pct_winning": prediction_result['fighter_A_pct_winning'],
            "fighter_B_pct_winning": prediction_result['fighter_B_pct_winning']
        })

    return all_fights_data


def ml_model(df):
    training_data = pd.read_csv('../data/cleaned_data_ml.csv')
    return prediction_model(training_data, df)


def get_fight_details():
    """
    Get fight details from the user.
    """
    fights_data = [
        {
            'fighter_1': 'Alex Pereira',
            'fighter_2': 'Jamahal Hill',
            'is_title_fight': True,
            'weight_class': 'Light Heavyweight',
            'is_male_fight': True
        },
        {
            'fighter_1': 'Ilia Topuria',
            'fighter_2': 'Max Holloway',
            'is_title_fight': True,
            'weight_class': 'Featherweight',
            'is_male_fight': True
        },
        {
            'fighter_1': 'Conor McGregor',
            'fighter_2': 'Michael Chandler',
            'is_title_fight': False,
            'weight_class': 'Lightweight',
            'is_male_fight': True
        },
        {
            'fighter_1': 'Alexa Grasso',
            'fighter_2': 'Valentina Shevchenko',
            'is_title_fight': True,
            'weight_class': 'Flyweight',
            'is_male_fight': False
        }
    ]
    return fights_data


if __name__ == "__main__":
    fights_data = get_fight_details()
    all_fights_data = process_fighter_data(fights_data)

    for fight_data in all_fights_data:
        print(fight_data)

