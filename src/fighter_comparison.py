import pandas as pd
import time
from scrape_fighters import get_all_fighter_urls
from scrape_basic_stats import get_fighter_basic_stats
from scrape_fight_urls import get_fight_urls
from scrape_fight_dates import fetch_webpage, extract_fight_dates_and_results
from scrape_fight_round_details import fight_details


def fighter_stats(fighter_url):
    """
    Fetches and compiles fight statistics for a given fighter URL.

    Args:
        fighter_url (str): The URL of the fighter's profile page.

    Returns:
        pd.DataFrame: A DataFrame containing fight statistics for the fighter.
    """
    all_fighter_stats = []
    all_fight_dates = []
    combined_fighter_round_df = pd.DataFrame()

    try:
        stats = get_fighter_basic_stats(fighter_url)
        all_fighter_stats.append(stats)

        html_content = fetch_webpage(fighter_url)
        if html_content:
            fight_dates = extract_fight_dates_and_results(html_content)
            for fight in fight_dates:
                fight['Name'] = stats['Name']  # Add fighter name to each fight date
            all_fight_dates.extend(fight_dates)

        fight_urls = get_fight_urls(fighter_url)
        fighter_round_df = fight_details(fight_urls, stats['Name'])

        combined_fighter_round_df = pd.concat([combined_fighter_round_df, fighter_round_df], ignore_index=True)
    except Exception as e:
        print(f"Failed to process data for fighter: {e}")

    # Create DataFrames from the collected data
    df_fighter_stats = pd.DataFrame(all_fighter_stats)
    df_fight_dates = pd.DataFrame(all_fight_dates)

    # Merge the DataFrames to get the final combined data
    final_combined_df = combined_fighter_round_df.merge(df_fighter_stats, on='Name', how='inner')
    final_combined_df = final_combined_df.merge(df_fight_dates, on=['Event', 'Name'], how='inner')

    return final_combined_df


def standardize_name(name):
    """
    Standardizes the fighter name to match the format used on the UFC stats website.

    Args:
        name (str): The name to standardize.

    Returns:
        str: The standardized name.
    """
    return ' '.join(word.capitalize() for word in name.split())


def fetch_specific_fighter_data(fighter_names):
    """
    Fetches fight data for specific fighters by their names and compiles it into a single DataFrame.

    Args:
        fighter_names (list): A list of fighter names to process.

    Returns:
        pd.DataFrame: A DataFrame containing fight data for the specified fighters.
    """
    base_url = 'http://ufcstats.com/statistics/fighters'
    all_fighter_urls = get_all_fighter_urls(base_url)
    combined_data = pd.DataFrame()

    for fighter_name in fighter_names:
        # First attempt with the standardized name
        standardized_name = standardize_name(fighter_name)
        fighter_url = all_fighter_urls.get(standardized_name)

        # If not found, try the original user input name
        if not fighter_url:
            print(f"Standardized name '{standardized_name}' not found. Trying original input '{fighter_name}'.")
            fighter_url = all_fighter_urls.get(fighter_name)

        # If still not found, skip to the next fighter
        if not fighter_url:
            print(f"URL for '{fighter_name}' not found even with the original input. Skipping...")
            continue

        print(f"Fetching data for {fighter_name} using URL: {fighter_url}")
        try:
            fighter_data = fighter_stats(fighter_url)
            combined_data = pd.concat([combined_data, fighter_data], ignore_index=True)
        except Exception as e:
            print(f"Failed to fetch data for {fighter_name}: {e}")
        time.sleep(1)  

    return combined_data



if __name__ == "__main__":
    fighter_1 = input("Enter Fighter 1: ").strip()
    fighter_2 = input("Enter Fighter 2: ").strip()
    fighter_names = [fighter_1, fighter_2]
    combined_fighter_data = fetch_specific_fighter_data(fighter_names)
    combined_fighter_data.to_csv('../data/specific_fighter_data.csv', index=False)
    print("Data collection complete and saved to 'specific_fighter_data.csv'")
