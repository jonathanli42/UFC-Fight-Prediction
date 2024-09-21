import requests
from bs4 import BeautifulSoup
import pandas as pd


def clean_text(text):
    return ' '.join(text.split())


def fetch_webpage(url):
    """
    Fetches the webpage content for a given URL.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: The content of the webpage, or None if the request fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content.decode('utf-8')  # Decode bytes to string
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
        return None


def extract_max_round(html_content):
    """
    Extracts the maximum round number from the fight details HTML content.

    Args:
        html_content (str): The HTML content of the fight details page.

    Returns:
        int: The maximum round number found.

    Example:
        5
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    th_elements = soup.find_all('th', class_='b-fight-details__table-col')

    max_round = 0
    for th in th_elements:
        if 'Round' in th.text:
            round_name = th.text.strip()
            round_number = int(round_name.split(' ')[1])
            if round_number > max_round:
                max_round = round_number

    return max_round


def extract_event_name(html_content):
    """
    Extracts the event name from the fight details HTML content.

    Args:
        html_content (str): The HTML content of the fight details page.

    Returns:
        str: The name of the event, or 'Unknown Event' if not found.

    Example:
        "UFC 268: Usman vs. Covington 2"
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    event_title = soup.find('h2', class_='b-content__title')
    if event_title:
        event_name = clean_text(event_title.text)
        return event_name
    return 'Unknown Event'


def parse_fight_data(html_content, max_round):
    """
    Parses the fight data from the fight details HTML content.

    Args:
        html_content (str): The HTML content of the fight details page.
        max_round (int): The maximum round number in the fight.

    Returns:
        list: A list of dictionaries containing fight data for each round.

    Example:
        [{'Round': 'Round 1', 'Fighter': 'Fighter 1', ...}, {'Round': 'Round 1', 'Fighter': 'Fighter 2', ...}, ...]
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = soup.find_all('section', class_='b-fight-details__section js-fight-section')

    totals_section = None
    for section in sections:
        header = section.find('p', class_='b-fight-details__collapse-link_tot')
        if header and 'Totals' in header.text:
            totals_section = section
            break
    else:
        return None

    per_round_link = totals_section.find_next('a', class_='b-fight-details__collapse-link_rnd js-fight-collapse-link')
    if not per_round_link or 'Per round' not in per_round_link.text:
        return None

    per_round_section = per_round_link.find_next('table', class_='b-fight-details__table js-fight-table')
    if not per_round_section:
        return None

    round_data = []
    round_bodies = per_round_section.find_all('tbody')
    if len(round_bodies) == 0:
        return None

    round_counter = 1
    fighter_counter = 0

    for tbody in round_bodies:
        rows = tbody.find_all('tr', class_='b-fight-details__table-row')
        for row in rows:
            fighter_data = row.find_all('td', class_='b-fight-details__table-col')
            if len(fighter_data) > 0:
                fighter_name_tags = fighter_data[0].find_all('p')
                fighter_names = [tag.text.strip() for tag in fighter_name_tags]
                metrics = fighter_data[1:]

                for j, fighter_name in enumerate(fighter_names):
                    fighter_info = {
                        'Round': f'Round {round_counter}',
                        'Fighter': fighter_name,
                        'KD': metrics[0].find_all('p')[j].text.strip(),
                        'Sig. Str.': metrics[1].find_all('p')[j].text.strip(),
                        'Sig. Str. %': metrics[2].find_all('p')[j].text.strip(),
                        'Total Str.': metrics[3].find_all('p')[j].text.strip(),
                        'TD': metrics[4].find_all('p')[j].text.strip(),
                        'TD %': metrics[5].find_all('p')[j].text.strip(),
                        'Sub. Att': metrics[6].find_all('p')[j].text.strip(),
                        'Rev.': metrics[7].find_all('p')[j].text.strip(),
                        'Ctrl': metrics[8].find_all('p')[j].text.strip(),
                    }
                    round_data.append(fighter_info)

                    fighter_counter += 1
                    if fighter_counter == 2:
                        fighter_counter = 0
                        round_counter += 1
                        if round_counter > max_round:
                            round_counter = 1

    return round_data


def parse_significant_strikes(html_content, max_round):
    """
    Parses the significant strikes data from the fight details HTML content.

    Args:
        html_content (str): The HTML content of the fight details page.
        max_round (int): The maximum round number in the fight.

    Returns:
        list: A list of dictionaries containing significant strikes data for each round.

    Example:
        [{'Round': 'Round 1', 'Fighter': 'Fighter 1', ...}, {'Round': 'Round 1', 'Fighter': 'Fighter 2', ...}, ...]
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    sig_strikes_section = None
    sections = soup.find_all('section', class_='b-fight-details__section js-fight-section')
    for section in sections:
        header = section.find('p', class_='b-fight-details__collapse-link_tot')
        if header and 'Significant Strikes' in header.text:
            sig_strikes_section = section
            break
    else:
        return None

    per_round_link = sig_strikes_section.find_next('a',
                                                   class_='b-fight-details__collapse-link_rnd js-fight-collapse-link')
    if not per_round_link or 'Per round' not in per_round_link.text:
        return None

    per_round_section = per_round_link.find_next('table', class_='b-fight-details__table js-fight-table')
    if not per_round_section:
        return None

    sig_strikes_data = []
    round_bodies = per_round_section.find_all('tbody')
    if len(round_bodies) == 0:
        return None

    round_counter = 1
    fighter_counter = 0

    for tbody in round_bodies:
        rows = tbody.find_all('tr', class_='b-fight-details__table-row')
        for row in rows:
            fighter_data = row.find_all('td', class_='b-fight-details__table-col')
            if len(fighter_data) > 0:
                fighter_name_tags = fighter_data[0].find_all('p')
                fighter_names = [tag.text.strip() for tag in fighter_name_tags]
                metrics = fighter_data[1:]

                for j, fighter_name in enumerate(fighter_names):
                    fighter_info = {
                        'Round': f'Round {round_counter}',
                        'Fighter': fighter_name,
                        'Sig. Str.': metrics[0].find_all('p')[j].text.strip(),
                        'Sig. Str. %': metrics[1].find_all('p')[j].text.strip(),
                        'Head': metrics[2].find_all('p')[j].text.strip(),
                        'Body': metrics[3].find_all('p')[j].text.strip(),
                        'Leg': metrics[4].find_all('p')[j].text.strip(),
                        'Distance': metrics[5].find_all('p')[j].text.strip(),
                        'Clinch': metrics[6].find_all('p')[j].text.strip(),
                        'Ground': metrics[7].find_all('p')[j].text.strip(),
                    }
                    sig_strikes_data.append(fighter_info)

                    fighter_counter += 1
                    if fighter_counter == 2:
                        fighter_counter = 0
                        round_counter += 1
                        if round_counter > max_round:
                            round_counter = 1

    return sig_strikes_data

def extract_weight_class(html_content):
    """
    Extracts the weight class from the fight details HTML content.

    Args:
        html_content (str): The HTML content of the fight details page.

    Returns:
        str: The weight class of the fight.

    Example:
        "Lightweight"
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    weight_class_element = soup.find('i', class_='b-fight-details__fight-title')
    if weight_class_element:
        weight_class = clean_text(weight_class_element.text)
        return weight_class
    return 'Unknown Weight Class'


def fight_details(fight_urls, fighter_name):
    """
    Extracts and combines fight details and significant strikes data for a given fighter from multiple fight URLs.

    Args:
        fight_urls (list): A list of URLs for the fighter's fights.
        fighter_name (str): The name of the fighter.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined fight details and significant strikes data for the fighter.

    Example:
             Event    Round  ...  Ground
        0  Event 1  Round 1  ...     0 of 0
        1  Event 1  Round 2  ...     1 of 1
        2  Event 2  Round 1  ...     2 of 2
        ...
    """
    all_fights_data = []

    for url in fight_urls:
        html_content = fetch_webpage(url)
        if html_content:
            event_name = extract_event_name(html_content)
            max_round = extract_max_round(html_content)
            weight_class = extract_weight_class(html_content)  # Extract weight class
            fight_data = parse_fight_data(html_content, max_round)
            sig_strikes_data = parse_significant_strikes(html_content, max_round)
            if fight_data and sig_strikes_data:
                for entry in fight_data:
                    entry['Event'] = event_name
                    entry['Weight Class'] = weight_class  # Add weight class to fight data
                for entry in sig_strikes_data:
                    entry['Event'] = event_name
                    entry['Weight Class'] = weight_class  # Add weight class to significant strikes data
                all_fights_data.append((fight_data, sig_strikes_data))

    combined_data = []
    for fight_data, sig_strikes_data in all_fights_data:
        df = pd.DataFrame(fight_data)
        sig_strikes_df = pd.DataFrame(sig_strikes_data)

        # Drop redundant columns from significant strikes dataframe
        sig_strikes_df.drop(columns=['Sig. Str.', 'Sig. Str. %'], inplace=True)

        # Merge DataFrames on Round, Fighter, Event, and Weight Class
        combined_df = pd.merge(df, sig_strikes_df, on=['Round', 'Fighter', 'Event', 'Weight Class'], how='outer', suffixes=('', '_Sig_Strikes'))
        combined_data.append(combined_df)

    # Concatenate all combined dataframes
    final_combined_df = pd.concat(combined_data, ignore_index=True)
    final_combined_df = final_combined_df[['Event', 'Weight Class', 'Round', 'Fighter', 'KD', 'Sig. Str.', 'Sig. Str. %', 'Total Str.', 'TD', 'TD %', 'Sub. Att', 'Rev.', 'Ctrl', 'Head', 'Body', 'Leg', 'Distance', 'Clinch', 'Ground']]

    final_combined_df = final_combined_df[final_combined_df['Fighter'] == fighter_name].reset_index(drop=True)
    final_combined_df = final_combined_df.rename(columns={'Fighter': 'Name'})

    return final_combined_df
