import requests
from bs4 import BeautifulSoup

def clean_text(text):
    """
    Cleans the given text by removing extra spaces.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    return ' '.join(text.split())

def fetch_webpage(url):
    """
    Fetches the content of a webpage.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: The content of the fetched webpage.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
        return None

def extract_fight_dates_and_results(html_content):
    """
    Extracts fight dates, event names, results, and fighter names from the given HTML content.

    Args:
        html_content (str): The HTML content of the webpage to parse.

    Returns:
        list: A list of dictionaries, each containing 'Event', 'Date', 'Result', 'Method', 'Fighter_1', and 'Fighter_2' of a fight.
              Example: [{'Event': 'UFC 281: Adesanya vs. Pereira', 'Date': 'Nov. 12, 2022', 'Result': 'win', 'Method': 'KO/TKO', 'Fighter_1': 'Alex Pereira', 'Fighter_2': 'Jiri Prochazka'}, ...]
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    fight_details_list = []

    event_rows = soup.find_all('tr',
                               class_='b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click')
    for event_row in event_rows:
        result_tag = event_row.find('a', class_='b-flag')
        event_name_tag = event_row.find_all('td', class_='b-fight-details__table-col l-page_align_left')[1].find('a',
                                                                                                                 class_='b-link b-link_style_black')
        date_tag = event_row.find_all('td', class_='b-fight-details__table-col l-page_align_left')[1].find_all('p',
                                                                                                               class_='b-fight-details__table-text')[
            1]
        method_tag = event_row.find_all('td', class_='b-fight-details__table-col l-page_align_left')[2]
        fighter_tags = event_row.find_all('td', class_='b-fight-details__table-col l-page_align_left')[0].find_all('p',
                                                                                                                   class_='b-fight-details__table-text')

        if result_tag and event_name_tag and date_tag and len(fighter_tags) == 2:
            result_text = clean_text(result_tag.find('i', class_='b-flag__text').text)
            event_name = clean_text(event_name_tag.text)
            event_date = clean_text(date_tag.text)
            method_text = clean_text(method_tag.text) if method_tag else 'Unknown'
            fighter1 = clean_text(fighter_tags[0].text)
            fighter2 = clean_text(fighter_tags[1].text)

            if event_name.startswith("UFC"):
                fight_details_list.append({
                    'Event': event_name,
                    'Date': event_date,
                    'Result': result_text,
                    'Method': method_text,
                    'Fighter_1': fighter1,
                    'Fighter_2': fighter2
                })

    return fight_details_list
