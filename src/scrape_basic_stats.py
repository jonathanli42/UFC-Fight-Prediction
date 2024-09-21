import requests
from bs4 import BeautifulSoup


def fetch_webpage(url):
    """
    Fetches the content of a webpage.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: The content of the webpage if the request is successful, None otherwise.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def parse_record(record_text):
    """
    Parses the record text to extract wins, losses, draws, and no contests.

    Args:
        record_text (str): The record text in the format 'Record: W-L-D (NC)'

    Returns:
        tuple: A tuple containing wins, losses, draws, and no contests.
    """
    parts = record_text.split()
    main_record = parts[1]
    record_parts = main_record.split('-')
    wins = int(record_parts[0])
    losses = int(record_parts[1])
    draws = int(record_parts[2])
    nc = 0
    if len(parts) > 2:
        nc_part = parts[2]
        nc = int(nc_part.strip('()').split()[0])
    return wins, losses, draws, nc


def convert_height(height):
    """
    Converts height from feet and inches to total inches.

    Args:
        height (str): The height in the format "X' Y\""

    Returns:
        str: The height in inches.
    """
    if height and "'" in height and '"' in height:
        parts = height.split("'")
        feet = int(parts[0].strip())
        inches = int(parts[1].strip().replace('"', ''))
        total_inches = (feet * 12) + inches
        return f"{total_inches} inches"
    return height


def convert_reach(reach):
    """
    Converts reach from inches to a standardized format.

    Args:
        reach (str): The reach in the format 'X"'

    Returns:
        str: The reach in inches.
    """
    if reach and '"' in reach:
        reach = reach.replace('"', '').strip()
        return f"{reach} inches"
    return reach


def get_fighter_basic_stats(fighter_url):
    """
    Retrieves basic stats of a fighter from their UFC stats page.

    Args:
        fighter_url (str): The URL of the fighter's UFC stats page.

    Returns:
        dict: A dictionary containing the fighter's basic stats.
    """
    html_content = fetch_webpage(fighter_url)
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, 'html.parser')
    name_element = soup.find('span', {'class': 'b-content__title-highlight'})
    record_element = soup.find('span', {'class': 'b-content__title-record'})

    if name_element and record_element:
        name = name_element.text.strip()
        record_text = record_element.text.strip()
        print(f"Raw record text: {record_text}")  # Debug print
        try:
            wins, losses, draws, nc = parse_record(record_text)
        except Exception as e:
            print(f"Error processing record: {e}")
            wins = losses = draws = nc = 0

        details = {}
        details_elements = soup.select('ul.b-list__box-list li.b-list__box-list-item')
        for item in details_elements:
            title_element = item.find('i', {'class': 'b-list__box-item-title'})
            if title_element:
                title = title_element.text.strip().replace(':', '')
                value = item.text.replace(title_element.text, '').strip()
                details[title] = value

        height = convert_height(details.get('Height', ''))
        reach = convert_reach(details.get('Reach', ''))
        stats = {
            'Name': name,
            'Wins': wins,
            'Losses': losses,
            'Draws': draws,
            'No Contests': nc,
            'Height': height,
            'Weight': details.get('Weight', ''),
            'Reach': reach,
            'Stance': details.get('STANCE', ''),
            'DOB': details.get('DOB', '')
        }
        return stats
    else:
        return {'Name': "Unknown", 'Wins': 0, 'Losses': 0, 'Draws': 0, 'No Contests': 0}
