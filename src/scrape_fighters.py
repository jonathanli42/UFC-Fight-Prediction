import requests
from bs4 import BeautifulSoup
def get_all_fighter_urls(base_url):
    """
    Retrieves the URLs of all fighters' profile pages from the UFC stats website.

    Args:
        base_url (str): The base URL of the UFC stats fighters page.

    Returns:
        dict: A dictionary with fighter names as keys and their profile URLs as values.

    Raises:
        requests.exceptions.RequestException: If the HTTP request to the UFC stats page fails.

    Notes:
        - The function iterates through all alphabetic characters to access all fighters.
        - It scrapes the page to find each fighter's name and profile URL.
        - The fighter's name is used as the key, and the profile URL is used as the value in the dictionary.
    """
    fighter_urls = {}

    for char in 'abcdefghijklmnopqrstuvwxyz':
        url = f"{base_url}?char={char}&page=all"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            fighter_rows = soup.find_all('tr', class_='b-statistics__table-row')
            for row in fighter_rows:
                name_elements = row.find_all('a', class_='b-link b-link_style_black')
                if len(name_elements) >= 2:
                    first_name = name_elements[0].text.strip()
                    last_name = name_elements[1].text.strip()
                    fighter_name = f"{first_name} {last_name}"
                    fighter_url = name_elements[0]['href']
                    fighter_urls[fighter_name] = fighter_url
        except requests.exceptions.RequestException as e:
            print(f"Request failed for character {char}: {e}")

    return fighter_urls

