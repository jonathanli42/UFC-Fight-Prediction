import requests
from bs4 import BeautifulSoup


def get_fight_urls(fighter_url):
    """
    Extracts the URLs of all fights for a given fighter from their UFC stats page.

    Args:
        fighter_url (str): The URL of the fighter's UFC stats page.

    Returns:
        list: A list of fight URLs.

    Example:
        >>> fighter_url = 'http://ufcstats.com/fighter-details/e5549c82bfb5582d'
        >>> fight_urls = get_fight_urls(fighter_url)
        >>> print(fight_urls)
        ['http://ufcstats.com/fight-details/abc123', 'http://ufcstats.com/fight-details/def456', ...]
    """
    try:
        response = requests.get(fighter_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        fight_urls = []
        for fight in soup.find_all('tr', class_='b-fight-details__table-row'):
            data_link = fight.get('data-link')
            if data_link and 'fight-details' in data_link:
                fight_urls.append(data_link)

        return fight_urls

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []
