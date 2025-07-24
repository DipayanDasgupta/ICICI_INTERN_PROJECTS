# utils/news_scraper.py

import subprocess
import json
import logging
import os

logger = logging.getLogger(__name__)

class NewsScraper:
    """
    A Python wrapper for the 'google-news-scraper' NPM package.
    Executes a Node.js script to scrape news and returns the parsed JSON data.
    """
    def __init__(self):
        # Verify that the Node.js script exists
        self.scraper_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'run_scraper.js')
        if not os.path.exists(self.scraper_script_path):
            raise FileNotFoundError(f"Scraper script not found at {self.scraper_script_path}")

    def scrape(self, search_term: str):
        """
        Scrapes Google News for a given search term.

        Args:
            search_term (str): The keyword or phrase to search for.

        Returns:
            list: A list of article dictionaries, or an empty list on failure.
        """
        if not search_term:
            logger.warning("Scrape called with an empty search term.")
            return []

        command = ["node", self.scraper_script_path, search_term]
        
        try:
            logger.info(f"Executing news scraper for search term: '{search_term}'")
            # Increased timeout for the scraper to run
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=90  # 90-second timeout for the subprocess
            )

            try:
                articles = json.loads(result.stdout)
                if isinstance(articles, list):
                    logger.info(f"Successfully scraped {len(articles)} articles for '{search_term}'.")
                    return articles
                else:
                    logger.error(f"Scraper returned non-list data for '{search_term}': {articles}")
                    return []
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from scraper output for '{search_term}'. Output: {result.stdout}")
                return []

        except subprocess.TimeoutExpired:
            logger.error(f"News scraper timed out for search term: '{search_term}'.")
            return []
        except subprocess.CalledProcessError as e:
            logger.error(f"News scraper script failed for '{search_term}' with exit code {e.returncode}.")
            logger.error(f"Stderr: {e.stderr}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while running the news scraper: {e}")
            return []