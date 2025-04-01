from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from concurrent.futures import ThreadPoolExecutor

def scrape_images(search_query, num_scrolls):
    website = 'https://images.google.com/'
    driver = webdriver.Chrome()
    driver.get(website)

    search_bar = driver.find_element(By.XPATH, '//textarea')
    search_bar.send_keys(search_query)
    time.sleep(4)

    for i in range(num_scrolls):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(1)
        print(f"{search_query.strip()} - Scrolled {i + 1}/{num_scrolls} times")

    driver.quit()

def main():
    # key: search query, value: max amount of scrolls (defaulted to 5)
    searches = {
        'balletcore site:pinterest.com filetype:jpg\n': 5,
        'hippie outfits site:pinterest.com filetype:jpg\n': 5,
        'coquette outfits site:pinterest.com filetype:jpg\n': 5,
        'cowgirl outfits site:pinterest.com filetype:jpg\n': 5,
        'goth outfits site:pinterest.com filetype:jpg\n': 5,
        'y2k 2000s outfits site:pinterest.com filetype:jpg\n': 5,
        'cottagecore outfits site:pinterest.com filetype:jpg\n': 5,
        'academia -myheroacademia -mha outfits site:pinterest.com filetype:jpg\n': 5,
        'old money outfits site:pinterest.com filetype:jpg\n': 5
    }

    with ThreadPoolExecutor(max_workers=5) as executor: # this just allows multiple drivers to execute in parallel
        executor.map(lambda item: scrape_images(item[0], item[1]), searches.items())

if __name__ == '__main__':
    main()
