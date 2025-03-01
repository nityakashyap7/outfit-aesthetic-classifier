from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

def main():

    website = 'https://images.google.com/'
    driver = webdriver.Chrome()
    driver.get(website)
    # time.sleep(2)

    # find the search bar and enter in search query
    search_bar = driver.find_element(By.XPATH, '//textarea[@class="gLFyf"]')
    search_bar.send_keys('male y2k outfits site:pinterest.com filetype:jpg\n')
    time.sleep(4) # need delay so search results can load

    # scroll to the bottom of the page to load more images
    keepScrolling = True
    numScrolls = 0
    prompt_stops = [3, 2, 1] # prompts you if you want to continue scrolling after 3, 2, then 1 scroll(s); idea is that search result relevancy degrades faster as u scroll down 
    stop = prompt_stops.pop(0) 
    while keepScrolling:
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END) # 1 scroll to end of page
        time.sleep(5)  # Wait for images to load
        numScrolls += 1
        print(f"Scrolled {numScrolls} times")

        # check if we should keep scrolling
        if numScrolls == stop:
            keepScrolling = input(f"Would you like to keep scrolling? (y/n)").lower == 'y'
            stop = prompt_stops.pop(0)


if __name__ == '__main__':
    main()
