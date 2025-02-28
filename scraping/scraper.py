from selenium import webdriver
from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
import time

def main():

    website = 'https://images.google.com/'
    driver = webdriver.Chrome()
    driver.get(website)
    # time.sleep(2)

    # find the search bar and enter in search query
    search_bar = driver.find_element(By.XPATH, '//textarea[@class="gLFyf"]')
    search_bar.send_keys('intext:y2k fashion pinterest\n')
    time.sleep(2)



if __name__ == '__main__':
    main()
