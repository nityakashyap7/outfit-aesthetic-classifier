from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import os
import requests
import base64
from PIL import Image
import io
import numpy as np


def scrape_one_search(class_name, search_query, num_scrolls=5):
    website = 'https://images.google.com/'
    driver = webdriver.Chrome()
    driver.get(website)
    time.sleep(2)


    # find the search bar and enter in search query
    search_bar = driver.find_element(By.XPATH, '//textarea')
    search_bar.send_keys(search_query)
    time.sleep(4) # need delay so search results can load

    for i in range(num_scrolls):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END) # 1 scroll to end of page
        time.sleep(2)  # Wait for images to load

    # get all the image elements up to scrolled point
    folder_name = class_name
    os.makedirs(folder_name, exist_ok=True)

    main_div = driver.find_element(By.XPATH, '//div[@role="main"]')
    images = main_div.find_elements(By.CSS_SELECTOR, 'img')
    print(len(images))
    for i, img in enumerate(images):
        src = img.get_attribute('src')

        if src:
            try:
                header, base64_data = src.split(',', 1)
                img_data = base64.b64decode(base64_data)

                image = Image.open(io.BytesIO(img_data)).convert('RGB') # Open the image from the byte stream and ensure it's in RGB format
                img_array = np.array(image) # Convert the image to a NumPy array (raw pixel values)
                
                np.save(os.path.join(folder_name, f'image_{i}.npy'), img_array) # Save the NumPy array to a .npy file

            except Exception as e:
                pass
                # print(f'Failed to download image_{i}.bin: {e}')


def main():
    # full list
    searches = [
        ['balletcore', ['balletcore site:pinterest.com filetype:jpg\n', 'ballet site:pinterest.com filetype:jpg\n', 'balletcore men outfits site:pinterest.com filetype:jpg\n'], 5],
        ['boho', ['hippie outfits site:pinterest.com filetype:jpg\n', 'bohemian outfits site:pinterest.com filetype:jpg\n', 'hippie outfits men site:pinterest.com filetype:jpg\n', 'bohemian outfits men site:pinterest.com filetype:jpg\n'], 5],
        ['coquette', ['coquette outfits site:pinterest.com filetype:jpg\n', 'dollette outfits site:pinterest.com filetype:jpg\n'], 5],
        ['southern', ['cowgirl outfits site:pinterest.com filetype:jpg\n', 'western outfits site:pinterest.com filetype:jpg\n', 'cowboy outfits site:pinterest.com filetype:jpg\n'], 5],
        ['punk', ['goth outfits site:pinterest.com filetype:jpg\n', 'punk outfits site:pinterest.com filetype:jpg\n', 'edgy outfits site:pinterest.com filetype:jpg\n', 'grunge outfits site:pinterest.com filetype:jpg\n', 'eboy egirl outfits site:pinterest.com filetype:jpg\n', 'goth outfits men site:pinterest.com filetype:jpg\n'], 5],
        ['y2k', ['y2k 2000s outfits site:pinterest.com filetype:jpg\n', 'y2k 2000s outfits men site:pinterest.com filetype:jpg\n', 'streetwear outfits site:pinterest.com filetype:jpg\n'], 5],
        ['cottagecore', ['y2k 2000s outfits site:pinterest.com filetype:jpg\n', 'y2k 2000s outfits men site:pinterest.com filetype:jpg\n', 'streetwear outfits site:pinterest.com filetype:jpg\n'], 5],
        ['academia', ['academia -myheroacademia -mha outfits site:pinterest.com filetype:jpg\n', 'light academia outfits site:pinterest.com filetype:jpg\n', 'dark academia outfits site:pinterest.com filetype:jpg\n'], 5],
        ['old-money' , ['old money outfits site:pinterest.com filetype:jpg\n', 'rich luxury classy outfits site:pinterest.com filetype:jpg\n'], 5],
        ['athleisure', ['athleisure outfits site:pinterest.com filetype:jpg\n'], 5],
    ]
    # bits (for testing)
    searches = [
        ['balletcore', ['balletcore site:pinterest.com filetype:jpg\n', 'ballet site:pinterest.com filetype:jpg\n', 'balletcore men outfits site:pinterest.com filetype:jpg\n'], 5],
        ['boho', ['hippie outfits site:pinterest.com filetype:jpg\n', 'bohemian outfits site:pinterest.com filetype:jpg\n', 'hippie outfits men site:pinterest.com filetype:jpg\n', 'bohemian outfits men site:pinterest.com filetype:jpg\n'], 5],
    ]

    for class_name, search_queries, num_scrolls in searches:
        for search_query in search_queries:
            print(f"Scraping {class_name} with query: {search_query.strip()}")
            scrape_one_search(class_name, search_query, num_scrolls)
        print(f"Finished scraping {class_name}.")


    

if __name__ == '__main__':
    main()
