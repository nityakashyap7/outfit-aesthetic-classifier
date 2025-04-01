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

def main():

    website = 'https://images.google.com/'
    driver = webdriver.Chrome()
    driver.get(website)
    # time.sleep(2)

    # find the search bar and enter in search query
    search_bar = driver.find_element(By.XPATH, '//textarea')
    search_bar.send_keys('male y2k outfits site:pinterest.com filetype:jpg\n')
    time.sleep(4) # need delay so search results can load

    # scroll to the bottom of the page to load more images
    # keep_scrolling = True
    # num_scrolls = 0
    # prompt_stops = [3, 2, 1] # prompts you if you want to continue scrolling after 3, 2, then 1 scroll(s); idea is that search result relevancy degrades faster as u scroll down 
    # stop = prompt_stops.pop(0) 
    # while keepScrolling:
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END) # 1 scroll to end of page
    time.sleep(1)  # Wait for images to load
    # num_scrolls += 1
    # print(f"Scrolled {num_scrolls} times")

    # check if we should keep scrolling
    # if num_scrolls == stop:
    #     keep_scrolling = input(f"Would you like to keep scrolling? (y/n)").lower() == 'y'
    #     stop = prompt_stops.pop(0)
    #     num_scrolls = 0

    # get all the image elements up to scrolled point
    folder_name = 'downloaded_images'
    os.makedirs(folder_name, exist_ok=True)


    main_div = driver.find_element(By.XPATH, '//div[@role="main"]')
    images = main_div.find_elements(By.CSS_SELECTOR, 'img')
    # img = images[0]   # testing only with first 10 images
    for i, img in enumerate(images):
        src = img.get_attribute('src')
    # print(f'Image: {src}')
        if src:
            try:
                header, base64_data = src.split(',', 1)
                print(f'Image: {src}')
                img_data = base64.b64decode(base64_data)

                image = Image.open(io.BytesIO(img_data)).convert('RGB') # Open the image from the byte stream and ensure it's in RGB format
                img_array = np.array(image) # Convert the image to a NumPy array (raw pixel values)
                
                np.save(os.path.join(folder_name, f'image_{i}.npy'), img_array) # Save the NumPy array to a .npy file
            except Exception as e:
                print(f'Failed to download image_{i}.bin: {e}')


if __name__ == '__main__':
    main()
