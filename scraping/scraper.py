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
import traceback

def scrape_one_search(class_name, search_query, num_scrolls=5):
    website = 'https://images.google.com/'
    driver = webdriver.Chrome()
    driver.get(website)
    time.sleep(2)

    # Find the search bar and enter search query
    search_bar = driver.find_element(By.XPATH, '//textarea')
    search_bar.send_keys(search_query)
    time.sleep(5)  # Delay to allow search results to load

    # Scroll down the page to load more images
    for _ in range(num_scrolls):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(2)  # Wait for images to load

    # Get all the image elements up to the scrolled point
    folder_name = class_name
    os.makedirs(folder_name, exist_ok=True)

    # Find the images on the page
    main_div = driver.find_element(By.XPATH, '//div[@role="main"]')
    images = main_div.find_elements(By.CSS_SELECTOR, 'img')
    print(f"Found {len(images)} images for query: {search_query.strip()}")

    # Find the starting index for saving images
    existing_files = [f for f in os.listdir(folder_name) if f.endswith('.npy')]
    next_index = len(existing_files)  # Start numbering from the last saved image

    for i, img in enumerate(images):
        src = img.get_attribute('src')
        
        # Skip if no source or if it's any other junk sources (ex: the Pinterest icon)
        if not src or any(sub in src for sub in [
            'pinterest.com/pinbutton',
            'pinterest.com/pinit',
            'logo',
            'favicon',
            'sprite',
            'static',
            'icon',
            'blank',
            'data:image/gif',  # often 1x1 pixel placeholders
        ]):
            continue

        try:
            # Check if the image is base64-encoded
            if src.startswith('data:image'):
                header, base64_data = src.split(',', 1)
                img_data = base64.b64decode(base64_data)
            else:
                # If it's a URL, attempt to download it
                print("HTTPS URL FOUND: ", src)
                response = requests.get(src, timeout=5)
                if response.status_code == 200:
                    img_data = response.content
                else:
                    continue  # Skip if the image couldn't be downloaded

            # Convert the image to a PIL Image object first
            image = Image.open(io.BytesIO(img_data))

            # Skip if the image is too small (likely an icon)
            if img_array.shape[0] < 100 or img_array.shape[1] < 100:
                continue
            
            image.convert('RGB')
            img_array = np.array(image)

            # Save the NumPy array to a .npy file
            np.save(os.path.join(folder_name, f'image_{next_index}.npy'), img_array)
            next_index += 1  # Increment index for next image

        except Exception as e:
            print(f'-----Failed to process image {i}----------')
            traceback.print_exc()
            continue  # Skip this image if an error occurs

    #driver.quit()  # Close the browser session after scraping


def main():
    searches = [
        ['balletcore', ['balletcore site:pinterest.com filetype:jpg\n', 'ballet site:pinterest.com filetype:jpg\n', 'balletcore men outfits site:pinterest.com filetype:jpg\n'], 5],
        ['boho', ['hippie outfits site:pinterest.com filetype:jpg\n', 'bohemian outfits site:pinterest.com filetype:jpg\n', 'hippie outfits men site:pinterest.com filetype:jpg\n', 'bohemian outfits men site:pinterest.com filetype:jpg\n'], 5], # earthy outfits
        ['coquette', ['coquette outfits site:pinterest.com filetype:jpg\n', 'dollette outfits site:pinterest.com filetype:jpg\n'], 5],
        ['southern', ['cowgirl outfits site:pinterest.com filetype:jpg\n', 'western outfits site:pinterest.com filetype:jpg\n', 'cowboy outfits site:pinterest.com filetype:jpg\n'], 5],
        ['punk', ['goth outfits site:pinterest.com filetype:jpg\n', 'punk outfits site:pinterest.com filetype:jpg\n', 'edgy outfits site:pinterest.com filetype:jpg\n', 'grunge outfits site:pinterest.com filetype:jpg\n', 'eboy egirl outfits site:pinterest.com filetype:jpg\n', 'goth outfits men site:pinterest.com filetype:jpg\n'], 5],
        ['y2k', ['y2k 2000s outfits site:pinterest.com filetype:jpg\n', 'y2k 2000s outfits men site:pinterest.com filetype:jpg\n', 'streetwear outfits site:pinterest.com filetype:jpg\n'], 5],
        ['cottagecore', ['cottagecore outfits site:pinterest.com filetype:jpg\n', 'fairycore outfits site:pinterest.com filetype:jpg\n'], 5],
        ['academia', ['academia -myheroacademia -mha outfits site:pinterest.com filetype:jpg\n', 'light academia outfits site:pinterest.com filetype:jpg\n', 'dark academia outfits site:pinterest.com filetype:jpg\n'], 5],
        ['old-money' , ['old money outfits site:pinterest.com filetype:jpg\n', 'rich luxury classy outfits site:pinterest.com filetype:jpg\n'], 5],
        ['athleisure', ['athleisure outfits site:pinterest.com filetype:jpg\n'], 5],
    ]

    for class_name, search_queries, num_scrolls in searches:
        for search_query in search_queries:
            print(f"Scraping {class_name} with query: {search_query.strip()}")
            scrape_one_search(class_name, search_query, num_scrolls)
        print(f"Finished scraping {class_name}.")


if __name__ == '__main__':
    main()
