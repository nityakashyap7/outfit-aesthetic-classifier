# outfit-aesthetic-classifier
#### USC Spring 2025 CSCI 467 course final project using CNNs and a custom scraped dataset

## Contributors
- Maia Nkonabang ()
- Nitya Kashyap (https://github.com/nityakashyap7)

## Introduction
Our project aims to develop a machine learning model that classifies outfit images into aesthetic categories such as streetwear, Y2K, grunge, and dark academia. The model will take an image—either of a person wearing an outfit or clothing laid out flat—and output the most similar aesthetic label, or the top two if multiple images are provided. Initially, we considered building an outfit recommendation generator, but given the scope of our coursework, we determined classification using a convolutional neural network (CNN) to be a more feasible approach. This project addresses the real-world challenge of identifying and organizing fashion aesthetics, helping users better understand their style and find similar looks online. It has applications in personal styling, fashion search tools, and trend analysis.

## Related works
There has been a lot of machine learning research in this field. One notable study by Ziegler et al. [1] used the DeepFashion dataset with an elastic warping rotation-invariant model to improve clothing category classification and fashion landmark detection. However, a key limitation was that elastic warping—where an image is stretched and distorted—could negatively impact results due to the dataset's limited size. Similarly, Jain and Kumar [1] leveraged the DeepFashion dataset with multiple models, including Decision Trees, Naïve Bayes, Random Forest, and Bayesian Forest, to build a classification system for clothing categories and subcategories. Their approach, however, was limited by the labor-intensive nature of manually labeling internet-sourced data. Another project by Jo et al. [1] developed a vector-based user-preferred fashion recommendation system using a CNN model and a private dataset. Despite its accuracy and precision, reliance on text-based search methods posed challenges since fashion recommendation heavily depends on clothing design.

After reviewing several studies in this space, it's clear that fashion classification presents diverse challenges, with limitations varying based on the dataset and model choices. The following section will outline how our project differs from existing approaches.

## Dataset and evolution
Since there is no existing dataset that labels outfit images with their corresponding aesthetic, we will create our own by scraping images from Google Images. Google Images is more bot-friendly than platforms like Pinterest and does not require authentication, making it a more practical source. To efficiently collect data, we will use Scrapy for sending requests, Selenium for handling CAPTCHA and dynamic content loading [2], and Scrapy’s AutoThrottle for managing request speed to avoid bot detection [3]. We will also incorporate rotating proxies to further minimize detection risks [4]. Our goal is to collect 3,000 images per aesthetic category, filtering out irrelevant content such as ads and videos. Since search results tend to favor women’s fashion, we will refine our queries (e.g., adding terms like “male grunge outfits”) to ensure more balanced data collection. Once compiled, the dataset will be stored in a structured format using pandas for easy processing and model training.
For evaluation, we will split our dataset into training, development, and test sets to ensure the model is tested on unseen data. The development set will be utilized to fine-tune the neural network’s hyperparameters. Since we are creating a balanced dataset, accuracy will serve as our primary evaluation metric. However, if we identify certain aesthetics being harder to classify, we will incorporate F1-score to better assess model performance.

## Plans for baselines and methods
Our initial baseline approach will use a non-machine learning method, where we match images from our dataset with the search tags used to find them. For our machine learning approach, we plan to combine a CNN for image feature extraction with a text-based classifier, such as Naïve Bayes, which we learned in lecture.

Some key challenges we anticipate include data collection, which we aim to address through automated scraping bots, and handling “noisy” labels—fashion images overloaded with trendy or irrelevant descriptors—which we plan to manage using supervised Naïve Bayes learning. Another challenge will be experimenting with different feature fusion techniques to combine visual and textual data effectively.

Ultimately, we aim to develop a machine learning model that classifies outfits into aesthetics by leveraging a dataset scraped from Google Images and applying CNN and Naïve Bayes classification models to categorize training, validation, and test data. We hope this project will help users better understand their style and discover fashion recommendations tailored to their preferences.

## References


[1] Shushi, A., & Abdulazeez, A. M. (2024). Fashion Design Classification based on machine learning and Deep Learning Algorithms: A Review. Indonesian Journal of Computer Science, 13(3). https://doi.org/10.33022/ijcs.v13i3.3980

[2] Krukowski, I. (2024, May 13). Web scraping tutorial using selenium & python (+ examples). ScrapingBee. https://www.scrapingbee.com/blog/selenium-python/

[3] Scrapy. (2024, November 19). Autothrottle extension. AutoThrottle extension - Scrapy 2.12.0 documentation. https://docs.scrapy.org/en/latest/topics/autothrottle.html

[4] TexAu. (n.d.). Proxy rotation. Definition, Importance & Best Practices. https://www.texau.com/glossary/proxy-rotation
