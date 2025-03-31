from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json
import os
import shutil
import random
from tqdm import tqdm

# 100 Words (can be extended to 500)
WORDS = [
    "hello", "thank you", "goodbye", "yes", "no", "please", "sorry", "help", "stop", "go",
    "where", "when", "why", "who", "how", "what", "love", "friend", "family", "mother",
    # Additional words can be added here
][:20]  # Limiting to first 20 words for testing; remove slice for full list

# Dataset structure - restructured to have multiple videos per word
dataset = {
    "train": {},  # Will contain words as keys, each pointing to a list of 4-5 video links
    "test": {}    # Will contain words as keys, each pointing to a list of 1-2 video links
}

CHECKPOINT_FILE = "sign_language_checkpoint.json"

# Selenium WebDriver setup (Headless for faster execution)
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--log-level=3")

try:
    driver = webdriver.Chrome(options=options)
except Exception as e:
    print(f"Error initializing WebDriver: {e}")
    exit(1)

# List of sign language sources for training data
TRAINING_SOURCES = [
    # Main sources
    {"name": "BSL", "url_template": "https://www.signbsl.com/sign/{word}"},
    {"name": "NZSL", "url_template": "https://www.nzsl.nz/signs/{word}"},
    {"name": "ASL", "url_template": "https://www.handspeak.com/word/search/index.php?id={word}"},
    
    # Secondary sources with slight variations to get more examples
    {"name": "BSL_Alt", "url_template": "https://www.british-sign.co.uk/british-sign-language/dictionary/{word}/"},
    {"name": "NZSL_Alt", "url_template": "https://www.nzsl.nz/signs/search/{word}"},
]

# List of sign language sources for testing data
TESTING_SOURCES = [
    {"name": "ISL", "url_template": "https://www.indiansignlanguage.org/dictionary/{word}"},
    {"name": "Auslan", "url_template": "https://auslan.org.au/dictionary/words/{word}.html"},
]

def scrape_video(url, retries=3):
    """Scrape video link for a given word, with retry mechanism."""
    for attempt in range(retries):
        try:
            driver.get(url)
            time.sleep(1.5)  # Allow page to load
            
            # Try different video selectors
            selectors = [
                By.TAG_NAME, "video",
                By.CSS_SELECTOR, "video source",
                By.CSS_SELECTOR, ".video-container video",
                By.CSS_SELECTOR, ".sign-video"
            ]
            
            for i in range(0, len(selectors), 2):
                try:
                    element = driver.find_element(selectors[i], selectors[i+1])
                    if selectors[i+1] == "video":
                        return element.get_attribute("src")
                    else:
                        return element.get_attribute("src") or element.get_attribute("data-src")
                except:
                    continue
                    
            # If we're here, no video was found with our selectors
            print(f"No video found at {url}")
            return None
            
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed to scrape {url}: {str(e)}")
                return None
            time.sleep(1)  # Wait before retrying

def scrape_word(word):
    """Scrape all sources for a given word."""
    train_links = []
    test_links = []
    
    # Scrape training sources
    for source in TRAINING_SOURCES:
        url = source["url_template"].format(word=word)
        link = scrape_video(url)
        if link:
            train_links.append(link)
    
    # Scrape testing sources
    for source in TESTING_SOURCES:
        url = source["url_template"].format(word=word)
        link = scrape_video(url)
        if link:
            test_links.append(link)
    
    # Add random variations in URL to get more results if needed
    if len(train_links) < 4:
        # Try with word variations (plural, -ing form, etc.)
        variations = [f"{word}s", f"{word}ing", f"{word}ed"]
        for variation in variations:
            if len(train_links) >= 5:
                break
            
            # Try each variation with a random source
            source = random.choice(TRAINING_SOURCES)
            url = source["url_template"].format(word=variation)
            link = scrape_video(url)
            if link and link not in train_links:
                train_links.append(link)
    
    return {
        "train": train_links[:5],  # Limit to 5 videos for training
        "test": test_links[:2]     # Limit to 2 videos for testing
    }

# Load checkpoint if available and valid
if os.path.exists(CHECKPOINT_FILE) and os.path.getsize(CHECKPOINT_FILE) > 0:
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            dataset = json.load(f)
            print("Checkpoint loaded.")
    except json.JSONDecodeError:
        print("Corrupt checkpoint file. Starting fresh.")
else:
    print("No checkpoint found. Starting fresh.")

# Scrape data with multithreading
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(executor.map(scrape_word, WORDS), total=len(WORDS), desc="Scraping Words"))

# Store results in dataset
for i, word in enumerate(WORDS):
    # Only store words that have at least one training video
    if results[i]["train"]:
        dataset["train"][word] = results[i]["train"]
    
    # Only store words that have at least one testing video
    if results[i]["test"]:
        dataset["test"][word] = results[i]["test"]
    
    # Save checkpoint every 5 words
    if (i + 1) % 5 == 0 or i == len(WORDS) - 1:
        temp_file = CHECKPOINT_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(dataset, f, indent=4)
        shutil.move(temp_file, CHECKPOINT_FILE)  # Atomic write
        print(f"Checkpoint saved at word {i+1}/{len(WORDS)}")

# Save final dataset
with open("sign_language_dataset_notfinal.json", "w") as f:
    json.dump(dataset, f, indent=4)

# Print stats
train_word_count = len(dataset["train"])
test_word_count = len(dataset["test"])
total_train_videos = sum(len(videos) for videos in dataset["train"].values())
total_test_videos = sum(len(videos) for videos in dataset["test"].values())

print(f"\nDataset Statistics:")
print(f"Words in training set: {train_word_count}")
print(f"Words in testing set: {test_word_count}")
print(f"Total training videos: {total_train_videos}")
print(f"Total testing videos: {total_test_videos}")
print(f"Average videos per word (training): {total_train_videos/train_word_count:.2f}")
print(f"Average videos per word (testing): {total_test_videos/test_word_count:.2f}")

# Close WebDriver
driver.quit()

print("Dataset saved successfully as sign_language_dataset_notfinal.json")