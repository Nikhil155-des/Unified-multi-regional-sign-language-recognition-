from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json
import os
import shutil
from tqdm import tqdm

# 100 Words (Extend to 500)
WORDS = [
     "hello", "thank you", "goodbye", "yes", "no", "please", "sorry", "help", "stop", "go",
    "where", "when", "why", "who", "how", "what", "love", "friend", "family", "mother",
    "father", "brother", "sister", "child", "baby", "grandmother", "grandfather", "uncle", "aunt",
    "cousin", "teacher", "student", "doctor", "nurse", "police", "firefighter", "engineer", "scientist",
    "artist", "musician", "dancer", "actor", "writer", "athlete", "coach", "driver", "chef", "farmer",
    "lawyer", "judge", "president", "minister", "secretary", "leader", "boss", "worker", "employee",
    "colleague", "friendship", "team", "group", "community", "nation", "world", "earth", "universe",
    "sun", "moon", "star", "planet", "galaxy", "space", "ocean", "river", "lake", "mountain",
    "forest", "tree", "flower", "grass", "bird", "animal", "dog", "cat", "horse", "cow",
    "sheep", "goat", "fish", "whale", "dolphin", "shark", "elephant", "lion", "tiger", "bear",
    "zebra", "giraffe", "monkey", "kangaroo", "panda", "penguin", "eagle", "owl", "snake", "spider",
    "insect", "butterfly", "bee", "ant", "mosquito", "house", "home", "apartment", "building", "school",
    "university", "library", "hospital", "church", "mosque", "temple", "supermarket", "store", "mall", "market",
    "restaurant", "cafe", "hotel", "airport", "station", "bus", "train", "car", "bike", "airplane",
    "boat", "ship", "submarine", "rocket", "road", "bridge", "tunnel", "street", "sidewalk", "park",
    "playground", "garden", "farm", "zoo", "museum", "theater", "stadium", "gym", "pool", "beach",
    "vacation", "travel", "holiday", "ticket", "passport", "luggage", "bag", "wallet", "money", "bank",
    "credit", "debit", "loan", "insurance", "salary", "job", "career", "business", "company", "industry",
    "technology", "computer", "internet", "website", "email", "phone", "tablet", "camera", "television", "radio",
    "newspaper", "magazine", "book", "notebook", "pen", "pencil", "paper", "scissors", "glue", "tape",
    "clock", "watch", "calendar", "time", "morning", "afternoon", "evening", "night", "yesterday", "today",
    "tomorrow", "week", "month", "year", "decade", "century", "history", "future", "past", "present",
    "happy", "sad", "angry", "excited", "nervous", "brave", "afraid", "strong", "weak", "healthy",
    "sick", "medicine", "doctor", "hospital", "ambulance", "rescue", "help", "support", "care", "love",
    "marriage", "wedding", "birthday", "party", "celebration", "festival", "holiday", "Christmas", "Easter", "New Year",
    "music", "song", "dance", "art", "painting", "drawing", "sculpture", "photography", "film", "cinema",
    "sport", "game", "exercise", "run", "walk", "swim", "jump", "lift", "stretch", "play",
    "learn", "study", "read", "write", "speak", "listen", "understand", "think", "imagine", "create",
    "eat", "drink", "cook", "bake", "boil", "fry", "grill", "mix", "cut", "serve",
    "family", "friend", "colleague", "team", "leader", "teacher", "student", "child", "parent", "grandparent"
]  # Add more words

# Dataset structure
dataset = {"train": {"BSL": {}, "NZSL": {}}, "test": {"ISL": {}, "Auslan": {}}}
CHECKPOINT_FILE = "sign_language_checkpoint.json"

# Selenium WebDriver setup (Headless for faster execution)
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--log-level=3")  # Suppress warnings

try:
    driver = webdriver.Chrome(options=options)
except Exception as e:
    print(f"Error initializing WebDriver: {e}")
    exit(1)


# Function to scrape video links
def scrape_video(url, retries=3):
    """Scrape video link for a given word, with retry mechanism."""
    for attempt in range(retries):
        try:
            driver.get(url)
            time.sleep(1)  # Reduced delay
            video = driver.find_element(By.TAG_NAME, "video")
            return video.get_attribute("src")
        except:
            if attempt == retries - 1:
                return None  # Final attempt failed
            time.sleep(1)  # Wait before retrying


# Functions to scrape different sign languages
def scrape_all(word):
    """Scrape all sign language sources for a word."""
    return {
        "BSL": scrape_video(f"https://www.signbsl.com/sign/{word}"),
        "NZSL": scrape_video(f"https://www.nzsl.nz/signs/{word}"),
        "ISL": scrape_video(f"https://www.indiansignlanguage.org/{word}"),  # Update if needed
        "Auslan": scrape_video(f"https://auslan.org.au/dictionary/{word}")
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
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(tqdm(executor.map(scrape_all, WORDS), total=len(WORDS), desc="Scraping Words"))

# Store results in dataset
for i, word in enumerate(WORDS):
    dataset["train"]["BSL"][word] = results[i]["BSL"]
    dataset["train"]["NZSL"][word] = results[i]["NZSL"]
    dataset["test"]["ISL"][word] = results[i]["ISL"]
    dataset["test"]["Auslan"][word] = results[i]["Auslan"]

    # Save checkpoint every 10 words
    if i % 10 == 0:
        temp_file = CHECKPOINT_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(dataset, f, indent=4)
        shutil.move(temp_file, CHECKPOINT_FILE)  # Atomic write
        print(f"Checkpoint saved at word {i}")

# Save final dataset
with open("sign_language_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

# Close WebDriver
driver.quit()

print("Dataset saved successfully as sign_language_dataset.json")
