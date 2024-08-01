import os
import zipfile
import requests
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
from selenium.webdriver.chrome.options import Options

import undetected_chromedriver as uc


def load_driver(
    window_size="auto",
    headless=True,
    driver="undetected_chromedriver",
    driver_version=None,
    chrome_type="CHROME",
    adblock=True,
    adblock_version="6.0.2-mv3",
    extension_cache_dir=os.path.join(os.path.dirname(__file__), "extensions"),
    *,
    service=None,
    additional_options=None,
):
    options = Options()
    if service is None:
        chrome_type = chrome_type.upper()
        if chrome_type == "CHROMIUM":
            chrome_type = ChromeType.CHROMIUM
        elif chrome_type == "CHROME":
            chrome_type = ChromeType.GOOGLE
        elif chrome_type == "BRAVE":
            chrome_type = ChromeType.BRAVE
        service = ChromeDriverManager(driver_version=driver_version, chrome_type=chrome_type).install()
    if headless:
        options.add_argument("--headless")
    if adblock:
        try:
            adblock_url = f"https://code.getadblock.com/releases/adblockchrome-{adblock_version}.zip"
            adblock_path = os.path.join(extension_cache_dir, f"adblockchrome-{adblock_version}")
            if not os.path.isdir(adblock_path):
                os.makedirs(os.path.join(adblock_path, ".."), exist_ok=True)
                # Download the adblock zip file
                response = requests.get(adblock_url)
                with open(f"{adblock_path}.zip", "wb") as file:
                    file.write(response.content)
                # Unzip the downloaded file
                with zipfile.ZipFile(f"{adblock_path}.zip", "r") as zip_ref:
                    zip_ref.extractall(adblock_path)
                # Remove the zip file after extraction
                os.remove(f"{adblock_path}.zip")
            options.add_argument(f"--load-extension={os.path.abspath(adblock_path)}")
        except Exception as e:
            print(f"Error loading adblock extension: {e}")
    if driver == "undetected_chromedriver":
        driver = uc.Chrome(headless=headless, options=options, driver_executable_path=service)
        if window_size != "auto":
            driver.set_window_size(*window_size)
        return driver
    elif driver == "chrome":
        options = Options()
        if additional_options is not None:
            for option in additional_options:
                options.add_argument(option)
        service = webdriver.chrome.service.Service(service)
        driver = webdriver.Chrome(service=service, options=options)
        if window_size != "auto":
            driver.set_window_size(*window_size)
        return driver
    else:
        raise ValueError(f"Unknown driver: {driver}")
