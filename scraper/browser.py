# browser.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def init_driver(headless=True):
    chrome_opts = Options()
    if headless:
        chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_opts)
    driver.maximize_window()
    return driver
