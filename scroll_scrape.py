from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
# import pandas as pd
import re
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()))

url ="https://paperswithcode.com/"
driver = webdriver.Chrome()
driver.get(url)
element = driver.find_elements_by_class_name("footer-contact-item")
actions = ActionChains(driver)
actions.move_to_element(element)