from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time 
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import csv 





driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()))
MAIN_URL = "https://paperswithcode.com/" # you can choose another chriteria
NEW_URL = "https://paperswithcode.com/latest" # newest papers


driver.get(NEW_URL)  


end_time = time.time() + 60 # pick however time you want , i got 25 seconds scrolltime
while True:
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    if time.time() > end_time:
        break
    time.sleep(1)  


soup = BeautifulSoup(driver.page_source, 'html.parser')
titles  = soup.find_all('h1')
abstracts = soup.find_all("p", class_="item-strip-abstract")


with open('trending_papers.csv', 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # writer.writerow(['title', 'abstract'])  # Write the header
    

    for title, abstract  in zip(titles, abstracts):
        writer.writerow([title.text.strip(), abstract.text.strip()]) 
        
 
        
driver.quit()
