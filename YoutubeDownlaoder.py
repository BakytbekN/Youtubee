#https://www.youtube.com/watch?v=a-hoA97VcoU&t=14s
#https://www.youtube.com/watch?v=t2hXCo4j1ws&t=124s
#https://www.youtube.com/watch?v=uSHy7aFrMSw
#https://www.youtube.com/watch?v=_ndPr4k9ZSQ&t=243s

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
 
def check_exists_by_xpath(xpath,driver):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True
 
def init_driver():
    driver = webdriver.Firefox()
    driver.wait = WebDriverWait(driver, 5)
    return driver
 
 
def lookup(driver,executeScript):
    driver.get('https://www.youtube.com/watch?v=_ndPr4k9ZSQ&t=243s')
    try:
        while(True):
            time.sleep(3)# google podumaet bot i zabanit tvoj IP... mozhet byt' takoe TOCHNO NE UVEREN
            button = driver.wait.until(EC.presence_of_element_located(((By.XPATH, u"//div[@id='comment-section-renderer']/button"))))
            if button.is_enabled():
                button.click()
            currentSiteName=driver.title
    except TimeoutException:
        print("should print now")
        driver.execute_script(executeScript)# Commenty na comment ne zagruzhayutsya vmeste nuzhno podgruzhat potom jetot javascript vyzovet potom i podgruzit
        time.sleep(40)

        content= driver.page_source
        soup=BeautifulSoup(content, 'html.parser')
        html = soup.prettify("utf-8")
        print("prittified")
        with open("output1.html", "wb") as file:
            file.write(html)
        print("fileCreated")

if __name__ == "__main__":
    javascriptExecutable=";"
    with open("JavaScriptCommentLoader.txt") as f: 
        for line in f:
            javascriptExecutable+=line
    driver = init_driver()
    lookup(driver,javascriptExecutable)
    time.sleep(5)
    driver.quit()