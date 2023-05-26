from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time
from typing import Literal, Mapping

Data_Type = Literal['domestic', 'interconnection']

X_PATHS: Mapping[str, str] = {
    'from_date': '//*[@id="ctl00_ctl00_Content_ChildContentLeft_PeriodControl_PeriodFromDatePicker"]',
    'to_date': '//*[@id="ctl00_ctl00_Content_ChildContentLeft_PeriodControl_PeriodToDatePicker"]',
    'load_button': '//*[@id="ctl00_ctl00_Content_LoadDataButton2"]',
    'save_icon': '//*[@id="ctl00_ctl00_Content_ReportViewerControl_ctl05_ctl04_ctl00_ButtonLink"]',
    'excel_type': '//*[@id="ctl00_ctl00_Content_ReportViewerControl_ctl05_ctl04_ctl00_Menu"]/div[2]/a'
}

DEPENDING_X_PATHS: Mapping['Data_Type', Mapping[str, str]] = {
    'domestic': {
        'points_type': '//*[@id="main-content"]/table/tbody/tr/td[2]/section',
        'granularity': '//*[@id="ctl00_ctl00_Content_ChildContentLeft_GranularityControl_GranularityRadioButtonList_1"]'
    },
    'interconnection': {
        'points_type': '//*[@id="main-content"]/table/tbody/tr/td[1]/section',
        'granularity': '//*[@id="ctl00_ctl00_Content_ChildContentLeft_GranularityRadioButtonList_1"]'
    }
}


def create_driver(path_to_save: str):
    """
    function to create web driver with chrome options
    """
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) ' \
                 'AppleWebKit/537.36 (KHTML, like Gecko) ' \
                 'Chrome/33.0.1750.517 ' \
                 'Safari/537.36'
    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument('user-agent={0}'.format(user_agent))
    chrome_options.add_experimental_option("prefs", {
        "profile.default_content_settings.cookies": 2,  # disable cookies
        "profile.default_content_settings.geolocation": 2,  # disable geolocation
        "profile.managed_default_content_settings.images": 2,  # disable image loading
        "download.default_directory": path_to_save  # set default directory
    })
    driver = webdriver.Chrome(options=chrome_options)
    return driver


def collect_fluxys_data(data_type: Data_Type, from_date: str, to_date: str, file_name: str):
    """
    function to collect data from fluxys with selenium for the set date range
    """
    # create driver
    driver = create_driver(os.getcwd())
    driver.get('https://gasdata.fluxys.com/en/transmission-ztp-trading-services/flow-data/')

    wait = WebDriverWait(driver, 15)

    # choose nominations and flows
    driver.find_element(By.XPATH, DEPENDING_X_PATHS[data_type]['points_type']).click()

    load_button = wait.until(
        EC.element_to_be_clickable((By.XPATH, X_PATHS['load_button'])))

    # set from date
    from_ = driver.find_element(By.XPATH, X_PATHS['from_date'])
    from_.clear()
    from_.send_keys(from_date)

    # set to date
    to_ = driver.find_element(By.XPATH, X_PATHS['to_date'])
    to_.clear()
    to_.send_keys(to_date)

    # choose granularity Daily
    driver.find_element(By.XPATH, DEPENDING_X_PATHS[data_type]['granularity']).click()

    # click button Load
    load_button.click()
    time.sleep(5)

    # click on save icon
    # save_icon = wait.until(
    #     EC.element_to_be_clickable((By.XPATH, X_Paths['save_icon'])))
    # save_icon.click()
    # time.sleep(1)
    driver.find_element(By.XPATH, X_PATHS['save_icon']).click()

    # click on Excel data type
    excel_type = wait.until(
        EC.element_to_be_clickable((By.XPATH, X_PATHS['excel_type'])))
    excel_type.click()
    # driver.find_element(By.XPATH, X_Paths['excel_type']).click()

    # waiting to save and close driver
    while not os.path.exists(file_name):
        time.sleep(1)
    driver.close()
