if __name__ == "__main__":
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service

    # service = Service(executable_path='./chromedriver.bin')
    # options = webdriver.ChromeOptions()
    # driver = webdriver.Chrome(service=service, options=options)
    # # ...
    # driver.quit()
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.binary_location = "/home/kosenko/verbalist/verbalist/datasets/openreview/chrome-linux/chrome"  # chrome binary location specified here
    options.add_argument("--start-maximized")  # open Browser in maximized mode
    options.add_argument("--no-sandbox")  # bypass OS security model
    options.add_argument(
        "--disable-dev-shm-usage"
    )  # overcome limited resource problems
    # options.add_experimental_option("excludeSwitches", ["enable-automation"])
    # options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(
        options=options,
        executable_path="/home/kosenko/verbalist/verbalist/datasets/openreview/chromedriver",
    )
    driver.get("http://google.com/")
