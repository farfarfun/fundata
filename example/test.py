import logging
import os

logging.info("info 信息")

url = "http://www.**.net/images/logo.gif"
filename = os.path.basename(url)
print(filename)
