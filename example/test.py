import os

from farlog import getLogger

logger = getLogger(__name__)
logger.info("info 信息")

url = "http://www.**.net/images/logo.gif"
filename = os.path.basename(url)
logger.info(filename)
