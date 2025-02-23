import logging

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",  
)

# Create a global logger instance
logger = logging.getLogger("app_logger")
