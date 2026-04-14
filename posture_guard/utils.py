import sys
import os

def get_resource_path(filename: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    
    return os.path.join(os.path.abspath("."), filename)
