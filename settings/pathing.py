import platform
import re

def os_parse_path(path):
    if(platform.system() is 'Windows'):
        path = re.sub(r'(?is)\\', '/', path)
    else:
        path = re.sub(r'(?is)/', '\\', path)
    return path
