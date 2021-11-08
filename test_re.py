import re
from bs4 import BeautifulSoup

trg = "d: fjldks) sdewdfds (test) I like this movie (dsjdlsjlfd"
if re.search('^[^\)]*:[^\)]*\)', trg) or re.search('\([\S]*$', trg):
        # print(f"idx:{idx}, src:{src} \n tgt:{trg}")
        trg = re.sub('^[^\)]*:[^\)]*\)', "", trg)
        trg = re.sub('\([^\)]*$', "", trg)
print(trg)

