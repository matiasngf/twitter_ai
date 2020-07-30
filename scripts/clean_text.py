import re
from emoji import UNICODE_EMOJI

def del_repeated(char, s, ):
    return re.sub(rf'\{char}[{char} ]*', ' '+char+' ', s)

def space_emogis(s):
    return re.sub("(["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "])", r' \1 ', s, flags=re.UNICODE)

def clean(s):
    s = space_emogis(s)
    s = re.sub(r'([@#])', r' \1 ', s.lower())
    s = re.sub('\n', ' ', s)
    s = del_repeated('?', s)
    s = del_repeated('!', s)
    s = del_repeated('¡', s)
    s = del_repeated('¿', s)
    s = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]+\.[a-z]+\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', 'URL', s)
    s = re.sub(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?', 'URL', s)
    regrex_filter = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\w+"
        u"?!@#"
    "]+", flags = re.UNICODE)
    s = ' '.join(regrex_filter.findall(s))
    s = re.sub(r'[ ]+', ' ', s)
    return s
