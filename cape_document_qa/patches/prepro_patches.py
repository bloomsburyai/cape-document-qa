import re
from docqa.data_processing import text_utils


text_utils.space_re = re.compile("[ \u202f\n]")
