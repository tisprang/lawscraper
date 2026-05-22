import pandas as pd
import re

SOURCE_COLUMNS = ["cn_source", "el_source", "el2_source", "le_source"]
FINDLAW_PATTERN = re.compile(r"https?://(?:www\.)?codes\.findlaw\.com/([a-z]{2})/", re.IGNORECASE)

def _findlaw_url_or_none(value):
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url:
        return None
    if not url.lower().startswith(("http://", "https://")):
        return None
    if "findlaw.com" not in url.lower():
        return None
    return url.split("#", 1)[0].strip()

def _state_from_findlaw_url(url):
    match = FINDLAW_PATTERN.search(url)
    if not match:
        return None
    return match.group(1).lower()

def collect_jobs_from_excel(excel_path="usstates50.xlsx"):
    df = pd.read_excel(excel_path)
    jobs = []
    seen = set()
    for _, row in df.iterrows():
        for column in SOURCE_COLUMNS:
            url = _findlaw_url_or_none(row.get(column))
            if not url:
                continue
            state = _state_from_findlaw_url(url)
            if not state:
                continue
            dedupe_key = (state, url.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            jobs.append({"state": state, "url": url, "column": column})
    return jobs

jobs = collect_jobs_from_excel()
print(len(jobs))
print(jobs[:5])
