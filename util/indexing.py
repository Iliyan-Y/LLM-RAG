from datetime import datetime
from typing import Dict, Optional
import re

def _parse_date(text: str) -> Optional[str]:
    """
    Parse a date string into ISO format (YYYY-MM-DD).
    Tries multiple common SEC date formats. Returns None if no parse.
    """
    if not text:
        return None
    text = text.strip().replace(" ,", ",")
    # Remove trailing punctuation
    text = re.sub(r"[\.,;:\)]$", "", text)
    fmts = [
        "%B %d, %Y",     # March 31, 2023
        "%b %d, %Y",     # Mar 31, 2023
        "%m/%d/%Y",      # 03/31/2023
        "%Y-%m-%d",      # 2023-03-31
        "%Y/%m/%d",      # 2023/03/31
        "%d %B %Y",      # 31 March 2023
        "%d %b %Y",      # 31 Mar 2023
    ]
    for f in fmts:
        try:
            return datetime.strptime(text, f).date().isoformat()
        except Exception:
            continue
    # Try to isolate a Month Day, Year pattern inside a longer string
    m = re.search(
        r"([A-Z][a-z]+ \d{1,2}, \d{4}|[A-Z][a-z]{2} \d{1,2}, \d{4}|\d{1,2} [A-Z][a-z]+ \d{4})",
        text,
    )
    if m:
        return _parse_date(m.group(1))
    return None


def _infer_quarter_from_month(month: int) -> Optional[int]:
    if 1 <= month <= 3:
        return 1
    if 4 <= month <= 6:
        return 2
    if 7 <= month <= 9:
        return 3
    if 10 <= month <= 12:
        return 4
    return None


def extract_sec_metadata(sample_text: str, file_name: str, dir_name: str) -> Dict:
    """
    Extract SEC filing metadata heuristically from the first pages of the PDF text.
    - filing_type: 10-K or 10-Q if present
    - period_end_date: for the fiscal/quarterly period ended ...
    - filing_date: 'Filed' date if present
    - company: inferred from parent folder (dir_name)
    - year/quarter: derived from period_end_date if possible
    """
    text = sample_text or ""
    # Normalize whitespace for simpler regex
    norm = re.sub(r"\s+", " ", text)

    # Filing type
    filing_type = None
    m = re.search(r"\bForm\s+(10-[KQ])\b", norm, re.IGNORECASE)
    if m:
        filing_type = m.group(1).upper()
    else:
        m = re.search(r"\b(10-[KQ])\b", norm, re.IGNORECASE)
        if m:
            filing_type = m.group(1).upper()

    # Period end date
    period_end_date = None
    candidates = [
        r"for the fiscal year ended\s+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
        r"for the quarterly period ended\s+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
        r"quarter ended\s+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
        r"year ended\s+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
        r"ended\s+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
    ]
    for pat in candidates:
        m = re.search(pat, norm, re.IGNORECASE)
        if m:
            period_end_date = _parse_date(m.group(1))
            if period_end_date:
                break

    # Filing date
    filing_date = None
    candidates_filed = [
        r"\bFiled\s+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b",
        r"\bDate of filing[:\s]+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b",
        r"\bFiling Date[:\s]+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b",
    ]
    for pat in candidates_filed:
        m = re.search(pat, norm, re.IGNORECASE)
        if m:
            filing_date = _parse_date(m.group(1))
            if filing_date:
                break

    # Derive year and quarter from period_end_date (preferred) or filing_date
    year = None
    quarter = None
    date_for_quarter = period_end_date or filing_date
    if date_for_quarter:
        try:
            dt = datetime.strptime(date_for_quarter, "%Y-%m-%d").date()
            year = dt.year
            quarter = _infer_quarter_from_month(dt.month)
        except Exception:
            pass

    # Company and category
    company = dir_name or None
    category = dir_name or None

    return {
        "filing_type": filing_type,
        "period_end_date": period_end_date,
        "filing_date": filing_date,
        "year": year,
        "quarter": quarter,
        "company": company,
        "category": category,
        "source_file": file_name,
    }
