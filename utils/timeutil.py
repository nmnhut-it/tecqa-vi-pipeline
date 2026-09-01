"""
Timestamp parsing and explicit temporal-anchor extraction (TECQA paper
Sec 4.3.1), shared by every stage, both languages and the offline scorer.

OWNER: CODE (docs/TEAM_PLAN.md H5).

Deliberately dependency-free: tecqa/eval/metrics.py and the notebook's replay
mode import this on machines with no API key, no torch and no numpy.

This is the ONLY anchor extractor. Stage 2 builds T_exp with it and the scorer
derives the gold anchor with it. If those were two different regex sets, Anchor
Recall would be measuring the gap between the two sets rather than the quality
of retrieval.

BEHAVIOUR NOTE (docs/EVAL_DESIGN.md Sec 9.1): one date mention yields ONE
anchor. Applying the day, month and year regexes independently — as the earlier
Stage-2 implementation did — turned "ngay 8 thang 5 nam 2013" into three anchors
(the day, the 1st of May, and New Year), which inflated the evidence chain to
3*K facts and dragged the anchor toward the start of the month and year. The
paper's Eq. T_exp <- ExtractRegex(q) wants the normalized ISO date of each
mention, so longest/most-specific match wins here.

Input:  MultiTQ/CronQuestions timestamp strings ("2013", "2013-05", "2013-05-08")
        and question text in Vietnamese or English.
Output: datetime.date objects.

Related: tecqa/stages/stage2_chain.py (T_exp), tecqa/eval/variants.py
(gold_anchors), tecqa/eval/metrics.py (anchor recall).
"""
import re
from datetime import date

LANG_VI = "vi"
LANG_EN = "en"
LANGS = (LANG_VI, LANG_EN)

DEFAULT_MONTH = 1
DEFAULT_DAY = 1

GRANULARITY_YEAR = "year"
GRANULARITY_MONTH = "month"
GRANULARITY_DAY = "day"
# Number of "-"-separated parts an ISO timestamp has at each granularity.
_PARTS_PER_GRANULARITY = {GRANULARITY_YEAR: 1, GRANULARITY_MONTH: 2, GRANULARITY_DAY: 3}
_GRANULARITY_BY_PARTS = {n: g for g, n in _PARTS_PER_GRANULARITY.items()}

_EN_MONTHS = {name.lower(): number for number, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}
_EN_MONTH_ALT = "|".join(_EN_MONTHS)

_VI_FULL_RE = re.compile(
    r"(?:ngày\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)
_VI_MONTH_RE = re.compile(r"tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)
_VI_YEAR_RE = re.compile(r"năm\s+(\d{4})", re.IGNORECASE)
# Day-first numeric dates. The dot separator matters: the translated corpus
# renders "14.12.2008" that way, and without it the date fell through to the
# bare-year pattern, anchoring the evidence chain on 1 January of that year.
_VI_DMY_NUM_RE = re.compile(r"\b(?:ngày\s+)?(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b",
                            re.IGNORECASE)

_EN_FULL_RE = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + _EN_MONTH_ALT + r"),?\s+(\d{4})", re.IGNORECASE)
_EN_MDY_RE = re.compile(
    r"(" + _EN_MONTH_ALT + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.IGNORECASE)
_EN_MONTH_RE = re.compile(r"(" + _EN_MONTH_ALT + r")\s+(\d{4})", re.IGNORECASE)
_EN_YEAR_RE = re.compile(r"\b(\d{4})\b")

_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})(?:-(\d{1,2}))?\b")
_INTERVAL_RE = re.compile(
    r"\b(?:between|from|từ)\s+(\d{4})\s+(?:and|to|đến)\s+(\d{4})\b",
    re.IGNORECASE)


def safe_date(year: int, month: int, day: int) -> date:
    """Calendar-invalid combinations (31 February) fall back to the 1st rather
    than raising — MultiTQ only ever needs relative distance between facts."""
    try:
        return date(year, month, day)
    except ValueError:
        return date(year, month, DEFAULT_DAY)


def parse_ts(ts: str) -> date:
    """Normalize "YYYY" / "YYYY-MM" / "YYYY-MM-DD" to a date. Missing month and
    day default to 1, which is what the paper's distance arithmetic assumes."""
    parts = str(ts).split("-")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else DEFAULT_MONTH
    day = int(parts[2]) if len(parts) > 2 else DEFAULT_DAY
    return safe_date(year, month, day)


def granularity_of(ts: str) -> str:
    """day / month / year, read off how many components the timestamp has."""
    return _GRANULARITY_BY_PARTS.get(len(str(ts).split("-")), GRANULARITY_DAY)


def trim_to_granularity(ts: str, granularity: str) -> str:
    """"2013-05-08" at year granularity -> "2013". Used only by the diagnostic
    hit_gran metric; the headline Hits@1 stays an exact string match."""
    keep = _PARTS_PER_GRANULARITY.get(granularity, _PARTS_PER_GRANULARITY[GRANULARITY_DAY])
    return "-".join(str(ts).split("-")[:keep])


# What unit the question wants back. Both languages in one place because Stage 3
# trims its answer with this and the error analysis measures against it; two
# copies would drift and the diagnostic would then be scoring its own regexes.
_ASKS_YEAR_RE = re.compile(
    r"(?:what|which)\s+year|year\s+was|năm\s+nào|năm\s+mấy|năm\s+đầu|năm\s+cuối",
    re.IGNORECASE)
_ASKS_MONTH_RE = re.compile(
    r"(?:what|which|specific|exact)\s+month|month\s+(?:did|was)"
    r"|tháng\s+(?:nào|mấy|chính\s+xác|cụ\s+thể)",
    re.IGNORECASE)


def asked_granularity(text: str) -> str:
    """The unit the question asks for: 'year', 'month', or None for a full date.

    Month is tested first: "in what month of what year" asks for a month, and
    the year pattern would otherwise win on ordering alone.
    """
    if _ASKS_MONTH_RE.search(text):
        return GRANULARITY_MONTH
    if _ASKS_YEAR_RE.search(text):
        return GRANULARITY_YEAR
    return None


def _vi_full(m):
    day, month, year = m.groups()
    return safe_date(int(year), int(month), int(day))


def _vi_month(m):
    month, year = m.groups()
    return safe_date(int(year), int(month), DEFAULT_DAY)


def _vi_dmy_num(m):
    day, month, year = m.groups()
    return safe_date(int(year), int(month), int(day))


def _year_only(m):
    return safe_date(int(m.group(1)), DEFAULT_MONTH, DEFAULT_DAY)


def _en_full(m):
    day, month, year = m.groups()
    return safe_date(int(year), _EN_MONTHS[month.lower()], int(day))


def _en_mdy(m):
    month, day, year = m.groups()
    return safe_date(int(year), _EN_MONTHS[month.lower()], int(day))


def _en_month(m):
    month, year = m.groups()
    return safe_date(int(year), _EN_MONTHS[month.lower()], DEFAULT_DAY)


def _iso(m):
    year, month, day = m.groups()
    return safe_date(int(year), int(month), int(day) if day else DEFAULT_DAY)


def _interval(m):
    """"between 2008 and 2012" is one mention naming two endpoints; the paper's
    before/after questions need both to bound the window."""
    start, end = m.groups()
    return [safe_date(int(start), DEFAULT_MONTH, DEFAULT_DAY),
            safe_date(int(end), DEFAULT_MONTH, DEFAULT_DAY)]


# Ordered most specific first. An earlier pattern consumes its span, so a full
# date is never re-counted as a bare month or year. Both languages share one
# ordered list: a Vietnamese question never contains "May 2013" and an English
# one never contains "thang", so there is no need to branch — and sharing the
# list is what guarantees the two conditions are measured identically.
_PATTERNS = (
    (_INTERVAL_RE, _interval),
    (_VI_FULL_RE, _vi_full),
    (_VI_DMY_NUM_RE, _vi_dmy_num),
    (_ISO_RE, _iso),
    (_EN_FULL_RE, _en_full),
    (_EN_MDY_RE, _en_mdy),
    (_VI_MONTH_RE, _vi_month),
    (_EN_MONTH_RE, _en_month),
    (_VI_YEAR_RE, _year_only),
    (_EN_YEAR_RE, _year_only),
)


def _overlaps(span, consumed) -> bool:
    start, end = span
    return any(start < taken_end and taken_start < end for taken_start, taken_end in consumed)


def extract_explicit_anchors(text: str, lang: str = None) -> list:
    """T_exp (Sec 4.3.1): every date mentioned in the question, normalized to a
    date object, each mention counted ONCE. Returns [] when the question names
    no date — the caller then falls back to implicit anchors (Eq. 6).

    `lang` is accepted and ignored: the pattern list covers both languages. It
    stays in the signature because callers pass it and because any future
    language-specific rule belongs here rather than at the call sites.
    """
    anchors, consumed = [], []
    for pattern, to_date in _PATTERNS:
        for match in pattern.finditer(text):
            if _overlaps(match.span(), consumed):
                continue
            found = to_date(match)
            anchors.extend(found if isinstance(found, list) else [found])
            consumed.append(match.span())
    # Preserve first-seen order while dropping repeats of the same calendar day.
    return list(dict.fromkeys(anchors))
