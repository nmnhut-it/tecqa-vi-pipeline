"""
Stage-3 reasoning prompts for TECQA, in two styles and two languages.

OWNER: CODE (docs/TEAM_PLAN.md H5).

`paper` reproduces the published setup: the English prompt is verbatim from
Appendix F.3 Prompt 4, the Vietnamese one is Hung's cross-lingual adaptation.
It is the baseline and must not be edited — the reproduction claim rests on it.

`strict` is our own, matched rule-for-rule across the two languages so a
VI-vs-EN comparison measures the language and not the prompt (docs/EVAL_DESIGN.md
Sec 9.3). It keeps the published rules and tightens two of them:

  Rule 2 — copy the chosen fact's date verbatim, shortening only when the
  question asks for a year or a month. The model was reporting '2010' where the
  gold answer was '2010-06-23'.

  Rule 3 — filter by the relation the question names, and never answer with the
  reference entity the question hangs off.

It also drops the published prompt's offer of `[]` as a valid answer, because
under Hits@1 an empty answer is scored wrong with certainty.

DO NOT put "guess anyway" in the main template. We tried exactly that — a fifth
rule saying to drop constraints one at a time and return the most plausible
candidate — and on a 12-question pilot it fixed one answer, broke two, and still
abstained twice. The model read it as blanket permission to ignore the relation
or the time window, which is what makes an answer right. Guessing belongs in
FALLBACK_SUFFIX below, which is only ever appended after an empty answer has
already been produced and can therefore not make a correct answer worse.

Input:  a style and a language code. Output: a template with the fields
question / answer_type / topk / grounded_entities / facts.
"""

PROMPT_PAPER = "paper"
PROMPT_STRICT = "strict"
PROMPT_STYLES = (PROMPT_PAPER, PROMPT_STRICT)
DEFAULT_STYLE = PROMPT_PAPER

LANG_EN = "en"
LANG_VI = "vi"


QA_REASONING_PROMPT = """\
Task: Answer Q using ONLY provided Facts. Output strictly a Python list \
string (e.g., ['A', '2024']) with NO extra text. Return [] if empty.
Execution Rules:
1. Data: Facts are [sub, rel, obj, date]. Ignore invalid dates if \
temporal reasoning is needed.
2. Type: If 'time', format as YYYY, YYYY-MM, or YYYY-MM-DD to match \
Q granularity (e.g. 'In what year' -> YYYY, 'In what month' -> YYYY-MM). \
If 'entity', extract sub or obj based on Q direction (e.g., Who acted? → Sub).
3. Logic: Filter facts by time constraints (before/after). For "first" \
→ sort earliest; "last" → sort latest.
4. Final: Deduplicate answers, preserve temporal order, apply Top K.
Input:
Q: {question}
Type: {answer_type} | Top K: {topk}
Grounded KG Entities: {grounded_entities}
Facts: {facts}
Response:"""


QA_REASONING_VI_PROMPT = """\
Nhiệm vụ: Trả lời câu hỏi Q bằng tiếng Việt chỉ dựa trên danh sách Facts tiếng Anh được cung cấp. \
Xuất ĐÚNG định dạng chuỗi Python list (ví dụ: ['Iran'] hoặc ['2014'] hoặc ['2012-04-27']) và KHÔNG có văn bản thừa nào khác. \
Trả về [] nếu không có dữ liệu phù hợp.

Quy tắc thực hiện:
1. Dữ liệu: Mỗi Fact có dạng [subject, relation, object, date]. \
Sử dụng 'Grounded KG Entities' để liên hệ các từ ngữ trong câu hỏi với thực thể tiếng Anh trong Facts (ví dụ: 'Thống đốc Thái Lan' -> Governor (Thailand), 'Bộ Quốc phòng Hoa Kỳ' -> Defense / Security Ministry (United States)).
2. Loại đáp án (Type):
   - Nếu là 'time':
     * Câu hỏi hỏi "năm nào" / "năm mấy" → định dạng đúng 'YYYY' (ví dụ: ['2010']).
     * Câu hỏi hỏi "tháng nào" → định dạng 'YYYY-MM' (ví dụ: ['2005-01']).
     * Câu hỏi hỏi "ngày nào" / "lúc nào" / "khi nào" → định dạng 'YYYY-MM-DD' (ví dụ: ['2007-09-20']).
   - Nếu là 'entity': trích xuất tên thực thể tiếng Anh chính xác từ Fact (sub hoặc obj tuỳ hướng câu hỏi).
3. Logic suy luận thời gian:
   - Bước 1 (Tìm mốc tham chiếu): Nếu câu hỏi có sự kiện mốc (ví dụ: "trước khi [A]...", "sau [B]..."), tìm mốc thời gian t_ref mà [A] hoặc [B] tham gia.
   - Bước 2 (Lọc & Loại trừ): Lọc các Facts diễn ra trước t_ref hoặc sau t_ref. TUYỆT ĐỐI KHÔNG chọn chính thực thể mốc [A]/[B] làm đáp án!
   - Bước 3 (Sắp xếp): Nếu yêu cầu "đầu tiên" -> chọn sự kiện sớm nhất (earliest); nếu "cuối cùng" / "gần đây nhất" -> chọn sự kiện muộn nhất (latest).
4. Giới hạn Top K: Lấy tối đa Top K kết quả theo thứ tự thời gian.

Đầu vào:
Q: {question}
Type: {answer_type} | Top K: {topk}
Grounded KG Entities: {grounded_entities}
Facts: {facts}
Response:"""


QA_REASONING_STRICT_PROMPT = """\
Task: Answer Q using ONLY the provided Facts. Output strictly a Python list \
string (e.g., ['Iran'] or ['2014'] or ['2012-04-27']) with NO extra text.

Execution Rules:
1. Data: every Fact is [subject, relation, object, date]. \
Use 'Grounded KG Entities' to link the wording of Q to the English entity names in the Facts \
(e.g. 'Governor of Thailand' -> Governor_(Thailand)).
2. Answer type:
   - If 'time': copy the date of the fact you chose EXACTLY as written.
     * Q asks "in what year" -> shorten to 'YYYY' (e.g. ['2010']).
     * Q asks "in what month" -> shorten to 'YYYY-MM' (e.g. ['2005-01']).
     * Q asks "on what day" / "when" -> keep the full 'YYYY-MM-DD' (e.g. ['2007-09-20']).
   - If 'entity': copy the entity name from the fact EXACTLY as written, choosing subject or object according to the direction of Q.
3. Temporal reasoning:
   - Step 1 (reference point): if Q names a reference event (e.g. "before [A]...", "after [B]..."), find the date t_ref at which [A] or [B] occurs.
   - Step 2 (filter): keep only the Facts before or after t_ref, AND only those whose relation is the one Q asks about. NEVER answer with the reference entity [A]/[B] itself.
   - Step 3 (order): "first" -> the earliest event; "last" / "most recent" -> the latest event.
4. Final: deduplicate answers, preserve temporal order, apply Top K.

Input:
Q: {question}
Type: {answer_type} | Top K: {topk}
Grounded KG Entities: {grounded_entities}
Facts: {facts}
Response:"""


QA_REASONING_STRICT_VI_PROMPT = """\
Nhiệm vụ: Trả lời câu hỏi Q chỉ dựa trên danh sách Facts tiếng Anh được cung cấp. \
Xuất ĐÚNG định dạng chuỗi Python list (ví dụ: ['Iran'] hoặc ['2014'] hoặc ['2012-04-27']) và KHÔNG có văn bản thừa nào khác.

Quy tắc thực hiện:
1. Dữ liệu: mỗi Fact có dạng [subject, relation, object, date]. \
Dùng 'Grounded KG Entities' để liên hệ từ ngữ trong Q với tên thực thể tiếng Anh trong Facts \
(ví dụ: 'Thống đốc Thái Lan' -> Governor_(Thailand)).
2. Loại đáp án:
   - Nếu là 'time': chép NGUYÊN VĂN mốc thời gian của fact đã chọn.
     * Q hỏi "năm nào" / "năm mấy" -> rút gọn thành 'YYYY' (ví dụ: ['2010']).
     * Q hỏi "tháng nào" -> rút gọn thành 'YYYY-MM' (ví dụ: ['2005-01']).
     * Q hỏi "ngày nào" / "lúc nào" / "khi nào" -> giữ đủ 'YYYY-MM-DD' (ví dụ: ['2007-09-20']).
   - Nếu là 'entity': chép NGUYÊN VĂN tên thực thể trong fact, chọn subject hay object tuỳ hướng của Q.
3. Suy luận thời gian:
   - Bước 1 (mốc tham chiếu): nếu Q có sự kiện mốc (ví dụ: "trước khi [A]...", "sau [B]..."), tìm mốc thời gian t_ref mà [A] hoặc [B] tham gia.
   - Bước 2 (lọc): chỉ giữ các Facts diễn ra trước hoặc sau t_ref, VÀ chỉ những Fact có đúng quan hệ mà Q hỏi. TUYỆT ĐỐI KHÔNG lấy chính thực thể mốc [A]/[B] làm đáp án.
   - Bước 3 (sắp xếp): "đầu tiên" -> sự kiện sớm nhất; "cuối cùng" / "gần đây nhất" -> sự kiện muộn nhất.
4. Kết thúc: khử trùng lặp, giữ thứ tự thời gian, áp dụng Top K.

Đầu vào:
Q: {question}
Type: {answer_type} | Top K: {topk}
Grounded KG Entities: {grounded_entities}
Facts: {facts}
Response:"""


TEMPLATES = {
    (PROMPT_PAPER, LANG_EN): QA_REASONING_PROMPT,
    (PROMPT_PAPER, LANG_VI): QA_REASONING_VI_PROMPT,
    (PROMPT_STRICT, LANG_EN): QA_REASONING_STRICT_PROMPT,
    (PROMPT_STRICT, LANG_VI): QA_REASONING_STRICT_VI_PROMPT,
}


# Appended to the SAME prompt, and only after the model has already answered
# with an empty list. That ordering is the whole point: the first pass stays as
# constrained as the published one, and permission to guess is granted only
# where the alternative is an answer Hits@1 has already scored wrong. It can
# therefore never turn a correct answer into a wrong one.
FALLBACK_SUFFIX_EN = """
Your previous answer was an empty list. An empty answer is always scored wrong, \
so it is never the best available answer. Re-read the Facts and give the single \
most plausible answer to Q, even if no Fact satisfies every constraint. \
Output only the Python list."""

FALLBACK_SUFFIX_VI = """
Câu trả lời trước của bạn là danh sách rỗng. Đáp án rỗng luôn bị tính là sai, \
nên nó không bao giờ là đáp án tốt nhất. Hãy đọc lại Facts và đưa ra MỘT đáp án \
hợp lý nhất cho Q, kể cả khi không Fact nào thoả mãn mọi ràng buộc. \
Chỉ xuất ra danh sách Python."""

FALLBACK_SUFFIXES = {LANG_EN: FALLBACK_SUFFIX_EN, LANG_VI: FALLBACK_SUFFIX_VI}


def fallback_suffix(language: str) -> str:
    """The retry instruction, in the language of the run."""
    return FALLBACK_SUFFIXES[LANG_VI if language == LANG_VI else LANG_EN]


def template_for(style: str, language: str) -> str:
    """Pick a Stage-3 template. Any language other than 'vi' reads as English,
    matching the rest of the pipeline, but an unknown STYLE raises: a typo
    there would silently score the wrong condition."""
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {style!r}; expected one of {PROMPT_STYLES}")
    return TEMPLATES[(style, LANG_VI if language == LANG_VI else LANG_EN)]
