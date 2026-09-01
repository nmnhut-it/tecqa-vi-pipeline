"""
Stage-1 prompts for TECQA (English and Vietnamese cross-lingual adaptation).

Verbatim reproduction from paper Appendix F.3 for English:
  - Prompt 1: Entity set extraction
  - Prompt 2: Relation set extraction
  - Prompt 3: Main entity identification

Option B Cross-lingual prompts for Vietnamese:
  - Extracts entities and maps role/noun phrases to English ICEWS schema entities
    (e.g., 'công dân Úc' -> 'Citizen (Australia)', 'phiến quân Philippines' -> 'Rebel (Philippines)')
"""

# ===========================================================================
# ENGLISH PROMPTS (Paper Appendix F.3)
# ===========================================================================

ENTITY_EXTRACTION_PROMPT = """\
Extract core topic entities (people, places, orgs) from the question \
as a Python list. Use double quotes if a string contains an apostrophe \
(e.g., "Xi’an’s"), otherwise use single quotes. Exclude generic terms \
with "country". Output ONLY the list (e.g., ['A', "B's"]) without \
explanation. Return [] if empty.
Examples:
Question: After the Danish Ministry of Defence and Security, who was \
the first to visit Iraq?
response: ['Iraq', 'the Danish Ministry of Defence and Security']
Question: When did the al-Shabaab insurgency use unconventional \
violence against Muslims in the United Kingdom?
response: ['al-Shabaab insurgency', 'Muslims in the United Kingdom']
Question: To whom did John Dramani Mahama make an appeal after 2010?
response: ['John Dramani Mahama']
Question: {question}
response:"""


RELATION_EXTRACTION_PROMPT = """\
Select exactly ONE relation from the Relation Set that best matches \
the core action in the Question, using the Entities List for context. \
Output ONLY the exact relation name.
Mapping Rules: wish/plan/aim → "Express intent..."; ask/urge/call \
on → "Appeal for..."; negotiate/talks → "Engage in negotiation..." \
or "Express intent to meet..."; visit/travel → "Make/Host a visit"; \
attack/force → "Use ... force"; criticize → "Make a statement..."
Examples:
Q: After the Danish Ministry..., who was the first to visit Iraq?
E: ["the Danish Ministry...", "Iraq"]
A: Make a visit
Q: Before China, with whom did Spain wish to cooperate economically?
E: ["Spain", "China"]
A: Express intent to cooperate economically
Q: Who did Greece appeal to for humanitarian aid in 2015?
E: ["Greece"]
A: Appeal for humanitarian aid
Q: {question}
E: {entities_list}
Relation Set: {relation_set}
A:"""


MAIN_ENTITY_PROMPT = """\
Task: Identify the single Main Entity from the list. Output ONLY the \
exact name.
Selection Logic: 1. If the grammatical subject is a specific entity \
(e.g., "Kitti..."), select the Subject.
2. If the subject is interrogative (e.g., "Who/What"), select the focus Object.
Examples:
Q: After the Danish Ministry..., who was the first to visit Iraq?
L: ["Iraq", "Defense / Security Ministry (Denmark)"]
A: Iraq (Subject is "who" → pick Object)
Q: When did Kitti Wasinondh last negotiate with Thailand?
L: ["Kitti Wasinondh", "Thailand"]
A: Kitti Wasinondh (Specific Subject → pick Subject)
Q: With whom did Catherine Ashton last wish to meet before Cambodia?
L: ["Catherine Ashton", "Cambodia"]
A: Catherine Ashton (Specific Subject → pick Subject)
Q: {question}
L: {entities_list}
A:"""


# ===========================================================================
# VIETNAMESE PROMPTS (Option B: Cross-lingual Entity Mention Extraction)
# ===========================================================================

ENTITY_EXTRACTION_VI_PROMPT = """\
Nhiệm vụ: Trích xuất các thực thể chính (nhân vật, tổ chức, quốc gia, nhóm đối tượng) \
từ câu hỏi tiếng Việt VÀ dịch/chuẩn hóa tên thực thể sang tiếng Anh theo định dạng \
chuẩn của Knowledge Graph ICEWS (ví dụ: 'công dân Úc' -> 'Citizen (Australia)', \
'phiến quân Philippines' / 'người nổi loạn Thái Lan' -> 'Rebel (Philippines)' / 'Rebel (Thailand)', \
'nhà thuyết giáo Hồi giáo của Iran' -> 'Muslim Cleric (Iran)', \
'chính phủ Đức' -> 'Government (Germany)', 'bộ quốc phòng Đan Mạch' -> 'Defense / Security Ministry (Denmark)', \
'Thống đốc Thái Lan' -> 'Governor (Thailand)', 'lãnh đạo Mông Cổ' / 'Thủ tướng Ấn Độ' -> 'Head of Government (Mongolia)' / 'Head of Government (India)', \
'báo Singapore' -> 'News Personnel (Singapore)', 'nhà ngoại giao Zambia' -> 'Diplomat (Zambia)', \
'Trung Quốc' -> 'China', 'Mỹ' / 'Quốc hội Mỹ' -> 'United States' / 'Congress (United States)', \
'Campuchia' -> 'Cambodia', 'Thái Lan' -> 'Thailand', 'Canada' -> 'Canada', 'Oman' -> 'Oman').

Quy tắc quan trọng:
1. Khi câu hỏi nhắc đến quốc gia chung (ví dụ: 'Canada', 'Oman', 'Việt Nam', 'Trung Quốc'), luôn xuất tên quốc gia chuẩn ('Canada', 'Oman', 'Vietnam', 'China'), không gán sang nhóm phụ.
2. Khi câu hỏi nhắc đến vai trò gắn với quốc gia (ví dụ: 'công dân Úc', 'phiến quân Philippines', 'Thống đốc Thái Lan', 'lãnh đạo Mông Cổ', 'báo Singapore'), xuất dạng Role (Country) ('Citizen (Australia)', 'Rebel (Philippines)', 'Governor (Thailand)', 'Head of Government (Mongolia)', 'News Personnel (Singapore)').

Chỉ xuất định dạng Python list các tên thực thể tiếng Anh (ví dụ: ['China', 'Barack Obama']). \
Không giải thích gì thêm. Trả về [] nếu không có.

Ví dụ:
Câu hỏi: Sau bộ quốc phòng Đan Mạch, ai là người đầu tiên đến thăm Iraq?
response: ['Iraq', 'Defense / Security Ministry (Denmark)']
Câu hỏi: Lúc nào Mallam Isa Yuguda đi thăm Ethiopia?
response: ['Mallam Isa Yuguda', 'Ethiopia']
Câu hỏi: Trước năm 2006, ai là công dân Úc bị bắt tại Thái Lan?
response: ['Citizen (Australia)', 'Thailand']
Câu hỏi: Okada Katsuya đã đến Thái Lan lần đầu tiên vào năm nào?
response: ['Okada Katsuya', 'Thailand']
Câu hỏi: Ai là nhà thuyết giáo Hồi giáo của Iran đến thăm Iraq vào năm 2008?
response: ['Muslim Cleric (Iran)', 'Iraq']
Câu hỏi: Người cuối cùng nào đã chỉ trích Thái Lan trước khi Thống đốc Thái Lan?
response: ['Thailand', 'Governor (Thailand)']
Câu hỏi: Sau báo Singapore, nước nào là nước đầu tiên lên án Trung Quốc?
response: ['News Personnel (Singapore)', 'China']
Câu hỏi: Năm nào phiến quân của Philippines lần cuối đến Malaysia?
response: ['Rebel (Philippines)', 'Malaysia']
Câu hỏi: Sau khi Thái Lan bị chỉ trích, người nào đã chỉ trích người nổi loạn ở Thái Lan?
response: ['Thailand', 'Rebel (Thailand)']
Câu hỏi: Iraq đã hợp tác với Hội đồng cao cấp về người tị nạn từ khi nào?
response: ['Iraq', 'High Commissioner for Refugees']
Câu hỏi: {question}
response:"""


RELATION_EXTRACTION_VI_PROMPT = """\
Chọn chính xác MỘT quan hệ từ Relation Set phù hợp nhất với hành động chính trong \
câu hỏi tiếng Việt, sử dụng Entities List làm ngữ cảnh.
Chỉ trả về tên quan hệ tiếng Anh chính xác từ Relation Set.

Quy tắc ánh xạ hành động:
- mong muốn đến thăm/muốn gặp/dự định gặp/muốn thương lượng → "Express intent to meet or negotiate"
- mong muốn/dự định/kế hoạch/thể hiện ý định hợp tác → "Express intent to cooperate..."
- kêu gọi/yêu cầu/thỉnh cầu viện trợ hoặc hỗ trợ → "Appeal for..." hoặc "Make an appeal or request"
- yêu cầu cuộc họp/muốn gặp gỡ → "Express intent to meet or negotiate"
- thương lượng/đàm phán/hội đàm → "Engage in negotiation"
- thăm/đến thăm/tiếp đón thực tế → "Make a visit" hoặc "Host a visit"
- ký kết thỏa thuận/hiệp ước → "Sign formal agreement"
- điều tra/thanh tra → "Investigate"
- tấn công/dùng vũ lực/bạo lực → "Use ... force"
- phê phán/chỉ trích/đổ lỗi/tuyên bố phản đối → "Criticize or denounce" hoặc "Make a statement"
- khen ngợi/tán thành/ủng hộ → "Praise or endorse"

Ví dụ:
Q: Sau bộ quốc phòng Đan Mạch, ai là người đầu tiên đến thăm Iraq?
E: ["Defense / Security Ministry (Denmark)", "Iraq"]
A: Make a visit
Q: Vào ngày 13-11-2005, quốc gia nào muốn đến thăm Trung Quốc?
E: ["China"]
A: Express intent to meet or negotiate
Q: Trước Trung Quốc, Tây Ban Nha muốn hợp tác kinh tế với ai?
E: ["Spain", "China"]
A: Express intent to cooperate economically
Q: Ai đã kêu gọi Hy Lạp viện trợ nhân đạo vào năm 2015?
E: ["Greece"]
A: Appeal for humanitarian aid
Q: {question}
E: {entities_list}
Relation Set: {relation_set}
A:"""


MAIN_ENTITY_VI_PROMPT = """\
Nhiệm vụ: Xác định duy nhất MỘT thực thể chính (Main Entity) từ danh sách thực thể \
làm tâm điểm cho đồ thị con. Chỉ xuất chính xác tên thực thể tiếng Anh.

Quy tắc chọn:
1. Nếu chủ ngữ trong câu là một thực thể cụ thể (ví dụ: 'Okada Katsuya...', 'Trung Quốc...'), chọn Chủ ngữ đó.
2. Nếu chủ ngữ là từ để hỏi (ví dụ: 'Ai', 'Quốc gia nào', 'Nước nào'), chọn thực thể đóng vai trò Tân ngữ/Đối tượng được hướng tới.

Ví dụ:
Q: Sau bộ quốc phòng Đan Mạch, ai là người đầu tiên đến thăm Iraq?
L: ["Iraq", "Defense / Security Ministry (Denmark)"]
A: Iraq
Q: Khi nào Kitti Wasinondh lần cuối cùng thương lượng với Thái Lan?
L: ["Kitti Wasinondh", "Thailand"]
A: Kitti Wasinondh
Q: Catherine Ashton muốn gặp ai lần cuối trước Campuchia?
L: ["Catherine Ashton", "Cambodia"]
A: Catherine Ashton
Q: {question}
L: {entities_list}
A:"""
