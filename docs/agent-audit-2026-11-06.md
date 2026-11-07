<div dir="rtl" align="right">

# בקרה הנדסית על מערך סוכני ה-AI של RoadGuard - 6.11.2026

דוח קריאה בלבד. לא שונה קוד, תצורה, prompt, DB או שירות. לא בוצע commit. כל מספר בדוח נגזר מנתונים קיימים: טבלת `hazards` על ה-VM (102 שורות, קריאה ב-`mode=ro`), קובצי ה-trace של סבבים r21, r22, r23, נקודות הקצה `/stats/*`, הקוד ב-main (1f16088) וארבעת הדפים בדפדפן (Chrome MCP). היכן שמדד לא ניתן לחישוב מהמידע הקיים, כתוב במפורש.

## 0. מה נבדק ועל מה

| מקור | תוכן | שימוש |
|---|---|---|
| `hazards` + `agent_trace_json` + `agent_value_json` (VM, 6.11 12:47 UTC) | 94 מקרי סבב r23 (5.11, build d8e93cd) + 9 דיווחים אמיתיים מהחשבונות benny ו-ovad (5.11 ערב, 6.11 צהריים) | כל מדדי הסוכנים, התקשורת, ה-ReAct, הסקנריו |
| `runs/round-r21/r22/r23-trace.txt` | אותן 94 תמונות בשלושה סבבים | עקביות בין ריצות זהות |
| `/stats/agents`, `/stats/kpi`, `/stats/ds_metrics?window_hours=48`, `/stats/public` | החלון של 24/48 שעות | אימות שהמדדים המוצגים תואמים ל-DB |
| `roadguard_agent.py`, `roadguard_api.py`, `webapp/assets/app.js`, `policy.yaml`, `docs/agents-architecture.md` | מימוש בפועל מול התכנון | סיבות שורש |
| Chrome: `orarr2.github.io/RoadGuard/#how`, `/roadguard/trace`, `/digest/latest`, `/roadguard/flow` (ובנוסף `/roadguard/supervisor`) | מה המשתמש והמפעיל רואים | סעיף 7 |
| ריצה חוזרת של `scripts/rag_relevance_eval.py` (קריאה בלבד, 6.11) | 16 זוגות class x road מול הקורפוס החי | סעיף 6 |

לא בוצע Replay מבודד של הצינור: אין סביבת בדיקה עם Ollama מחוץ ל-VM, והרצה על ה-VM יוצרת מקרים ב-DB הייצורי. הסקנריו של סעיף 8 מופו למקרים שכבר רצו.

## 1. מיפוי הסוכנים: תכנון מול ביצוע

הגרף (LangGraph): `sanitize -> signal_extractor -> supervisor -> {profile_builder | rag_retriever | decision} -> supervisor -> ... -> decision -> {approval_gate | await_evidence | END}`. ה-Supervisor הוא צומת ניתוב דטרמיניסטי (תקציב + `next_hop`) שמוסיף הערה אחת מ-LLM בביקור הראשון.

| סוכן | תפקיד מתוכנן (`docs/agents-architecture.md` טבלה 20) | מה הוא עושה בפועל (טרייסים + קוד) | קלט | פלט | LLM / כלים | מה הוא רשאי להחליט | כפילות / תרומה |
|---|---|---|---|---|---|---|---|
| `sanitize` | הגנת גבול על 3 ערוצי טקסט | כנ"ל, ובנוסף: זיהוי התמונה בקטלוג (sha256 ואז phash) ו"חתימת" known_image | location, ocr, reporter_text, sha256, phash | דגלי scrub, `known_image_shortcut`, `image_match_reason` | אין | חותם known_image (33/94) | הערך הגדול ביותר ליחידת זמן: 0.1 שניות, מכריע 35% מהמקרים |
| `signal_extractor` (Intake) | קורא את טקסט המדווח, מפיק claims וסתירות | כשאין טקסט: מדולג (94/94 ב-batch). כשיש טקסט (9 מקרים אמיתיים): 4 סבבי LLM, 14 מ-28 קריאות כלי נדחו, התוצאה הסופית הגיעה מכללי מילות מפתח ב-9/9 | reporter_text, yolo top1, form | `reporter_intake`, `reporter_signals` | llama3.2:3b; `extract_measurements`, `lookup_prior_reports` | שום החלטה; הסתירה היא אות ל-Decision | ה-LLM לא תרם דבר מעבר לכללים; כפילות מלאה עם ה-regex |
| `supervisor` | ניתוב + תקציב; הערת LLM "advisory" | ניתוב דטרמיניסטי עובד. ההערה הוחלפה ב-41/94 כי סתרה את הקלט. `force_action` לא הופעל ב-0/111 ריצות. `pipeline_concern` ו-`expected_difficulty` אינם נקראים על ידי אף רכיב | state, budget | `supervisor_plan` | qwen2.5:3b, קריאה אחת ~5-6 שניות (לא נספרת בעלות) | אין (AGENT_SUPERVISOR_ROUTING=0) | ה-LLM הוא קישוט לטרייס |
| `profile_builder` (Perception) | פרופיל מפגע + severity | החלק הדטרמיניסטי (lookup בקטלוג, severity מהאותות) הוא מה שעץ הכללים צורך. ה-LLM (2 קריאות, 51 שניות חציון) הציע שינוי severity שנחסם ב-63% מהריצות; נותרו ממנו 9 הורדות confidence ושינוי severity אחד | yolo, image_quality, metadata | `hazard_profile`, `perception_confidence` | qwen2.5:3b; `inspect_image_quality`, `lookup_similar_cases`, `recall_agent_notes` | severity/confidence בתוך השערים | הזיז את עץ הכללים ב-6/94; הכלים החזירו מידע אפס (ראה סעיף 5) |
| `rag_retriever` (Research) | שליפת תקנים + דירוג coverage | השליפה דטרמיניסטית: 4 שאילתות שונות בלבד ל-94 מקרים. ה-LLM שינה תוויות coverage/confidence ב-44 ריצות בלי שום השפעה על פסק דין | `hazard_profile`, road_type | `procedures`, `tool_result_ids`, `research_coverage(_detail)` | qwen2.5:3b; `query_repair_standards` | coverage בלבד | השליפה תורמת; ה-LLM לא |
| `decision` | פסק דין מנומק עם ציטוטים, ReAct | 58/94 הוכרעו דטרמיניסטית (catalog_match), 13 בעץ הכללים, 23 על ידי ה-LLM. ה-LLM לא קרא לאף כלי ב-30/30 ריצות (hops=0). 6/40 תשובות LLM לא נפרסו. 8/26 נימוקים נכשלו ב-fact-check | פרופיל, procedures, intake, dataset_match | action, priority, rationale, cited_sources, evidence_request | qwen2.5:3b; 5 סכמות כלים נשלחות בכל קריאה | בתוך ה-floors | ה-LLM שינה את התוצאה מול עץ הכללים ב-2/102 (llm_flip_rate 2.0%) |
| `approval_gate` | שורת אישור למפקח | 43 שורות pending, 0 הוכרעו, הוותיקה 29 שעות מול SLA של 24 | action, priority | `approvals` | אין | אין | עובד; אין צרכן |
| `await_evidence` | המשך אחרי ראיה | 6 ריצות batch (מענה סינתטי "אין תמונה אחרת" -> human_review, שימוש חוזר בפרופיל וב-RAG עבד 6/6); 2 בקשות אמיתיות ממתינות, 0 מולאו | update_state | `evidence_round` | אין | אין | עובד טכנית; זרימת המשתמש שבורה (ממצא F2) |

הפערים בין התכנון לביצוע: (1) המסמך מתאר את ה-LLM של Perception ו-Research כ-opt-in כבוי; בייצור הוא דולק. (2) המסמך מבטיח "Supervisor שמנתב"; בפועל אינו מנתב. (3) הטבלה מתארת 2 hops ל-Decision; הקוד 1 (סבב 25). (4) דף "How it works" הציבורי מבטיח "5 סוכנים, כל אחד על מודל קוד פתוח אחר"; בפועל qwen2.5:3b לארבעה ו-llama3.2:3b לאחד.

## 2. טבלת מדדים לכל סוכן (r23, 94 מקרים; בסוגריים 9 הדיווחים האמיתיים)

| סוכן | הפעלות | הצלחה במשימה | כשלים | תרומה לתוצאה | זמן ריצה (חציון / p95) | קריאות LLM | טוקנים (in / out) | בעיות מרכזיות |
|---|---|---|---|---|---|---|---|---|
| sanitize | 94 (9) | 100% | 0 | 33 known_image נחתמו, 5 scrubs | 0.1s | 0 | 0 | אין |
| signal_extractor | 94 מדולגות (9 עם LLM) | לא ניתן למדוד נכונות claims ללא תווית; תוצאה מכללים 9/9 | 14/28 קריאות כלי נדחו, 3 כשלי סכמה | 0 מעבר לכללים | (70s / 89s) | 0 (30) | 0 (49.8k / 3.0k) | לולאת "final answer in tool call" (F6); הטקסט לא נשמר ב-DB (F16) |
| supervisor | 94 (9) | הערה תקינה 53/94 (56%) | 41 הערות הוחלפו | 0 (לא מנתב, לא נקרא) | 5.3s / 6.0s | 94 (לא נספרות בעלות) | לא נרשם | ערך אפס במחיר ~8 דקות לסבב (F9) |
| profile_builder | 94 (9) | severity דטרמיניסטי תקין; הצעת ה-LLM נחסמה ב-56/89 | 4 כשלי כלי, 63/89 עם הפרה כלשהי | הזיז את עץ הכללים ב-6/94 (כולן standard->human_review דרך confidence=low; נכונותן לא מאומתת) | 50.7s / 74.3s (39s / 51s) | 174 (16) | 329.5k / 26.9k | כלים שמחזירים קבועים; 63% הצעות נחסמות (F7) |
| rag_retriever | 94 (9) | top-3 רלוונטיים 16/16 בגריד; sufficiency 0/94 | 18 "required_not_called" (fallback דטרמיניסטי) | שליפה: כן; LLM: 0 שינוי פסק דין ב-44 ריצות ששינה תוויות | 31.1s / 35.2s (28s / 60s) | 160 (15) | 248.5k / 15.1k | הקורפוס אינו עונה על שאלות ההחלטה (F4); LLM מיותר (F8) |
| decision | 100 (8) | LLM: 23 פסקי דין, 22 זהים ל-final; unparseable 3/36 (3/4 אמיתיים) | fact-check נכשל 8/26; floors 31 | שינה תוצאה מול העץ ב-2/102 | LLM: 69.3s / 124s; דטרמיניסטי: 0.1s | 52 (6) | 176.9k / 11.2k | ReAct ללא כלים, סכמות מיותרות, 15% unparseable (F10) |
| approval_gate | 42 (5) | 100% | 0 | תור שאין לו צרכן | 0 | 0 | 0 | 43 pending, כולן מעל SLA (F18) |
| await_evidence | 6 (0 מולאו) | שימוש חוזר 6/6 | 0 | מסלול batch תקין | 1.1s | 0 | 0 | למשתמש אמיתי המסלול נתקע (F1, F2) |

סך מקרה (batch): חציון 4 קריאות LLM, 96.5 שניות (p95 200), 6.9k טוקנים נכנסים. מקרה אמיתי עם טקסט: חציון 8 קריאות, 170 שניות (p95 260), 15.3k טוקנים. תקרה: 360 שניות. מתוך ~10,300 שניות LLM בסבב r23: profile 4,599, decision 2,816, rag 2,345, supervisor ~500 (לא נמדד בטרייס).

**מדדים שאינם ניתנים לחישוב היום:** דיוק/precision/recall של פסק דין מול אמת (יש 4 שורות `agent_feedback`, שתיים מהן מבדיקות עם מזהי hazard מזויפים); נכונות ה-severity שה-LLM הציע ונחסם; CPU seconds לסוכן (אין בטרייס); עקביות על אותו קוד (שלושת הסבבים רצו על builds שונים, סעיף 3.4). כולם ניתנים למדידה מהלוגים הקיימים ברגע שיהיו הצבעות מפקח; לא נדרש מנגנון חדש.

## 3. ממצאים לפי עדיפות

### P0

**F1. פסק דין needs_more_evidence ללא בקשת ראיה: המקרה נתקע לנצח והמשתמש מתבקש לשלוח תמונה שאין לאן לשלוח.**
- הבעיה: כש-RAG floor (סבב 25) הופך standard של ה-LLM ל-needs_more_evidence, הוא לא יוצר שורת `evidence_requests`. הסטטוס נשאר `open`, אין approval, אין TTL.
- ראיה: `hz_c9113fc7b54e`, `hz_34b5a89f1264`, `hz_eca49f5ec87d`, `hz_351f0aa8f1dd` (r23, web_17/22/26/39): `agent_decision=needs_more_evidence`, `status=open`, `evidence_requests` ריק, `missing_evidence=[]`. קוד: `roadguard_agent.py` decision_node, בלוק "Round 25 P1 - RAG enforcement" (~11500-11529) בונה decision בלי `evidence_request`; `_route_after_decision` (11765) שולח ל-await_evidence; `roadguard_api.py` 6286: `waiting = _evidence_request_out(decision, hazard_id) is not None` -> False -> `status="open"`.
- השפעה: באתר האזרח מקבל "one more photo would help ... Please send one more photo" (`CITIZEN_COPY.needs_more_evidence`), אבל `setPendingEvidence` לא נקרא (הסטטוס אינו waiting_for_evidence), ולכן ההעלאה הבאה פותחת מקרה חדש. המקרה המקורי לא מגיע לאף תור.
- גורם משוער: ה-floor נכתב כשינוי action בלבד; מסלול בקשת הראיה (`_phrase_evidence_request` + כלי `request_additional_evidence`) שייך ל-LLM.
- פתרון מוצע: אחת משתיים, בלי פיצ'ר חדש: (א) כשה-floor בוחר needs_more_evidence, לעבור באותו מסלול שה-LLM עובר (ניסוח בקשה + שורת evidence_requests), או (ב) לבחור human_review במקום. בנוסף, sweep חד-פעמי ל-4 המקרים הקיימים.
- סיכון: טסטי סבב 25 (`rag_insufficient_coverage`), טסטי evidence cap.
- אימות: טסט שכל final עם action=needs_more_evidence מחזיק request_id או שסטטוס ה-DB אינו open; הרצה חוזרת של 4 המקרים.

**F2. זרימת הראיה למשתמש אמיתי שבורה: הדפדפן שוכח את הבקשה אחרי 10 דקות, אותה תמונה נדחית, אין דרך לסיים בלי תמונה, והתוצאה היא מקרים כפולים.**
- הבעיה: (1) `app.js` 862 `PENDING_EVIDENCE_TTL_MS = 10 דקות` מול `policy.yaml` `evidence_ttl_hours: 72` בשרת; אחרי 10 דקות העלאה "ממשיכה" פותחת מקרה חדש. (2) `app.js` 1129: אותה תמונה נחסמת ("already in flight") בלי להסביר שצריך קובץ אחר. (3) אין כפתור "אין לי תמונה נוספת" אף שהשרת תומך במסלול "אותה תמונה + הערה -> evidence_declined -> human_review" (`batch_runner.py` 249-256 משתמש בו). (4) טבלת "My reports" מציגה waiting_for_evidence בלי דרך לחזור לבקשה.
- ראיה: `hz_ec81ccd10484` (12:26, Dereh Hficus Ashdod, D40 0.48) ממתין עם `ev_28f9e09a2a`; המשתמש ניסה שוב ב-12:51 ונוצר `hz_1180331078d9` זהה (אותו phash neighbour pic_id=7624, אותו נימוק) עם בקשה שנייה `ev_af40ad8a37`. ב-`/digest/latest` שניהם מופיעים כשתי שורות needs_more_evidence 45. מתוך 2 בקשות ראיה אמיתיות, 0 מולאו.
- השפעה: המשתמש נתקע (בדיוק מה שדיווחת), תור המפקח מקבל כפילויות, סבב הראיות (הפיצ'ר שסעיף 2 ב-roadmap מתגאה בו) לא נסגר אף פעם במציאות.
- גורם משוער: TTL לקוח שנבחר כדי להימנע מ-404 על בקשה שפגה, בלי להתאים לשרת; הודעת הכפילות נכתבה למקרה של לחיצה כפולה.
- פתרון מוצע: ליישר את ה-TTL בלקוח ל-72 שעות או לשחזר את הבקשה הפתוחה מ-`/report/{id}` (השדה `evidence_request` כבר קיים) כשהמשתמש נכנס; לחשוף את המסלול הקיים "אין לי תמונה נוספת" ככפתור בכרטיס הבקשה (שולח את אותו הקובץ + ההערה הקבועה, כמו ה-batch); לשנות את הודעת הכפילות ל"זו אותה תמונה; בחר קובץ אחר או לחץ אין לי תמונה נוספת".
- סיכון: טסטי auth/evidence ב-webapp אינם אוטומטיים; שינוי הודעות בלבד.
- אימות: שחזור ידני: בקשה, סגירת החלון, חזרה אחרי 15 דקות, שליחת תמונה שנייה -> אותו hazard_id, סטטוס עוזב waiting_for_evidence.

### P1

**F3. ריצה שנקטעה מוצגת לאזרח כהצלחה רגילה.**
- ראיה: `hz_388c10cf3f84` (6.11 12:22, menahem Begin 8 Ashdod, D40 0.69). ההפעלה מחדש של השירות ב-12:25 (פריסת סבב 27 שלי) הרגה את המשימה; `recover_interrupted_runs` (`roadguard_api.py` 2050) ניתב ל-human_review עם הנימוק "service restarted while the run was in progress". האזרח ראה "Thank you ... A municipal supervisor will take a look at this one personally"; ב-digest השורה נראית כמו כל human_review (priority 50, source ריק, `agent_value_json` ריק).
- השפעה: מידע לא נכון על מה שקרה; המפקח לא יודע שאין כאן ניתוח; המדדים (justified, cost) מדלגים על המקרה בשקט.
- פתרון מוצע: לסמן את המקרה (למשל `fallback_reason=graph_interrupted` כבר קיים ב-`_route_to_human`) ולהציג אותו בשלושת המקומות: הודעת אזרח נפרדת ("נבדוק את הדיווח שוב"), תג "interrupted" ב-digest, ודילוג על הפעלה מחדש של השירות כשיש ריצה פעילה (`app.state.async_slots` תפוס) בסקריפטי הפריסה. סיכון: נמוך. אימות: הפעלה מחדש מכוונת בזמן ריצת בדיקה בסביבה מבודדת.

**F4. כלל ה-sufficiency של Decision אינו בר-השגה עם הקורפוס הקיים, ולכן ה-RAG floor (סבב 25) הוא קבוע ולא רשת ביטחון.**
- ראיה: `_decision_coverage` (`roadguard_agent.py` 9128) דורש חלון תיקון (regex "within N days" וכד') וגם דחיפות או סיכון בטיחותי בתוך הפסקאות שנשלפו. התוצאה: `insufficient_coverage` ב-94/94 מקרי r23 וב-9/9 אמיתיים. סריקה של כל 4,663 ה-chunks בקורפוס v2: רק 112 מזכירים חלון זמן, ורק 9 chunks בכל הקורפוס היו מספקים את הכלל לבדם ומזכירים pothole/crack/pavement; אף אחד מהם אינו ב-top-3 של ארבע שאילתות המחלקות (גריד הרלוונטיות: כל 48 הפסקאות הן מ-IRC:82-2015, 13092, Distress Manual, 7941).
- השפעה: כל standard של ה-LLM על תמונה חדשה ללא שכן בקטלוג הופך ל-needs_more_evidence (4/4 ב-r23) ונופל ל-F1. ה-20 ה-standard שה-LLM הציע ונשארו, נשארו רק כי היה להם phash neighbour (התנאי `not dataset_match`), כלומר הכלל מתנהג הפוך מהכוונה: ככל שהתמונה חדשה יותר, הסיכוי ל-standard קטן יותר.
- גורם משוער: הכלל נכתב כשהקורפוס היה 15 פסקאות seed; קורפוס v1/v2 של בני לא כולל SLA עירוני לתיקון.
- פתרון מוצע: להחזיר את ה-sufficiency למעמד של מידע ל-Decision ולנימוק (כפי שהיה עד סבב 25) ולא floor, עד שהקורפוס יכיל מסמך SLA/חלונות תיקון; במקביל לבקש מבני את המסמך (הצ'רטר העירוני/אמנת השירות שכבר נמצאים חלקית: `2023-Shilton-303-Drachim.pdf` מזכיר "standard response times"). חלופה קטנה: להקל את הכלל ל"דחיפות או סיכון" בלבד.
- סיכון: הטסטים של סבב 25; אם מסירים את ה-floor, פסקי standard של ה-LLM על תמונות חדשות יעברו (כמו עד 4.11).
- אימות: `scripts/rag_relevance_eval.py` + ספירת sufficiency על 94 המקרים אחרי השינוי; טסט שמקרה עם חלון תיקון בפסקאות הופך sufficient.

**F5. ה-floor של phash_class_mismatch הופך דמיון חלש לראיה מכרעת: 24/103 מקרים נשלחו לבדיקה אנושית על סמך שכן במרחק 12-16 ביטים.**
- ראיה: מרחקי ה-phash במקרים שה-floor פעל: 9, 12, 12, 12, 13×5, 14×2, 15×10, 16×10 (מתוך 64 ביטים). סף known_image הוא 10 (`find_dataset_image(max_distance=10)`), אבל ה-floor ב-`_decide_rule_based` (9508-9513) בודק רק `match == "phash"` בלי סף מרחק, על שכן שמגיע מחיפוש ברדיוס 24. דוגמה אמיתית: `hz_e81a4bdaa26f` (6.11, D40 בביטחון 0.85 על כביש משני) הופנה למפקח כי תמונה דומה במרחק 14 בקטלוג מתויגת D00/D10/D20. הנימוק שנכתב, "approximate similarity cannot validate the class", סותר את עצמו: אותו דמיון מקורב הוא שמכריע.
- השפעה: 22/94 ב-batch ו-2/9 אמיתיים לתור המפקח ללא ראיה; ב-23 מהם profile+RAG רצו עם LLM (~90 שניות, 4 קריאות) לשווא, כי ה-floor ידוע כבר אחרי sanitize.
- פתרון מוצע: להחיל את ה-floor רק על מרחק קטן או שווה ל-10 (אותו סף של "תמונה דומה") או לדרוש שגם החיזוי השמור של השכן יסכים עם תוויותיו; לדלג על ה-LLM של העובדים כשה-floor ייפול (ראה F12).
- סיכון: טסטי סבב 21 (`phash_class_mismatch`), שני המקרים מ-r13 (`hz_770ac7c30461`, `hz_995e3102af69`) חייבים להישאר human_review אם מרחקם קטן מהסף.
- אימות: ספירת floors על 94 המקרים לפני ואחרי; בדיקה ידנית של 5 המקרים בגבול.

**F6. ה-LLM של Intake מבזבז 70 שניות למקרה ותוצאתו נזרקת: לולאת "final answer in a tool call".**
- ראיה: 9 הדיווחים האמיתיים: `source=rules` ב-9/9; 14 מ-28 סבבי הכלי נדחו עם "arguments look like another tool's final answer" (המודל שלח את התשובה הסופית כקריאה ל-`extract_measurements`, חזר על זה 2-3 פעמים עד תקרת 90 השניות); `extract_measurements` החזיר null בכל השדות 8/8; `lookup_prior_reports` נכשל 3 פעמים על סכמה (`hazard_class` null). קוד: ב-`_react_worker_rounds` (3840-3861) הבדיקה של סבב 18 `_worker_tool_call_is_other_tools_reply` רצה לפני בדיקת סבב 11 `_worker_final_in_tool_call`, ושתיהן אותו תנאי; מאחר ש-`extract_` ברשימת `_WORKER_LOOKUP_PREFIXES`, תשובה סופית שמגיעה דרך extract_measurements תמיד נדחית ולעולם אינה מתקבלת כתשובה. ב-Decision, סבב 10b דווקא מקבל תשובה כזו (`_looks_like_final_answer`).
- השפעה: 40% מזמן התגובה למשתמש אמיתי (170 שניות חציון מול 96 ב-batch); 30 קריאות llama ללא תועלת; זיהוי הסתירה "המדווח כתב pothole אבל המודל מצא D00" הגיע ממילות המפתח.
- פתרון מוצע: ב-`_react_worker_rounds` לבדוק קודם אם הארגומנטים הם התשובה הסופית של הסוכן עצמו (אותו `_worker_final_in_tool_call`) ולקבלם כתשובה, ורק אחרת לדחות ככלי זר; לא להציע `extract_measurements` כשה-regex לא מצא מספרים בטקסט.
- סיכון: טסט סבב 18 (דחיית final דרך `lookup_similar_cases` ב-r11) חייב להישאר; הפתרון מבחין לפי מפתחות הסוכן ולא לפי שם הכלי.
- אימות: replay של 9 הטרייסים עם התשובות המוקלטות; זמן Intake יורד מ-70 ל-~20 שניות.

**F7. ה-LLM של Perception: 63% מהצעותיו נחסמות, הכלים שלו מחזירים קבועים, והמדד "justified" סופר שימוש בכלים כערך.**
- ראיה: 89 ריצות LLM: שינוי severity נחסם ב-56 (`severity_blocked_downgrade` 49, `_upgrade` 7), טענת rationale סותרת ב-17 (`lane_blockage_claim_but_form_says_none`), הפרה כלשהי ב-63/89. ה-LLM השפיע בפועל: confidence medium->low ב-9, severity moderate->minor ב-1; עץ הכללים זז ב-6/94. כלים: `inspect_image_quality` החזיר "high" 93/93 כי `batch_runner.py` שולח `blur_score=150` קבוע; `lookup_similar_cases` החזיר 0 ב-91/91 (המיקום הוא שם קובץ); `recall_agent_notes` החזיר 0 ב-48/48 (טבלת `agent_notes` ריקה). למרות זאת `per_agent_justified` סימן YES ב-85/94 עם הסיבה `agent_reasoned_with_tools`.
- השפעה: הסוכן היקר ביותר (4,599 שניות LLM בסבב, 45% מזמן ה-LLM) עם תרומה מאומתת של 6 מקרים שכיוונם לא נבדק מול מפקח.
- פתרון מוצע: להשאיר את הפרופיל הדטרמיניסטי כברירת מחדל ולהפעיל את ה-LLM רק כשיש אות שהכללים לא קוראים (טקסט מדווח, ביטחון 0.4-0.6, יותר ממחלקה אחת בתמונה) ולעולם לא במקרה חתום/floor; להוציא את `recall_agent_notes` מההיצע עד שהטבלה מתמלאת; ב-`_compute_per_agent_justification` לספור כלי כ"ערך" רק כשהתוצאה שלו אינה ריקה/קבועה.
- סיכון: טסטי `test_worker_autonomy.py`, סבבים 18 ו-21 (השערים נשארים).
- אימות: replay של r23 עם ה-LLM מדולג: `rule_tree_action` ו-`final_action` זהים ב-94/94, זמן הסבב יורד ב-~40%.

**F8. ה-LLM של Research לא משנה פסק דין אף פעם, והשליפה עצמה חוזרת על עצמה.**
- ראיה: 94 ריצות, 4 מחרוזות שאילתה שונות ו-4 קבוצות id שונות (השאילתה נגזרת מ-class ו-road_type). ה-LLM שינה coverage direct->partial ב-25 וconfidence ב-44, ובאף אחד מהם פסק הדין לא השתנה (`research_llm_changed=44`, שינויי verdict 0). ב-18 ריצות הסוכן לא קרא לכלי החובה וה-fallback הדטרמיניסטי שלף במקומו. "findings" לדוגמה: "The articles have a relevance score of 0.3416566252708435, indicating".
- השפעה: 2,345 שניות ו-160 קריאות בסבב ללא תועלת; תווית coverage שה-LLM משנה מופיעה למפעיל כאילו משהו השתנה.
- פתרון מוצע: להפעיל את ה-LLM של Research רק כשהרלוונטיות הדטרמיניסטית אינה direct (25 מקרים במקום 94), או לכבות אותו (`AGENT_RAG_RETRIEVER_LLM=0`, כפי שהמסמך מגדיר כברירת מחדל); לשמור תוצאות שליפה לפי (class, road_type) בתוך סבב (`decision_cache` קיים ו-ריק).
- סיכון: טסטי RAG (`test_rag_truth.py`, סבב 16); חוזה ה-handoff לא משתנה.
- אימות: replay: אותם `tool_result_ids` ואותם פסקי דין ב-94/94.

**F9. ה-LLM של Supervisor הוא קישוט: לא מנתב, לא נקרא, ושגוי ב-44%.**
- ראיה: `_maybe_run_supervisor_agent` (7515): "Routing is NEVER changed"; `AGENT_SUPERVISOR_ROUTING` אינו מוגדר ב-env (ברירת מחדל 0); `force_action` ריק ב-111/111; `pipeline_concern`/`expected_difficulty` נצרכים רק בפורמט הטרייס. ההערה הוחלפה ב-41/94 (`replaced:detector_confidence_band`). בדיקת המילים `_CONF_WORD_RE` (10606) מזהה רק "high detector confidence"/"detector confidence is high", ולכן "detected with high confidence" על ביטחון 0.00 עבר כ-ok (`hz_18a448b99627`). העלות (~5.3 שניות, קריאה אחת) אינה נספרת ב-`_compute_run_cost` כי לרשומת ה-supervisor אין בלוק `llm`.
- פתרון מוצע: או לחבר את `pipeline_concern` למשהו (payload של Decision), או לכבות את קריאת ה-LLM ולהשאיר את צומת הניתוב הדטרמיניסטי; להרחיב את ה-regex ל"with (very )?(low|high) confidence"; להוסיף בלוק `llm` לרשומת הטרייס כדי שהעלות תיספר.
- סיכון: טסטי supervisor (`test_round20_integrity.py` B). אימות: replay ללא שינוי פסקי דין; ~8 דקות פחות לסבב.

**F10. "ReAct" ב-Decision אינו ReAct, ו-15% מהתשובות לא נפרסות.**
- ראיה: `DECISION_REACT_MAX_HOPS=1` (סבב 25); hops=0 ב-30/30 ריצות; חמש סכמות כלים (3,448 תווים, ~860 טוקנים) נשלחות בכל קריאה על פרומפט של 8,937 תווים; חציון 69 שניות לתשובת JSON אחת; `llm_reply_unparseable` ב-6/40 (3/36 batch, 3/4 אמיתיים: `hz_306f7517191d`, `hz_8863cf4ca9e7`, `hz_a77c1cfc4e7d`) -> עץ הכללים החליט; fact-check נכשל ב-8/26 (`urgency_not_in_sources` 7) ונשמר כ-"kept" עם נימוק דטרמיניסטי; תמיכת הציטוטים היא `class_keyword` (26) או `catalog` (79) בלבד.
- פתרון מוצע: כל עוד התקרה 1 והכלים לא נקראים, לא לשלוח את הסכמות (חיסכון ~20% מהפרומפט, ~10-15 שניות לקריאה); תיקון JSON דטרמיניסטי (חילוץ האובייקט הראשון, השלמת סוגריים) לפני הכרזת unparseable; או להחזיר 2 hops אם רוצים כלים באמת.
- סיכון: טסטי סבב 10 (מקובעים עכשיו ל-2 hops), סבב 19/26. אימות: שיעור פרסינג על 40 התשובות המוקלטות.

**F11. טקסט המדווח אינו נשמר ב-DB במסלול ה-async, ולכן המפקח לא רואה מה האזרח כתב.**
- ראיה: 9/9 דיווחים אמיתיים: `hazards.reporter_text=''` בעוד הטרייס מראה `has_text=True` ואת הטקסט ("A side road with small potholes and damaged sections..."). `roadguard_api.py` 6379-6395: `hazards_submit_async` יוצר את השורה בלי `reporter_text`; `update_hazard_verdict` אינו כותב אותו; המסלול הסינכרוני (6445) כן כותב. ב-digest ובדף ה-report אין שדה לטקסט.
- השפעה: המפקח מכריע בלי המילים של המדווח; המדדים על "reporter claims" לא ניתנים לבדיקה ידנית.
- פתרון מוצע: להעביר `reporter_text=sanitize_reporter_text(...)["text"]` גם ביצירת השורה ב-async (שורה אחת), ולהציג בכרטיס המקרה ב-digest. סיכון: אין. אימות: טסט על submit-async.

**F12. מקרים חתומים ו-floored עדיין משלמים על LLM של העובדים: 45% מזמן הסבב.**
- ראיה: 33 מקרים נחתמו known_image ב-sanitize; ב-30 מהם profile ו-RAG רצו עם LLM (2-4 קריאות, 47-82 שניות). 58 מקרי catalog_match צרכו 206 קריאות LLM ו-4,682 שניות מתוך ~10,300. ב-23 מקרי phash floor אותו דבר.
- פתרון מוצע: כשה-`known_image_shortcut` דולק או כשה-floor של phash ייפול (שניהם ידועים אחרי sanitize), להריץ את העובדים במצב דטרמיניסטי (אותו קוד, בלי קריאת LLM). פסקי הדין לא משתנים (הם דטרמיניסטיים ממילא). סיכון: שדות הפרופיל ל-digest ממשיכים להתמלא מהמסלול הדטרמיניסטי. אימות: replay: אותם 58 פסקי דין, סבב קצר ב-~45%.

**F13. דף "How it works" הציבורי מבטיח מה שהקוד לא עושה.**
- ראיה (Chrome, `orarr2.github.io/RoadGuard/#how`): "5 AI agents ... each running a different open-source language model" (בפועל qwen2.5:3b לארבעה, llama3.2:3b ל-Intake); "a supervisor agent short-circuits obviously-hard cases straight to a human" (F9: לא מנתב); "a research agent looks up the matching municipal repair standard" (בפועל מדריכי IRC/FHWA, ללא תקן עירוני, sufficiency 0/103).
- פתרון מוצע: תיקון ניסוח בלבד ב-`webapp/index.html`. סיכון: אין.

**F14. מה שהאזרח מקבל: אישור קבלה גנרי בלי מה זוהה, בלי רמת ודאות, בלי מה הלאה; ובטבלה שלו slugs פנימיים.**
- ראיה: `CITIZEN_COPY` (`app.js` 1351-1372) מציג לכל פסק דין "Thank you - your report has been received" + משפט אחד; בכוונה בלי class, confidence, severity, priority או נימוק. טבלת "My reports" מציגה `human_review`, `needs_more_evidence`, `known_image` ו-priority מספרי (0 ל-known_image). שני דיווחים אמיתיים (`hz_306f7517191d`, `hz_18a448b99627`) נשלחו עם D00 בביטחון 0.00, כלומר הגלאי לא מצא כלום, ועברו את כל הצינור (229 ו-161 שניות) עד human_review; האזרח לא נאמר לו שלא זוהה נזק.
- פתרון מוצע (בקשתך מס' 2): פירוט קצר ודטרמיניסטי מהשדות הקיימים, לא טקסט חופשי של ה-LLM: מה זוהה (שם המחלקה בשפה פשוטה), רמת ביטחון במילים (low/moderate/high מה-policy), חומרה, מה הצעד הבא ומי עושה אותו, ומה חסר (למשל "לא זוהה נזק בתמונה; מפקח יבדוק"). כל השדות כבר ב-`/report/{id}`. להחליף את ה-slugs בטבלה בתוויות (יש `HAZARD_CLASS_LABELS` לדוגמה).
- סיכון: אין בשרת; ב-webapp שינוי תצוגה. אימות: Chrome על 9 המקרים.

### P2

**F15. דפי המפעיל: ה-trace מציג את הישן לפני החדש, ה-inbox סותר את עצמו.**
- `/roadguard/trace` (`roadguard_api.py` 228-273): replay של 200 המקרים האחרונים מהישן לחדש, קריאת `/report` נפרדת לכל אחד; אחרי 3 שניות הדף הראה 17 מקרי r23 מה-5.11 ואף לא אחד מהיום (177k תווים). `/roadguard/supervisor`: "Overview Total: 0 ... Human review: 0" ליד רשימת 43 אישורים ממתינים, 42 מהם "overdue 26-29h" מול SLA 24 שעות. `/digest/latest` מציג את הכפילות של F2 כשתי שורות. `/roadguard/flow`: "Agreement with rule tree 100% / target 85%" מוצג כ-KPI "correct_action" אף שאין אמת.
- פתרון: replay מהחדש לישן ומוגבל (24 שעות או 20 מקרים); מונה ה-Overview לפי אותו חלון של הרשימה; תווית ה-KPI "agreement" ולא "correct".

**F16. priority 0 נרשם על needs_more_evidence למרות דגל ההחלפה.**
- ראיה: `hz_a7a8f0c54b02` (r23): ה-final הראשון `priority=0`, `priority_from_severity_check=True`, בעוד `check_hazard_severity("D00", 0.45, "main", "none", 0)` מחזיר 32 (אומת מקומית); הריצה השנייה (evidence_declined) כבר 32. קוד: decision_node 11344-11354 מחליף את `decision["priority"]`; יש לבדוק מי בונה את תוכן הטרייס/בקשת הראיה מעותק קודם (`_phrase_evidence_request`, `_apply_evidence_cap`). לא אומת סופית בקוד; מוצע replay של המקרה.

**F17. ממצאי בטיחות שנבדקו ונמצאו תקינים (לידיעה):** הפרדה בין sha256 (זהות) ל-phash (דמיון) נשמרת: known_image נחתם רק ב-sha256 (28/28 agrees), ו-`phash_neighbor_cited_as_evidence` לא נפל ב-0/26 נימוקי LLM; שימוש חוזר בסבב ראיה עבד 6/6; `_scrub_state_on_fact_check_failure` נדרש פעם אחת; עץ הכללים דטרמיניסטי לחלוטין (priority זהה ב-71/71 מקרים שחזרו על אותו action בשלושה סבבים).

### P3

**F18. חשבונאות ומדדים:** `run cost` לא סופר את קריאת ה-supervisor; "agents justified" סופר כלים ריקים כערך (profile 85/94 YES); KPI `correct_action` הוא הסכמה עם עץ הכללים (100%); `agent_feedback` מכיל 2 שורות מבדיקות עם מזהים מזויפים (`hz_test_r23...`, verdict ריק באחת); `/stats/agents` ו-`/flow` מראים "signal_extractor justified 7/8" בעוד תוצאתו נזרקה ב-8/8 (F6).

**F19. עקביות בין ריצות:** 80 תמונות רצו בשלושת הסבבים r21-r23 (builds שונים): אותו action ב-71/80 (89%); 9 ההיפוכים כולם בין standard ל-needs_more_evidence/human_review על תמונות web בביטחון נמוך, כלומר האי-דטרמיניזם מרוכז במסלול ה-LLM. על אותו build לא נמדד.

**F20. התור האנושי ללא צרכן:** 43 approvals pending (0 הוכרעו), 29 work orders במצב draft, 0 הצבעות מפקח על 102 מקרים; מדדי "advantage" ו-precision לא יכולים להתמלא עד שמישהו מכריע. זה ממצא תפעולי, לא קוד.

## 4. תקשורת ושיתוף פעולה בין הסוכנים

| בדיקה | ממצא | ראיה |
|---|---|---|
| כל סוכן מקבל את המידע הנכון מהקודם | כן, דרך state מפורש (`hazard_profile`, `procedures`, `reporter_intake`); ה-handoff_out מתועד | 102 טרייסים, 0 handoff חסר |
| מידע שאובד במעבר | טקסט המדווח אובד בין ה-API ל-DB (F11); `supervisor_plan` מיוצר ולא נצרך (F9) | 6379-6395; 11539 |
| נתונים לא מאומתים שעוברים הלאה | הצעות ה-LLM של Perception נחסמות בשערים לפני המעבר (56/89); טענות Intake שלא נמצאו בטקסט נזרקות (`dropped` ב-2/9) | אותם טרייסים |
| סתירה בין סוכנים | Decision מצטט "catalog:pic_id=N" כמקור גם כשהמקרה אינו known_image (image_match_cite) - מקור "תמיכה" שאינו תקן | 79 ציטוטי catalog |
| עבודה כפולה | signal regex + LLM (F6); RAG LLM + relevance דטרמיניסטי (F8); profile LLM + שערים (F7); 4 שאילתות RAG זהות ל-94 מקרים (F8) | |
| הפעלות מיותרות | עובדים רצים על מקרים שכבר הוכרעו (F12); supervisor LLM (F9) | |
| כשלי state/handoff | אחד: F16 (priority 0) | |
| החלטה על פלט סוכן בלי לבדוק אותו | לא נמצא: ה-floors והשערים הם הבדיקה; עץ הכללים לא מאמין ל-LLM | |
| הפרדה נצפה/הוסק/אומת | בטרייס כן (`observed` vs `llm_delta` vs `rationale_check`); למשתמש ולמפקח לא (F14) | |
| אכיפה עקבית של ה-Supervisor | הניתוב הדטרמיניסטי עקבי; האכיפה היא ב-Decision (floors), לא ב-Supervisor | |

## 5. ReAct ואוטונומיה

- בחירת כלים לפי צורך: Perception קורא לשלושת הכלים כמעט תמיד (85/83/46 מ-94) ללא קשר למקרה; Research קורא לכלי החובה ב-71/94 ומדלג ב-18 (ה-fallback משלים); Decision אינו קורא לכלים כלל (0/30). Intake קורא ל-`extract_measurements` גם כשאין מספרים (8/8 null).
- הבחנה בין כלי שהסוכן בחר לכלי fallback: קיימת וטובה (`source=agent_choice | deterministic_fallback | agent_retrieval`).
- פירוש תוצאות כלי: כשהתוצאה ריקה הסוכן ממשיך בלי לציין זאת; "agent_reasoned_with_tools" נרשם כהצדקה.
- קריאות כושלות שהריצה ממשיכה כאילו הצליחו: 4 כשלי `lookup_similar_cases` (misroute) ב-profile ו-17 ב-Intake; הריצה ממשיכה (נכון), אבל ההצדקה והמדדים לא משקפים זאת.
- קריאות כפולות / פרמטרים חסרים: 4 כפילויות באותו מקרה (profile), 3 קריאות עם `hazard_class=null` (Intake).
- עצירה כשיש מספיק מידע: אין: מקרה חתום ב-sanitize עדיין עובר 2 עובדים עם LLM (F12).
- העברה לבירור במקום מסקנה לא מבוססת: קיימת (floors, evidence), אבל ב-needs_more_evidence של ה-floor המסלול שבור (F1).

הערך המעשי של הפעלות הכלים ב-r23: 0 תוצאות שהשפיעו על פסק דין. הסיבה המבנית: ה-batch שולח `blur_score=150`, `is_road_scene=true`, מיקום = שם קובץ, וטבלת ההערות ריקה. ב-9 המקרים האמיתיים: `lookup_prior_reports` 0 (כתובות חדשות), `extract_measurements` null (אין מידות בטקסט). המסקנה: הכלים הם נכונים לעולם אמיתי עם היסטוריה, אבל היום הם לא מודדים כלום, ולכן "ReAct" היום הוא עלות בלבד.

## 6. RAG ומקורות

- רלוונטיות השליפה: 16/16 זוגות class x road עם 3/3 פגיעות מחלקה (ריצה חוזרת 6.11, קורפוס v2 חי); מרחקים D20 0.21, D40 0.22, D10 0.30-0.36, D00 0.34-0.39.
- האם התוכן תומך במסקנה: הפסקאות מתארות שיטות תיקון (IRC:82-2015, FHWA 7941), לא מדיניות עירונית; לכן `sufficiency` 0/103 (F4). התוויות `support=direct` ב-266/282 פסקאות מודדות מילת מפתח של המחלקה, לא תמיכה בהחלטה.
- התאמה בין סוג המפגע, הסיווג והמידע: כן (class_hit 100%).
- מידע חלקי שמוצג כהוכחה: הציטוט "catalog:pic_id=N" מופיע ב-`cited_sources` של 79 פסקי דין כמקור, גם כשההתאמה היא phash (דמיון) ולא זהות. למפקח זה נראה כמו מקור.
- Exact vs phash: מופרדים נכון (`image_match_status.match`), known_image רק ב-sha256; אך ה-floor של phash (F5) וציטוט ה-catalog מטשטשים את ההפרדה בתוצאה.
- מה לא נמצא: ה-RAG מציין `missing=[repair_window, traffic_control, safety_risk]` בכל מקרה, אבל הטקסט למשתמש ולמפקח לא אומר זאת במילים פשוטות.
- קורפוס v2 (3.11): 1,254 chunks חדשים שאינם מגיעים ל-top-3 באף שאילתת מחלקה; השפעה על הסוכנים: אפס.

## 7. הפלט הסופי למשתמש (Chrome)

| שאלה | מצב בפועל | ראיה |
|---|---|---|
| מה זוהה, סוג המפגע, רמת הביטחון | לא מוצג לאזרח (בכוונה, `CITIZEN_COPY`); בטבלת My reports רק slug + מספר | app.js 1351-1404, 110-123 |
| ודאי / לא ודאי | לא מוצג | |
| מיקום | מוצג בטבלה | |
| חומרה ומשמעותה | לא | |
| טיפול / בדיקה נוספת / אישור אנושי | משפט גנרי אחד לכל action; needs_more_evidence מציג את הבקשה (כשקיימת) | |
| סטטוס טיפול | בטבלה `open`/`waiting_for_evidence`; 101 מ-102 מקרים הם `open` לנצח (אין הכרעות) | /stats/public: open_now 101 |
| פעולה נדרשת מהמשתמש | רק בבקשת ראיה, ורק 10 דקות (F2) | |
| פרטים חסרים שמשפיעים על אמינות | לא נאמר ("לא זוהה נזק", "הקורפוס לא ענה") | F14 |
| מסקנה שסותרת את הנתונים | `hz_e81a4bdaa26f`: D40 בביטחון 0.85 -> "supervisor will look personally" בגלל דמיון במרחק 14 (F5) | |
| טענות שה-fact-checker פסל | לא מוצגות לאזרח; למפקח מוצג הנימוק הדטרמיניסטי + "(note: the fact-check removed ...)" (תקין) | `hz_ec81ccd10484` rationale |
| חשיפת פנימיות | לאזרח: slugs ומספרי priority בטבלה; מזהה מקרה; לא tool calls. לדפי המפעיל: הכל (מכוון) | |
| הבחנה זיהוי/הערכה/אימות/החלטה | לא לאזרח | |
| תהליך נכשל אך מוצג כהצלחה | F3 (`hz_388c10cf3f84`) ו-F1 (4 מקרים) | |
| סטטוס באתר מול DB | תואם (open/waiting_for_evidence); ה-Overview ב-/supervisor לא תואם (F15) | |

דוגמה מלאה ממקרה אמיתי: `hz_ec81ccd10484`. האזרח ראה: "The agent asks for one more photo ... Please photograph the damage from the other side so the lane line and its edge are visible." (ההודעה נוסחה על ידי ה-LLM; הנימוק הפנימי מציין lane none וסתירת lane blockage שהוסרה). מה שהיה צריך לראות: "זוהה בור (D40) ברמת ביטחון בינונית (0.48) בכביש משני. כדי לקבוע אם זה טיפול שגרתי צריך תמונה מזווית אחרת; אם אין לך, לחץ 'אין לי תמונה נוספת' והמקרה יעבור למפקח. מספר דיווח: ...".

## 8. 12 התרחישים על מקרים אמיתיים

| # | תרחיש | מקרה | מה קרה בשרשרת | פלט למשתמש | תקין? |
|---|---|---|---|---|---|
| 1 | תמונה חדשה, מפגע ברור | `hz_82a897abc201` (D20 0.82), `hz_9bc5b16e0a09` (D20 0.75) | profile/rag/decision LLM, standard, fact-check ok, 220 שניות | "received, regular maintenance" | כן, אבל 5 קריאות LLM ו-220 שניות ל-standard שעץ הכללים נתן ב-0.1 שניות |
| 2 | סיווג לא ודאי | `hz_ec81ccd10484` (D40 0.48, אמיתי) | needs_more_evidence מה-LLM, בקשה נוסחה | בקשת תמונה; נתקע (F2) | חלקית |
| 3 | סתירה מודל מול מאגר | `hz_61a9021dadd6` (sha match, labels D00/D10 vs live D20), `hz_c31d543db98c` | floor known_image_catalog_* -> human_review, בלי LLM של Decision | "supervisor will look" | כן |
| 4 | Exact SHA match + נתונים תואמים | `hz_76895d7dd4f9`, `hz_6cb7f223d635` ועוד 26 | sanitize חתם known_image; profile+rag עדיין רצו עם LLM (F12) | "we already know about this one" | פסק דין נכון, עלות מיותרת |
| 5 | phash match בסיווג שונה | `hz_e81a4bdaa26f` (אמיתי, 0.85), `hz_aa254a0f1dd8` + 22 | floor -> human_review במרחק 12-16 (F5) | "supervisor will look" | לא: ראיה חלשה מכריעה |
| 6 | ללא טקסט / מידע משלים | כל 94 מקרי r23 | Intake מדולג (0 שניות), הצינור ממשיך | | כן |
| 7 | מידע חסר/סותר בין סוכנים | `hz_306f7517191d` (D00 0.00 מול "pothole" בטקסט) | הסתירה זוהתה בכללים -> human_review; supervisor כתב "very low confidence... no action needed" | "supervisor will look" | הניתוב נכון; הסוכנים לא הסבירו שלא זוהה נזק |
| 8 | כלי נכשל / מחזיר ריק | `hz_9a544d4b8db4`, `hz_cd6da379d3a5` (lookup_similar_cases נדחה), Intake ב-7/9 אמיתיים | הריצה ממשיכה; justified=YES בכל זאת | | ממשיך נכון, מדווח לא נכון |
| 9 | סוכן מחזיר טענה לא נתמכת | profile: 17 מקרים (`rationale_claim:lane_blockage...`), decision: 8 (`urgency_not_in_sources`), למשל `hz_f8b1ef3efdac` | השערים חוסמים/מחליפים; הטרייס מתעד | לא מגיע למשתמש | כן |
| 10 | נדרשת בדיקה אנושית | 43 מקרים ב-approvals | שורת approval; אף אחת לא הוכרעה (F20) | "supervisor will look" | הטכניקה כן, התהליך לא |
| 11 | סיום ללא LLM מיותר | 3 מקרים חתומים בלי LLM (`hz_dce081882f3f`, `hz_b36f55115b42`) מול 30 עם | | | דוגמה לכך שאפשר; לא ברירת מחדל (F12) |
| 12 | פלט פנימי תקין, תוצאה למשתמש שגויה/לא ברורה | `hz_388c10cf3f84` (F3), 4 מקרי F1, הכפילות של F2, `hz_18a448b99627` (0.00 -> human_review עם supervisor note "high confidence") | | | לא |

## 9. משאבים ואופטימיזציה (בלי חומרה נוספת)

| הצעה | חיסכון צפוי (r23, 94 מקרים) | סיכון | אימות |
|---|---|---|---|
| F12: עובדים דטרמיניסטיים במקרים חתומים/floored | ~4,700 שניות (45% מהסבב), ~206 קריאות LLM | אפס לפסקי דין | replay: 58 פסקי דין זהים |
| F8: RAG LLM רק כשהרלוונטיות אינה direct, שליפה מטמון | ~1,700 שניות, ~120 קריאות | אפס | אותם ids |
| F7: Perception LLM רק על אות שהכללים לא קוראים | ~3,000 שניות, ~120 קריאות | 6 מקרים שבהם ה-LLM הוריד confidence: ייבדקו ידנית | replay |
| F9: ביטול קריאת ה-supervisor LLM | ~500 שניות | אפס | |
| F10: בלי סכמות כלים ב-Decision כש-hops=0 | ~10-15 שניות לקריאה (~860 טוקנים) | אפס | פרסינג על 40 תשובות |
| F6: קבלת final-in-tool-call ב-Intake | ~50 שניות למקרה אמיתי | טסט סבב 18 | replay 9 טרייסים |
| מקביליות | לא מומלץ: Ollama על 4 OCPU ARM הוא צוואר הבקבוק (המודל טעון פעם אחת); ביקשת בעבר לא להריץ 2 תמונות במקביל | | |
| סיום מוקדם | מתכסה ב-F12: ההכרעה ידועה אחרי sanitize ב-58/94 | | |

יחד: מקרה batch חציוני יורד מ-96 ל-~40 שניות, מקרה אמיתי מ-170 ל-~90, ללא שינוי בפסקי הדין של עץ הכללים ועם אותם floors.

## 10. המלצות לאישור, לפי סדר עדיפות

| # | סוג | שינוי | קבצים / רכיבים | מה ישתפר | בדיקות מונעות רגרסיה |
|---|---|---|---|---|---|
| 1 | תיקון באג | F1: needs_more_evidence מה-floor יוצר בקשת ראיה או הופך human_review; sweep ל-4 המקרים | `roadguard_agent.py` decision_node (בלוק סבב 25), `roadguard_api.py` `_run_graph_and_persist` | אין מקרים תקועים; האזרח לא מתבקש לשלוח לריק | `tests/test_round25_*`, טסט חדש: כל NME מחזיק request או סטטוס waiting |
| 2 | שיפור הצגה למשתמש | F2: TTL לקוח = שרת או שחזור מ-`/report`; כפתור "אין לי תמונה נוספת" על המסלול הקיים; הודעת כפילות ברורה | `webapp/assets/app.js` (856-942, 1129) | סבב הראיות נסגר; אין כפילויות | בדיקה ידנית ב-Chrome |
| 3 | תיקון באג | F11: שמירת `reporter_text` ב-submit-async + הצגה ב-digest | `roadguard_api.py` 6379, `_build_live_digest_html` | המפקח רואה מה האזרח כתב | `tests/test_auth_step1.py` |
| 4 | חיזוק אכיפה קיימת | F5: סף מרחק ל-floor של phash (10) | `roadguard_agent.py` 9508 | פחות 20 human_review לסבב ללא ראיה | `tests/test_round21_*`, 2 מקרי r13 |
| 5 | חיזוק אכיפה קיימת | F4: sufficiency כמידע ולא floor עד שהקורפוס מכיל SLA; בקשה לבני למסמך | `roadguard_agent.py` decision_node, בני | standard אפשרי על תמונות חדשות | `tests/test_round25_*` |
| 6 | אופטימיזציה | F12: עובדים דטרמיניסטיים במקרה חתום/floored | `profile_builder_node`, `rag_retriever_node` (דגל מהstate) | 45% זמן, 50% קריאות | replay r23 |
| 7 | שיפור פלט סוכן | F6: קבלת final-in-tool-call לפני דחיית כלי זר; extract_measurements רק עם מספרים | `_react_worker_rounds` 3840-3861 | Intake 70 -> 20 שניות, התוצאה לא נזרקת | `tests/test_round18_*`, `test_worker_autonomy.py` |
| 8 | אופטימיזציה | F8: RAG LLM מותנה + מטמון שליפה | `rag_retriever_node`, `decision_cache` | 1,700 שניות | `tests/test_rag_truth.py` |
| 9 | אופטימיזציה | F7: Perception LLM מותנה; recall_agent_notes מחוץ להיצע | `profile_builder_node` | 3,000 שניות | `tests/test_worker_autonomy.py` |
| 10 | שיפור הצגה למשתמש | F14: פירוט דטרמיניסטי קצר לאזרח + תוויות במקום slugs | `app.js` `renderUploadVerdict`, `loadMyReports` | המשתמש מבין מה זוהה ומה הלאה | Chrome |
| 11 | שיפור הצגה למשתמש | F3: סימון ריצה שנקטעה בשלושת הדפים; לא להפעיל מחדש שירות בזמן ריצה | `recover_interrupted_runs`, `CITIZEN_COPY`, digest, `rg-restart` | אמת למשתמש ולמפקח | |
| 12 | שיפור פלט סוכן | F10: בלי סכמות כשאין hops; תיקון JSON דטרמיניסטי | `_llm_decide_react`, `_parse_decision_json` | פחות unparseable, פרומפט קצר | `tests/test_round10_loop.py` |
| 13 | אופטימיזציה | F9: כיבוי supervisor LLM או חיבור ה-concern ל-Decision; regex; עלות נספרת | `_maybe_run_supervisor_agent`, `_CONF_WORD_RE`, `_compute_run_cost` | 8 דקות לסבב; חשבונאות מלאה | `tests/test_round20_*` |
| 14 | שיפור הצגה | F15: trace מהחדש לישן ומוגבל; Overview ב-/supervisor; תווית KPI | `_TRACE_HTML` replayHistory, `_SUPERVISOR_HTML`, `/stats/kpi` | דפי מפעיל נכונים ומהירים | Chrome |
| 15 | תיקון ניסוח | F13: דף How it works | `webapp/index.html` | הבטחות נכונות | |
| 16 | מדדים | F18: justified לפי תוצאה לא ריקה; ניקוי 2 שורות feedback מזויפות (שאילתת DELETE מפורשת, באישורך) | `_compute_per_agent_justification`, DB | מדדים כנים | `tests/test_round23_*`, `test_round26_ds_metrics.py` |
| 17 | בדיקה | F16: replay `hz_a7a8f0c54b02` לאיתור מקור priority 0 | | | |

עצרתי כאן. לא יושם שום שינוי. ממתין לאישורך ולבחירה מפורשת של המשימות.

### תוספת: מה אושר ובוצע באותו יום (6.11.2026 אחר הצהריים)

- בהוראתך: TTL של בקשת ראיה בדפדפן 10 דקות -> שעה (commit 533d1d4), סף הזיהוי ב-webapp 0.35 -> 0.25 (66fa80b).
- סבב 28 (fcfd821, פרוס): needs_more_evidence לעולם לא מגיע לאזרח. רצועה 0.4-0.6 -> human_review בעץ הכללים; RAG floor -> human_review (סוגר את F1); תקרת בקשות הראיה 0, כך שגם הצעה כזו של ה-LLM הופכת ל-human_review; policy.yaml 2026.11.06; 21 תרחישי הערכה ו-3 טסטים עודכנו; סוויטה 379. ששת המקרים התקועים נותבו למפקח ואין בקשות ראיה פתוחות. F2 אינו רלוונטי עוד למקרים חדשים (אין בקשות); הממצאים F4 (ה-floor קבוע על הקורפוס הזה), F5, F6, F7, F8, F9, F11, F12, F13, F14, F15 נשארים פתוחים לבחירתך.
- ריפו הצוות: workflow + עותק מסונכרן של ה-webapp נדחפו (af9f39a). ה-Pages עצמו דורש הפעלה על ידי בעל הריפו (Settings > Pages > Source: GitHub Actions); ה-token של ה-workflow אינו יכול להפעיל אותו ולך אין admin שם. ה-origin של ריפו הצוות נוסף ל-CORS בשרת ול-OAuth client בקונסולת Google.

נספח: צילומי מסך מהבדיקה (6.11, Chrome) נשמרו בתיקיית ה-tool-results של השיחה: webapp #how, /roadguard/trace, /digest/latest (Cases), /roadguard/flow.

### תוספת 2: מה יושם ב-7.11.2026 (אישורך: "כן, לפי הסדר")

כל 17 ההמלצות של סעיף 10 טופלו; הכל ב-main (dbeeee2, c7194f5, 49c31fd, 763d41a, cd70adf, 47c664f), פרוס ב-VM ב-7.11 ללא ריצה פעילה, סוויטה 426.

- F11: `reporter_text` נשמר ב-submit-async ומוצג ב-digest ("Reporter wrote"); ה-digest קורא את ה-DB המוגדר ולא נתיב קבוע.
- F5: רצפת phash נופלת רק עד מרחק 10 (`known_image.phash_floor_max_distance`); שכנים רחוקים מצוטטים כהקשר.
- F4: לפי בחירתך נשאר כמו היום: `standard` על `insufficient_coverage` עובר למפקח, עד שהקורפוס יכיל מסמך SLA עירוני (הבקשה לבני פתוחה).
- F12: `sanitize` מסמן מקרה שהקטלוג קבע (`deterministic_workers_reason`) ושלושת העובדים מדלגים על ה-LLM.
- F6: לפי בחירתך, תשובה סופית של עובד שהגיעה כקריאת כלי מתקבלת; `extract_measurements` מוצע רק כשיש ספרה בטקסט.
- F8: ה-LLM של Research רץ רק כשהכיסוי הדטרמיניסטי אינו direct; מטמון שליפה לשעה לפי מחלקה וסוג כביש.
- F7: ה-LLM של Perception רץ רק על טקסט מדווח, רצועה 0.4-0.6 או יותר ממחלקה אחת; `recall_agent_notes` מוצע רק כשיש לקחים.
- F14: פירוט דטרמיניסטי לאזרח (זוהה, ביטחון במילים, חומרה, הצעד הבא, למה) ותוויות בטבלת הדיווחים.
- F3: ריצה שנקטעה מסומנת ב-`/report`, ב-digest ובהודעה לאזרח; `rg-restart` מסרב להפעיל מחדש בזמן ריצה.
- F10: בלי סכמות כלים כשהתקרה מתחת ל-2; תיקון JSON דטרמיניסטי לפני unparseable.
- F9: לפי בחירתך, `supervisor_note` נכנס ל-payload של Decision; "with high confidence" נתפס; קריאת ה-Supervisor נספרת בעלות.
- F15: trace מהחדש לישן (24 שעות, עד 20); Overview בעמוד המפקח סופר את הצי ב-24 השעות; ה-KPI נקרא `rule_tree_agreement`.
- F13: דף "How it works" תואם לקוד.
- F18: כלי נספר כערך רק כשהחזיר תוצאה; שתי שורות ה-feedback המזויפות נמחקו (באישורך).
- F16: המקור ל-priority 0 ב-hz_a7a8f0c54b02: סבב התיקון החזיר 0 אחרי שהמילוי הראשון כבר רץ; המילוי רץ עכשיו גם אחרי התיקון.

אימות אחרי הפריסה (Chrome, 7.11): דף ה-trace מציג "replaying 7 hazard(s) from the last 24 hours, newest first"; עמוד המפקח "Total (last 24h): 7"; ב-digest המקרה hz_388c10cf3 מסומן interrupted; האתר האישי ואתר הצוות מגישים את ה-webapp עם הסיכום לאזרח.

</div>
