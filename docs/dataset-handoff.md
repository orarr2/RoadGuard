<div dir="rtl" align="right">

# RoadGuard · בקשת מידע לחיבור הסוכנים ל-batch

**מטרה**: לאפשר הרצת שלושת הסוכנים (Perception → Research → Decision) על ה-test set של המאגר המתויג, ולהפיק את מספר ההצלחה שה-Charter מגדיר (85% החלטות נכונות, 90% נימוקים מעוגנים).

**קהל יעד**: Shai (CV) · Ovad (Data)

**מועד נדרש**: לפני הפגישה הקרובה, שם נסגור את הפרטים.

---

## 1 · הסוכנים — מי הם ומה כל אחד עושה

לפני שאתם רואים את רשימת השדות, חשוב להבין מה המערכת עושה איתם. יש שלושה סוכנים שמחליטים בטור, אחד אחרי השני:

**Perception Agent** · *סוכן התפיסה* — ה"עיניים". מקבל את פלט YOLO של Shai ואת המטא-נתונים מהטופס, ובונה *פרופיל מפגע מובנה*: סוג המפגע, רמת ביטחון מערכתית (לא רק של YOLO), דרגת חומרה ראשונית, האם זה מפגע חוזר, ואזהרות על איכות התמונה. לא מחליט כלום — רק מכין לסוכנים הבאים תמונת מצב מסודרת.

**Research Agent** · *סוכן המחקר* — ה"ספרן". לוקח את הפרופיל ושולף מה-RAG של Benny את הנהלים הרלוונטיים (לדוגמה: *"בור על כביש ראשי → תיקון תוך 24 שעות, מקור: iroads.co.il/..."*). חייב להיות **קריאה בלבד**. לא מחליט, לא כותב, לא פועל.

**Decision Agent** · *סוכן ההחלטה* — ה"פה". הסוכן היחיד שמותר לו להחליט ולפעול. מקבל את הפרופיל מ-Perception ואת הנהלים מ-Research, ובוחר אחת מארבע פעולות: `urgent` / `standard` / `needs_more_evidence` / `human_review`. חייב לצטט את ה-ID של הנוהל שהסתמך עליו — אסור להמציא.

---

## 2 · נדרש מ-Shai (CV) · קובץ `yolo_predictions.csv`

**מה זה מכיל**: הפלט המספרי של YOLO על כל תמונה ב-test set. **לאיזה סוכן זה הולך**: כל השדות האלה מוזנים ל-Perception Agent — זה הקלט החזותי שלו, הוא רואה את זה ולא את התמונה עצמה.

### שדות חובה

| שדה | סוג | ערכים | למה זה נדרש |
|---|---|---|---|
| `image_id` | string | שם קובץ (`Japan_000847.jpg`) | מפתח חיבור בין הקבצים · לא עובר לסוכן, רק משמש לקישור ל-ground truth של Ovad |
| `top1_class` | string | `D00` · `D10` · `D20` · `D40` | **לב ההחלטה** · Perception קורא את זה; Decision משתמש בו כדי לקבוע חומרה (D40 חמור יותר מ-D00) ולשלוף מ-RAG את הנוהל הנכון |
| `top1_confidence` | float | 0.0 - 1.0 | **קובע את ההחלטה**: מתחת ל-0.4 → human_review אוטומטי; בין 0.4-0.6 → needs_more_evidence; מעל 0.6 → פעולה מלאה |

### שדות מומלצים (לא חובה, אבל משפרים איכות ההחלטה)

| שדה | סוג | למה זה נדרש |
|---|---|---|
| `top2_class` | string | מאפשר חישוב `top1_margin = top1_conf - top2_conf`. Perception משתמש בו כדי לדעת אם המודל "מתלבט" בין שתי מחלקות (margin קטן = לא בטוח). בלעדיו Perception יחשב ביטחון רק מ-top1 |
| `top2_confidence` | float | ראה למעלה · משמש רק בצירוף עם `top2_class` |
| `bbox` | `x1,y1,x2,y2` | Perception מחשב ממנו `area_ratio` (בשילוב עם `image_width/height`). פרוקסי לגודל יחסי של המפגע בתמונה — לא לעומק הנדסי |
| `image_width` · `image_height` | int | חישוב `area_ratio` (ראה למעלה) |

### פורמט דוגמה (CSV)

```csv
image_id,top1_class,top1_confidence,top2_class,top2_confidence,bbox,image_width,image_height
Japan_000847.jpg,D40,0.91,D20,0.42,"120,240,340,460",1280,720
Norway_000017.jpg,D20,0.73,D10,0.58,"80,320,210,410",800,600
India_002104.jpg,D00,0.65,,,"",960,540
```

### הנחיות מחייבות

1. **רק 4 מחלקות**: `D00` · `D10` · `D20` · `D40`. אם המודל מחזיר שמות ידידותיים (`Pothole`, `AlligatorCrack`), למפות לפי הטבלה ב-`webapp/assets/config.js` ו-`src/tools/download_dataset.py`.
2. **רק test set**: לא train, לא validation. מניעת leakage.
3. **שורה אחת לכל תמונה**: רק ה-detection החזק ביותר. אם יש detections נוספים ורוצים להעביר אותם — קובץ נפרד `yolo_detections_all.csv` עם `detection_index`.
4. **ללא תמונות גולמיות**: רק טקסט/מספרים. התמונות עצמן לא נשלחות לסוכן.

---

## 3 · נדרש מ-Ovad (Data) · שני קבצים

### קובץ 1 — `ground_truth.csv`

**מה זה מכיל**: המחלקה האמיתית של כל תמונה, שחולצה מה-annotations המקוריים של RDD2022.

**לאיזה סוכן זה הולך**: **לאף סוכן**. השדה הזה הוא רק לצורך *מדידה* — אנחנו משווים את מה שסוכן ה-Decision החליט מול המחלקה האמיתית כדי לדעת אם הוא צדק. בלי זה אין איך למדוד הצלחה.

| שדה | סוג | ערכים | למה זה נדרש |
|---|---|---|---|
| `image_id` | string | אותו מפתח כמו ב-Shai | חיבור |
| `true_class` | string | `D00` · `D10` · `D20` · `D40` | המחלקה הנכונה · משמש להשוואה מול `top1_class` של YOLO ומול ההחלטה של הסוכן |

**דוגמה**:
```csv
image_id,true_class
Japan_000847.jpg,D40
Norway_000017.jpg,D20
India_002104.jpg,D00
```

---

### קובץ 2 — `eval_set.csv`

**מה זה מכיל**: 50-200 תמונות שתוייגו ידנית בשלושה שדות שחסרים ב-RDD2022, בתוספת ההחלטה האמיתית הנכונה. **הקובץ הזה חובה** — בלעדיו אי אפשר למדוד את הסוכן.

**לאיזה סוכן זה הולך**: שני סוגי שדות שונים —
- `road_type` + `lane_blocked` → **מטא-נתונים** שעוברים ל-Perception וגם ל-Decision (Perception משתמש בחסימה כדי לקבוע חומרה, Decision משתמש בשניהם לנוסחת התעדוף).
- `expected_action` + `kind` + `rationale` → **רק למדידה**. לא עוברים לסוכן, משמשים כדי לדעת אם הסוכן צדק.

| שדה | סוג | ערכים | הולך לסוכן? | למה זה נדרש |
|---|---|---|---|---|
| `image_id` | string | אותו מפתח | — | חיבור |
| `road_type` | string | `main` · `secondary` · `bike_path` · `parking` | ✅ Perception + Decision | סוג הכביש · קובע את ה-priority (ראשי = +25, חניה = +5) ואת הנוהל שה-Research ישלוף |
| `lane_blocked` | string | `none` · `partial` · `full` | ✅ Perception + Decision | חסימת נתיב · קובע urgency (חסימה מלאה בכביש ראשי = urgent אוטומטי) |
| `expected_action` | string | `urgent` · `standard` · `needs_more_evidence` · `human_review` | ❌ מדידה בלבד | ההחלטה הנכונה · משווים מול מה שסוכן ה-Decision החליט |
| `kind` | string | `simple` · `mid` · `hard` | ❌ מדידה בלבד | רמת קושי · משמש לפילוח תוצאות (האם הסוכן טוב גם ב-hard?) |
| `rationale` | string | טקסט חופשי קצר | ❌ מדידה בלבד | הסבר למה זאת ההחלטה הנכונה · עוזר ב-debug כשהסוכן טועה |

**דוגמה**:
```csv
image_id,road_type,lane_blocked,expected_action,kind,rationale
Japan_000847.jpg,main,partial,urgent,simple,D40 pothole blocking a lane on arterial road
Norway_000017.jpg,secondary,none,standard,simple,D20 cracking on secondary road routine queue
India_002104.jpg,main,none,needs_more_evidence,mid,low-confidence D00 with no corroboration
```

### חלוקה מומלצת של 50 תסריטים (לפי Charter §2)

| expected_action | כמות | אפיון טיפוסי |
|---|---|---|
| `urgent` | 20 | D40 חוסם נתיב בכביש ראשי · confidence ≥ 0.6 |
| `standard` | 15 | D20 בכביש משני · confidence ≥ 0.6 |
| `needs_more_evidence` | 10 | confidence בין 0.4-0.6 · ללא corroboration |
| `human_review` | 5 | edge cases: תמונה גרועה, מחלוקת בין נהלים, confidence < 0.4 |

**עדיפות**: אם הזמן קצר, 50 תסריטים זו הכמות המחייבת לעמוד ב-Charter. 100-200 יוסיפו רובסטיות סטטיסטית.

---

## 4 · איך הקבצים מתחברים לסוכנים

```
yolo_predictions.csv  (Shai)
   top1_class, top1_confidence, bbox, image_dims   ──┐
                                                      ▼
eval_set.csv          (Ovad)                     Perception Agent
   road_type, lane_blocked                         (בונה פרופיל)
                                                      │
                                                      ▼
                                                 Research Agent
                                                 (שולף נהלים)
                                                      │
                                                      ▼
                                                 Decision Agent
                                                 (בוחר פעולה)
                                                      │
                                                      ▼
                                                 predicted_action
                                                      │
                                                      ▼
    expected_action (Ovad eval_set) ⟵ השוואה ⟶  predicted_action
                                                      │
                                                      ▼
                                              success_rate (%)
```

**join key**: `image_id` בכל הקבצים.

**הסקריפט שמריץ**: `src/agent/multiagent/measure.py`. אני מתאים את ה-loader לפורמט הזה. אתם רק מספקים את הקבצים.

---

## 5 · מגבלות ואילוצים מה-Charter

- **4 מחלקות בלבד** (`D00`/`D10`/`D20`/`D40`) · Charter §3 מגדיר זאת מפורשות. לא D30, לא Pothole-Big וכו'.
- **4 פעולות בלבד** · Charter §5.
- **אין inference של עומק/חומרה הנדסית**. ה-priority מחושב מ-class + confidence + road + lane בלבד (Charter §3: *"Bounding-box area will not be presented as physical damage area or true severity"*).

---

## 6 · איך למסור

- להעלות את הקבצים לתיקייה `data/batch_eval/` ברפו.
- לפתוח PR קטן עם הקבצים · אני מריץ validation אוטומטית ומאשר.

### שמות הקבצים הסופיים

```
data/batch_eval/yolo_predictions.csv    (Shai)
data/batch_eval/ground_truth.csv        (Ovad)
data/batch_eval/eval_set.csv            (Ovad)
```

---

**שאלות?** פתח thread בקבוצת הצוות. אני זמין לעבור על דוגמת שורה אחת כדי לוודא שהפורמט תקין לפני שאתם מריצים על כל ה-test set.

</div>
