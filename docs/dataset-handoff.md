<div dir="rtl" align="right">

# RoadGuard · בקשת מידע לחיבור הסוכנים ל-batch

**מטרה**: לאפשר הרצת שלושת הסוכנים (Perception → Research → Decision) על ה-test set של המאגר המתויג, ולהפיק את מספר ההצלחה שה-Charter מגדיר (85% החלטות נכונות, 90% נימוקים מעוגנים).

**קהל יעד**: Shai (CV) · Ovad (Data)

**מועד נדרש**: לפני הפגישה הקרובה, שם נסגור את הפרטים.

---

## 1 · נדרש מ-Shai (CV)

### קובץ אחד
`yolo_predictions.csv` (או JSON — פורמט לא קריטי כל עוד השדות קיימים).

### שדות חובה

| שדה | סוג | ערכים מותרים | הערה |
|---|---|---|---|
| `image_id` | string | שם קובץ מהמאגר, לדוגמה `Japan_000847.jpg` | מפתח join עם Ovad |
| `top1_class` | string | `D00` · `D10` · `D20` · `D40` | ממופה כבר ל-RDD2022 lexicon |
| `top1_confidence` | float | 0.0 - 1.0 | ביטחון של המחלקה שה-YOLO הכי בטוח בה |

### שדות מומלצים (לא חובה, אבל משפרים איכות)

| שדה | סוג | הערה |
|---|---|---|
| `top2_class` | string | המחלקה השנייה אחרי `top1_class`. בלעדי זה - top1_margin ייקבע 0 |
| `top2_confidence` | float | הביטחון בה. `top1_margin = top1_confidence - top2_confidence` |
| `bbox` | string או 4 מספרים | `x1,y1,x2,y2` של המסגרת סביב המפגע |
| `image_width` · `image_height` | int | מידות התמונה בפיקסלים · מאפשר לחשב `area_ratio` |

### פורמט דוגמה (CSV)

```csv
image_id,top1_class,top1_confidence,top2_class,top2_confidence,bbox,image_width,image_height
Japan_000847.jpg,D40,0.91,D20,0.42,"120,240,340,460",1280,720
Norway_000017.jpg,D20,0.73,D10,0.58,"80,320,210,410",800,600
India_002104.jpg,D00,0.65,,,"",960,540
```

### הנחיות

1. **רק 4 מחלקות**: `D00` / `D10` / `D20` / `D40`. אם המודל מחזיר שמות ידידותיים (`Pothole`, `AlligatorCrack`...), למפות אותם לפי `webapp/assets/config.js` וטבלת המיפוי ב-`src/tools/download_dataset.py`.
2. **רק test set**: לא train, לא validation. מניעת leakage.
3. **שורה אחת לכל תמונה** — אם YOLO זיהה כמה detections בתמונה, להחזיר רק את ה-top-1 (המחלקה החזקה ביותר). אם רוצים לייצא גם את ה-detections הנוספים — להוסיף קובץ נפרד `yolo_detections_all.csv` עם עמודה `detection_index`.
4. **ללא תמונות גולמיות**: רק הפלט המספרי. התמונות עצמן לא נדרשות לסוכן.

---

## 2 · נדרש מ-Ovad (Data)

### שני קבצים

**קובץ 1 — ground truth**: `ground_truth.csv`
התיוג האמיתי של המחלקה לכל תמונה. חילוץ ישיר מה-annotations של RDD2022.

| שדה | סוג | ערכים מותרים |
|---|---|---|
| `image_id` | string | אותו מפתח כמו ב-Shai |
| `true_class` | string | `D00` · `D10` · `D20` · `D40` |

**דוגמה**:
```csv
image_id,true_class
Japan_000847.jpg,D40
Norway_000017.jpg,D20
India_002104.jpg,D00
```

**קובץ 2 — eval enriched set**: `eval_set.csv`
50-200 תמונות שתוייגו ידנית בשלושת השדות שחסרים ב-RDD2022, בתוספת ההחלטה האמיתית שסוכן טוב היה מקבל.

| שדה | סוג | ערכים מותרים | הערה |
|---|---|---|---|
| `image_id` | string | אותו מפתח | חייב להיות קיים גם ב-Shai |
| `road_type` | string | `main` · `secondary` · `bike_path` · `parking` | סוג הכביש בתמונה |
| `lane_blocked` | string | `none` · `partial` · `full` | חסימת נתיב |
| `expected_action` | string | `urgent` · `standard` · `needs_more_evidence` · `human_review` | ההחלטה הנכונה |
| `kind` | string | `simple` · `mid` · `hard` | רמת קושי · 20 simple, 25 mid, 5 hard לפי החלוקה של 50 |
| `rationale` | string | טקסט חופשי, קצר | למה זאת ההחלטה הנכונה - עוזר ל-debug |

**דוגמה**:
```csv
image_id,road_type,lane_blocked,expected_action,kind,rationale
Japan_000847.jpg,main,partial,urgent,simple,D40 pothole blocking a lane on arterial road
Norway_000017.jpg,secondary,none,standard,simple,D20 cracking on secondary road routine queue
India_002104.jpg,main,none,needs_more_evidence,mid,low-confidence D00 with no corroboration
```

### חלוקה המומלצת של 50 תסריטים (לפי Charter §2)

| expected_action | כמות | אפיון |
|---|---|---|
| `urgent` | 20 | D40 חוסם נתיב בכביש ראשי · confidence ≥ 0.6 |
| `standard` | 15 | D20 בכביש משני · confidence ≥ 0.6 |
| `needs_more_evidence` | 10 | confidence בין 0.4-0.6 · ללא corroboration |
| `human_review` | 5 | edge cases: תמונה גרועה, מחלוקת בין פרוצדורות, confidence < 0.4 |

**עדיפות**: אם זמן קצר, 50 תסריטים זאת הכמות המחייבת לעמוד ב-Charter. 100-200 יוסיפו רובסטיות סטטיסטית.

---

## 3 · איך הקבצים מתחברים לסוכן

**join key**: `image_id`.

```
yolo_predictions.csv  (Shai)
         ⋈ image_id
ground_truth.csv      (Ovad · כל התמונות)
         ⋈ image_id
eval_set.csv          (Ovad · 50-200 תמונות)
         ↓
     adapter → Perception → Research → Decision
         ↓
     predicted_action per image
         ↓
  compare to expected_action → success rate
```

**הסקריפט שמריץ את הסוכן** נמצא ב-`src/agent/multiagent/measure.py`. אני מתאים את ה-loader לפורמט הזה. אתם רק מספקים את הקבצים.

---

## 4 · מגבלות ואילוצים

- **4 מחלקות בלבד** (`D00`/`D10`/`D20`/`D40`) · ה-Charter §3 מגדיר זאת מפורשות. לא D30, לא Pothole-Big וכו'.
- **4 פעולות בלבד** (`urgent`/`standard`/`needs_more_evidence`/`human_review`) · ה-Charter §5.
- **אין inference של עומק/חומרה הנדסית**. ה-priority מחושב מ-class/confidence/road/lane, שום דבר אחר (Charter §3: *"Bounding-box area will not be presented as physical damage area or true severity"*).

---

## 5 · איך למסור

- להעלות את הקבצים לתיקייה `data/batch_eval/` ברפו.
- לפתוח PR קטן עם הקבצים · אני מריץ validation אוטומטית ומאשר.

### שמות הקבצים הסופיים
```
data/batch_eval/yolo_predictions.csv    (Shai)
data/batch_eval/ground_truth.csv        (Ovad)
data/batch_eval/eval_set.csv            (Ovad)
```

---

**שאלות? פתח thread בקבוצת הצוות. אני זמין לעבור על דוגמת שורה אחת כדי לוודא שהפורמט תקין לפני שאתם מריצים על כל ה-test set.**

</div>
