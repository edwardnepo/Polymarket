# חדשות מול Polymarket

אב-טיפוס מחקרי שבודק האם סנטימנט בכתבות חדשותיות קשור לתנועות הסתברות בשווקי Polymarket, והאם ניתן להשתמש בו לחיזוי תנועה עתידית.

## שאלת המחקר

**האם סנטימנט בכתבות חדשותיות קשור לתנועות הסתברות בשווקי Polymarket, והאם ניתן להשתמש בו לחיזוי תנועה עתידית?**

הפרויקט הנוכחי מתמקד בכתבות חדשותיות + נתוני Polymarket בלבד. Twitter/X ו-Reddit אינם ממומשים באב-הטיפוס הנוכחי.

## מה המערכת עושה

- אוספת כתבות RSS ממקורות בינלאומיים וישראליים.
- מנתחת סנטימנט באמצעות מודל NLP.
- אוספת שווקים והיסטוריית הסתברויות מ-Polymarket.
- מחשבת מתאם Pearson.
- בונה דאטהסט Machine Learning.
- מאמנת מודל חיזוי ומשווה אותו ל-baseline.
- מציגה מסקנות, ניתוח מקורות תקשורת ותחזיות בדשבורד Streamlit.

## צילום מסך

_כאן ניתן להוסיף צילום מסך של הדשבורד לאחר הרצה מקומית._

## התקנה והרצה מהירה

```bash
cd Polymarket_Geopolitics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pipeline.run_pipeline --dry-run
streamlit run frontend/app.py
```

## הרצת הדשבורד בלבד

אם קיימים קבצי דמו בתיקיית `output/`, ניתן להריץ:

```bash
streamlit run frontend/app.py
```

אין צורך להגדיר `PYTHONPATH=.`. קובץ הדשבורד מוסיף את תיקיית הפרויקט לנתיב הייבוא באופן אוטומטי, כך שהפקודה עובדת גם מקומית וגם ב-Streamlit Community Cloud.

הדשבורד יטען את הקבצים:

- `output/articles.json`
- `output/markets.json`
- `output/run.json`
- `output/ml_results.json`
- `output/ml_dataset.json`

במצב זה אין צורך ב-Firebase.

## חידוש נתונים

```bash
python -m pipeline.run_pipeline --dry-run
```

פקודה זו מריצה את כל ה-pipeline וכותבת snapshot מקומי ל-`output/`.

## מה רואים בדשבורד

- סיכום מנהלים.
- סיכום תוצאות אוטומטי.
- ניתוח לפי כלי תקשורת.
- תחזית עתידית לפי המודל.
- ניתוח מתקדם עם Pearson, מפות חום, חשיבות מאפיינים, טבלאות מקור ונתוני גלם.

## מצב דמו ללא Firebase

Firebase הוא אופציונלי. לצורך בדיקת מורה, הדשבורד יכול לעבוד ישירות מקבצי `output/*.json` ולהציג:

**מצב דמו — נתונים מקומיים**

## אזהרת סודות

אין להעלות ל-GitHub:

- `.env`
- `firebase_creds.json`
- כל קובץ secrets אחר

קבצים אלה נמצאים ב-`.gitignore`.

## פריסה עתידית

הפרויקט מוכן להעלאה ידנית ל-GitHub ולפריסה ב-Streamlit Community Cloud.

הגדרות מומלצות:

- App entry point: `frontend/app.py`
- Dependencies: `requirements.txt`
- Demo data: `output/*.json`
- Secrets: לא להעלות ל-GitHub; להגדיר בנפרד רק אם רוצים Firebase.
