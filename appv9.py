from flask import Flask, render_template, request
import os
import re
import sqlite3
import fitz
import spacy
import category
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

nlp_model = spacy.load(os.path.join(os.getcwd(), "resume_ner_model"))
nlp_loc_model = spacy.load("en_core_web_trf")



def init_db():
    conn = sqlite3.connect('resumes.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            filename TEXT,
            degree TEXT,
            graduation_year INTEGER,
            skills TEXT,
            places TEXT,
            category TEXT,
            similarity REAL,
            similarity_percentile REAL,
            composite_score REAL,
            composite_percentile REAL,
            threshold_score INTEGER,
            selection_status TEXT,
            phone TEXT,
            email TEXT,
            github TEXT,
            linkedin TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()


def save_resume_to_db(resume):
    conn = sqlite3.connect('resumes.db')
    cursor = conn.cursor()
    def stringify(value):
        if isinstance(value, list):
            return ', '.join(value)
        return str(value or '')
    cursor.execute('''
        INSERT INTO resumes (
            name, filename, degree, graduation_year,
            skills, places, category, similarity,
            similarity_percentile, composite_score,
            composite_percentile, threshold_score,
            selection_status, phone, email, github, linkedin
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        stringify(resume.get('name')),
        stringify(resume.get('filename')),
        stringify(resume.get('degree')),
        resume.get('GRAD_YEAR'),
        stringify(resume.get('special_skills')),
        stringify(resume.get('places')),
        resume.get('category'),
        float(resume.get('SIMILARITY', 0.0)),
        float(resume.get('similarity_percentile', 0.0)),
        float(resume.get('composite_score', 0.0)),
        float(resume.get('composite_percentile', 0.0)),
        int(resume.get('threshold_score', 3)),
        resume.get('selection_status', 'Not Selected'),
        resume.get('phone', ''),
        resume.get('email', ''),
        resume.get('github', ''),
        resume.get('linkedin', '')
    ))
    conn.commit()
    conn.close()


def extract_email(text):
    pattern = re.compile(r'[a-zA-Z0-9+._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    matches = pattern.findall(text)
    return matches[0] if matches else ''


def extract_phone(text):
    pattern = re.compile(r'(?:\+91[\-\s]?|0)?[6-9]\d{9}')
    matches = pattern.findall(text)
    return matches[0] if matches else ''


def extract_linkedin(text):
    pattern = re.compile(r'(https?://www\.linkedin\.com/in/[A-Za-z0-9\-_%]+)/?')
    matches = pattern.findall(text)
    return matches[0] if matches else ''


def extract_github(text):
    pattern = re.compile(r'(https?://github\.com/[A-Za-z0-9\-_%]+)/?')
    matches = pattern.findall(text)
    return matches[0] if matches else ''


def extract_places(text):
    doc = nlp_loc_model(text)
    places = {ent.text.strip() for ent in doc.ents if ent.label_ in ('GPE','LOC','FAC')}
    return list(places)


def extract_graduation_year(text):
    pattern = re.compile(
        r'(?:graduat(?:ed|ion)\s*(?:year\s*)?|class\s+of\s*)?\b(19|20)\d{2}\b',
        flags=re.IGNORECASE
    )
    years = [int(m.group(0)) for m in pattern.finditer(text)]
    current_year = datetime.now().year
    valid = [y for y in years if 1950 <= y <= current_year]
    return max(valid) if valid else None


def getNameAndOtherDetails(file_path):
    doc = fitz.open(file_path)
    text = ' '.join(page.get_text() for page in doc)
    grad_year = extract_graduation_year(text)
    ner_doc = nlp_model(text)
    label_data = {}
    for ent in ner_doc.ents:
        lbl = ent.label_.upper()
        txt = ent.text.strip()
        if lbl in label_data:
            if isinstance(label_data[lbl], list):
                label_data[lbl].append(txt)
            else:
                label_data[lbl] = [label_data[lbl], txt]
        else:
            label_data[lbl] = txt
    label_data['PLACES']    = extract_places(text)
    label_data['GRAD_YEAR'] = grad_year
    return label_data

career_fields = {
    0: 'Advocate', 1: 'Arts', 2: 'Automation Testing', 3: 'Blockchain',
    4: 'Business Analyst', 5: 'Civil Engineer', 6: 'Data Science',
    7: 'Database', 8: 'DevOps Engineer', 9: 'DotNet Developer',
    10: 'ETL Developer', 11: 'Electrical Engineering', 12: 'HR',
    13: 'Hadoop', 14: 'Health and fitness', 15: 'Java Developer',
    16: 'Mechanical Engineer', 17: 'Network Security Engineer',
    18: 'Operations Manager', 19: 'PMO', 20: 'Python Developer',
    21: 'SAP Developer', 22: 'Sales', 23: 'Testing', 24: 'Web Designing'
}
label_to_index = {str(lbl): idx for idx, lbl in enumerate(category.loaded_model.classes_)}

def getCategory(key):
    if isinstance(key, str):
        if key in label_to_index:
            key = label_to_index[key]
        else:
            try:
                key = int(key)
            except (ValueError, TypeError):
                return 'Unknown'
    try:
        key = int(key)
    except (ValueError, TypeError):
        return 'Unknown'
    return career_fields.get(key, 'Unknown')


def matchJob_Description(filesList, app, job_description):
    special_skills_input = request.form.get('special_skills','').lower().strip()
    try:
        threshold_score = int(request.form.get('threshold_score','3'))
    except ValueError:
        threshold_score = 3

    tfidf_vectorizer = TfidfVectorizer()
    job_desc_vec = tfidf_vectorizer.fit_transform([job_description])

    ranked = []
    for f in filesList:
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)
        raw     = category.extract_text_from_pdf(path)
        cleaned = category.cleanResume(raw)
        ner     = getNameAndOtherDetails(path)

        email    = extract_email(raw)
        phone    = extract_phone(raw)
        linkedin = extract_linkedin(raw)
        github   = extract_github(raw)

        sim = cosine_similarity(
            job_desc_vec,
            tfidf_vectorizer.transform([raw])
        )[0][0] * 100

        has_skill = 1 if special_skills_input in [s.lower() for s in ner.get('SKILL', [])] else 0
        pred = category.loaded_model.predict(
            category.loaded_tfidf.transform([cleaned])
        )

        ranked.append({
            'SIMILARITY': sim,
            'filename':   f.filename,
            'name':       ner.get('PERSON','No Name'),
            'degree':     ner.get('EDU','No Degree'),
            'GRAD_YEAR':  ner.get('GRAD_YEAR'),
            'special_skills': ner.get('SKILL', []),
            'places':         ner.get('PLACES', []),
            'catPredict':     pred,
            'has_special_skill': has_skill,
            'threshold_score':   threshold_score,
            'email': email,
            'phone': phone,
            'linkedin': linkedin,
            'github': github
        })

    total = len(ranked)
    sims  = [r['SIMILARITY'] for r in ranked]
    comps = []
    for r in ranked:
        r['similarity_percentile'] = sum(1 for x in sims if x <= r['SIMILARITY'])/total*100
        comp = (r['similarity_percentile'] * (threshold_score/5.0)
               + r['has_special_skill'] * ((6-threshold_score)*10))
        r['composite_score'] = comp
        comps.append(comp)

    for r in ranked:
        r['composite_percentile'] = sum(1 for x in comps if x <= r['composite_score'])/total*100

    cutoff = 50 + (threshold_score-3)*10
    for r in ranked:
        r['selection_status'] = 'Selected' if r['composite_percentile'] >= cutoff else 'Not Selected'
        r['category']         = getCategory(r['catPredict'][0])
        save_resume_to_db(r)

    return sorted(ranked, key=lambda x: x['composite_score'], reverse=True)


@app.route('/', methods=['GET','POST'])
def upload_resumes():
    results = []
    if request.method == 'POST':
        files = request.files.getlist('resumes')
        jd    = request.form.get('job_description','')
        results = matchJob_Description(files, app, jd)
    return render_template('index.html', files=results)

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('resumes.db')
    cur  = conn.cursor()
    cur.execute('''
        SELECT name, filename, degree, graduation_year,
               skills, places, category,
               similarity, composite_score, threshold_score, selection_status
        FROM resumes
    ''')
    rows = cur.fetchall()
    conn.close()

    degrees      = [r[2] or 'Unknown' for r in rows]
    grad_years   = [r[3] for r in rows if r[3]]
    skills_list  = [s.strip() for r in rows for s in (r[4] or '').split(',') if s.strip()]
    places_list  = [p.strip() for r in rows for p in (r[5] or '').split(',') if p.strip()]
    categories   = [r[6] or 'Unknown' for r in rows]
    similarities = [r[7] for r in rows]
    composites   = [r[8] for r in rows]
    thresholds   = [r[9] for r in rows]
    selections   = [r[10] for r in rows]

    stats = {
        'degree_counts':    Counter(degrees),
        'grad_year_counts': Counter(grad_years),
        'skill_counts':     Counter(skills_list),
        'place_counts':     Counter(places_list),
        'category_counts':  Counter(categories),
        'selection_counts': Counter(selections),
        'threshold_counts': Counter(thresholds),
        'similarities':     similarities,
        'composites':       composites
    }

    return render_template('dashboard.html', stats=stats)

@app.route('/resumes')
def view_resumes():
    conn = sqlite3.connect('resumes.db')
    cur  = conn.cursor()
    cur.execute('SELECT * FROM resumes ORDER BY composite_score DESC')
    data = cur.fetchall()
    conn.close()
    return render_template('resumes.html', resumes=data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
