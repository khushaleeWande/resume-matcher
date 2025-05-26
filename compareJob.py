import spacy
import PyPDF2
import category 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
import fitz
import os

csv_filename = "ranked_resumes.csv"

# nlp = spacy.blank('en')
nlp_model = spacy.load('resume_ner_model')
nlp_ner_model = spacy.load("en_core_web_sm")

job_description = "Develop"

def getNameAndOtherDetails(fileName) :
    doc = fitz.open(fileName)
    label_data = {}
    text = ""

    for page in doc:
        text += page.get_text()  
        tx = " ".join(text.split('\n'))
        print(tx)
        
    doc = nlp_model(text)

    for ent in doc.ents:
        label = ent.label_.upper()
        text = ent.text.strip() 
        print(f'{ent.label_.upper():{30}}- {ent.text}')
 
        if label in label_data:
            if isinstance(label_data[label], list):
                label_data[label].append(text)
            else:
                label_data[label] = [label_data[label], text]
        else:
            label_data[label] = text 

    return label_data



def matchJob_Description(filesList,app,job_description) :
    tfidf_vectorizer = TfidfVectorizer()
    job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

    # Rank resumes based on similarity
    ranked_resumes = []
    for resume_path in filesList:
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_path.filename)
        resume_path.save(file_path)

        resume_text = category.extract_text_from_pdf(file_path)
        cleaned_text = category.cleanResume(resume_text)
        extractedNER_Data = getNameAndOtherDetails(file_path)

        NAME = extractedNER_Data.get("NAME", "No Name Found")
        DEGREE = extractedNER_Data.get("DEGREE", "No Name Found")
        SKILLS = extractedNER_Data.get("SKILLS", "No Name Found")
        LANGUAGES_KNOWN = extractedNER_Data.get("LANGUAGES KNOWN", "No Name Found")
        graduation_year = extractedNER_Data.get("GRADUATION YEAR", "No Name Found")

        if isinstance(extractedNER_Data, str):
            print("Error")
                
        new_resumes = [cleaned_text]
        new_data_features = category.loaded_tfidf.transform(new_resumes)

        predicted_labels = category.loaded_model.predict(new_data_features)
        print(predicted_labels)


        resume_vector = tfidf_vectorizer.transform([resume_text])
        similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100
        ranked_resumes.append({"SIMILARITY" : similarity,
                               "filename": resume_path.filename,
                    "name" : NAME,
                    "threshold_score": "0.0",
                    "degree" : DEGREE,
                    "languages_known" : LANGUAGES_KNOWN,
                    "graduation_year": graduation_year,
                    "special_skills": SKILLS,
                    "catPredict" :predicted_labels})

    
    ranked_resumes.sort(key=lambda x: x["catPredict"], reverse=True)

    sorted_ranked_resumes = sorted(
        ranked_resumes, 
        key=lambda x: (x["catPredict"], x["SIMILARITY"]), 
        reverse=True
    )

    # Printing or returning the sorted list
    for resume in sorted_ranked_resumes:
        print(resume)

    return sorted_ranked_resumes