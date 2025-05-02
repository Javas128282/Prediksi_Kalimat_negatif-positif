from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords_indonesia = stopwords.words('indonesian')

# Dataset kecil (termasuk kalimat target)
corpus = [
    "Saya sangat puas dengan layanan layanan ini",
    "Makanan di restoran ini enak sekali",
    "Respon mereka cepat dan tepat",  
    "Saya kecewa dengan layanan ini",
    "Pelayanannya sangat lambat dan tidak ramah",
    "Pelayanan sangat lambat dan mengecewakan"

]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords_indonesia)
X = vectorizer.fit_transform(corpus)

# Lihat hasil TF-IDF dalam bentuk tabel
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

'''if 'respon' in df.columns:
    df['respon'] *= 2.0         untuk menaikan bobot'''

print("Semua Bobot di setiap kalimat:")
print(df)
