import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Data contoh
data = [
    ("Saya sangat senang hari ini", "positif"),
    ("Ini adalah pengalaman yang buruk", "negatif"),
    ("Saya kecewa dengan layanan ini", "negatif"),
    ("Pelayanan mereka sangat baik", "positif"),
    ("Aku tidak suka makanannya", "negatif"),
    ("Mereka luar biasa ramah", "positif"),
    ("Saya sangat senang hari ini", "positif"),
    ("Ini pengalaman yang menyenangkan", "positif"),
    ("Saya merasa bahagia sekali", "positif"),
    ("Mereka sangat membantu saya", "positif"),
    ("Pelayanannya luar biasa", "positif"),
    ("Saya kecewa dengan layanan ini", "negatif"),
    ("Pengalaman ini sangat buruk", "negatif"),
    ("Aku tidak suka makanannya", "negatif"),
    ("Mereka tidak ramah sama sekali", "negatif"),
    ("Layanan pelanggan sangat mengecewakan", "negatif"),
    ("Saya sangat puas dengan layanan ini", "positif"),
    ("Makanan di restoran ini enak sekali", "positif"),
    ("Pengalaman belanja yang menyenangkan", "positif"),
    ("Saya akan kembali lagi ke sini", "positif"),
    ("Pelayanannya sangat ramah dan cepat", "positif"),
    ("Film ini benar-benar menghibur", "positif"),
    ("Saya menyukai kualitas produknya", "positif"),
    ("Mereka sangat membantu dalam proses ini", "positif"),
    ("Hasilnya sesuai dengan harapan saya", "positif"),
    ("Sistemnya mudah digunakan dan efisien", "positif"),
    ("Tempatnya bersih dan nyaman", "positif"),
    ("Harga yang ditawarkan sangat terjangkau", "positif"),
    ("Pengirimannya sangat cepat", "positif"),
    ("Saya merasa senang dengan hasilnya", "positif"),
    ("Mereka memberikan pelayanan terbaik", "positif"),
    ("Kualitas barangnya sangat baik", "positif"),
    ("Saya senang bekerja dengan tim ini", "positif"),
    ("Aplikasinya sangat membantu pekerjaan saya", "positif"),
    ("Sangat direkomendasikan untuk orang lain", "positif"),
    ("Terima kasih atas pengalaman yang luar biasa", "positif"),
    ("Pengalaman yang tidak terlupakan", "positif"),
    ("Saya mendapatkan banyak manfaat dari produk ini", "positif"),
    ("Sangat puas dengan hasil akhirnya", "positif"),
    ("Layanannya benar-benar profesional", "positif"),
    ("Respon mereka cepat dan tepat", "positif"),
    ("Saya sangat kecewa dengan layanan ini", "negatif"),
    ("Makanan di restoran ini sangat buruk", "negatif"),
    ("Belanja di sini benar-benar mengecewakan", "negatif"),
    ("Saya tidak akan kembali lagi", "negatif"),
    ("Pelayanannya sangat lambat dan tidak ramah", "negatif"),
    ("Film ini membosankan dan tidak menarik", "negatif"),
    ("Kualitas produk sangat jelek", "negatif"),
    ("Tidak ada bantuan saat saya butuh", "negatif"),
    ("Hasilnya jauh dari harapan", "negatif"),
    ("Sistemnya membingungkan dan lambat", "negatif"),
    ("Tempatnya kotor dan tidak nyaman", "negatif"),
    ("Harganya mahal dan tidak sebanding", "negatif"),
    ("Pengirimannya sangat lama", "negatif"),
    ("Saya sangat tidak puas dengan hasilnya", "negatif"),
    ("Mereka tidak peduli dengan pelanggan", "negatif"),
    ("Barangnya rusak saat diterima", "negatif"),
    ("Timnya tidak kooperatif sama sekali", "negatif"),
    ("Aplikasinya sering error", "negatif"),
    ("Saya tidak akan merekomendasikan ini ke siapa pun", "negatif"),
    ("Sangat buruk dan tidak profesional", "negatif"),
    ("Ini adalah pengalaman terburuk saya", "negatif"),
    ("Tidak ada manfaat dari produk ini", "negatif"),
    ("Saya merasa tertipu", "negatif"),
    ("Respon mereka sangat lambat dan tidak membantu", "negatif"),
    ("Sangat mengecewakan dari awal sampai akhir", "negatif"),
    ("Aku Suka", "positif"),
    ("Aku suka sekali", "positif"),
    ("Aku sangat suka", "positif"),
    ("Aku tidak suka", "negatif"),
    ("Aku sangat tidak suka", "negatif"),
    ("Aku benci", "negatif"),
    ("Aku sangat benci", "negatif"),
    ("Aku senang", "positif"),
    ("Aku sangat senang", "positif"),
    ("Aku kecewa", "negatif"),
    ("Aku sangat kecewa", "negatif"),
    ("Aku marah", "negatif"),
    ("Aku sangat marah", "negatif"),
    ("Aku bahagia", "positif"),
    ("Aku sangat bahagia", "positif"),
    ("Aku sedih", "negatif"),
    ("Aku sangat sedih", "negatif"),
    ("Aku terharu", "positif"),
    ("Aku sangat terharu", "positif"),
    ("Aku bersemangat", "positif"),
    ("Aku sangat bersemangat", "positif"),
    ("Aku tidak bersemangat", "negatif"),
    ("Aku sangat tidak bersemangat", "negatif"),
    ("Aku puas", "positif"),
    ("Aku sangat puas", "positif"),
    ("Aku kecewa sekali", "negatif"),
    ("Aku marah sekali", "negatif"),
    ("Aku Senang dan Bahagia Hari ini", "positif"),
    ("Saya Senang Hari ini", "positif"),
    ("saya suka hari ini", "positif"),
    ("Aku sangat sedih dan marah hari ini", "negatif"),
    ("Aku tidak Suka hari ini!", "negatif"),
]

# Unduh stopwords NLTK
nltk.download('stopwords')

# Ambil stopwords Bahasa Indonesia
from nltk.corpus import stopwords

stop_words_indonesia = stopwords.words('indonesian')
if "tidak" in stop_words_indonesia:
    stop_words_indonesia.remove("tidak")  # Hapus dari stopwords sehingga tidak diabaikan dari TF-IDF

texts = [t[0] for t in data]#untuk mengambil kalimat
labels = [t[1] for t in data]#untuk mengambil label negatif dan positif

vectorizer = TfidfVectorizer(stop_words=stop_words_indonesia)
#TfidfVectorizer yaitu untuk melakukan tokenisasi secara otomatis
#dan TfidfVectorizer secara otomatis mengubah teks menjadi vektor angka/numerik

X = vectorizer.fit_transform(texts)
#fit_transform untuk mengubah list menjadi matriks 2x4 sehingga dipahami algoritma

feature_names = vectorizer.get_feature_names_out()
''''untuk mengambil kata kata unik contohnya seperti kalimat:
    texts = ["saya suka apel", "saya makan apel"] nanti akan menjadi :
    ['apel', 'makan', 'saya', 'suka']
'''

# Lihat hasil TF-IDF dalam bentuk tabel
df = pd.DataFrame(X.toarray(), columns=feature_names)
#untuk mengubah kata kata unik menjadi array 

if "tidak" in df.columns :
    df["tidak"] *= 2.0 # untuk menaikan bobot

# Train-test split dan latih model
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, random_state=42)
#untuk melatih/testing antara kata kata unik(df) dengan label(positif dan negatif)

model = MultinomialNB()#menggunakan model Multinomial Naive Bayes
'''Model MultinomialNB digunakan untuk menghitung frekuensi (jumlah)
kemunculan kata-kata dalam dokumen dan memanfaatkan informasi itu untuk klasifikasi teks.'''

model.fit(X_train, y_train)

test_text = ["Aku Suka kafe dan restoran yang bagus ini!"]
X_new = vectorizer.transform(test_text)
df_new = pd.DataFrame(X_new.toarray(), columns=feature_names)#mengubah menjadi dataframe dan array
df_new = df_new.reindex(columns=df.columns, fill_value=0)
if "tidak" in df_new.columns:
    df_new["tidak"] *= 2.0  # Sama seperti saat training

# Prediksi
print("Prediksi:", model.predict(df_new))

