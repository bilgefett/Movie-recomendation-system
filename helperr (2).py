import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import zipfile

def load_data():
    # Veriyi yükle ve çıkart
    file_name = "C:/Users/asus/Downloads/ml-latest-small.zip"
    with zipfile.ZipFile(file_name, 'r') as z:
        z.extractall()

    # Veri çerçevelerini oku
    ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
    movies_df = pd.read_csv("ml-latest-small/movies.csv")

    # Filmlerin türlerini işleyelim
    movies_df['genre'] = movies_df['genres'].apply(lambda x: x.split('|')[0])

    # Film özellikleri (burada sadece 'genre' kullanılıyor)
    movie_features = movies_df[['genre']]
    
    # Cosine similarity hesapla (film türlerine dayalı)
    cosine_sim = cosine_similarity(pd.get_dummies(movie_features))

    # Film ID'leri ile indeksleri eşleştir
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'])}

    return ratings_df, movies_df, cosine_sim, movie_id_to_index

def cosine_similarity_sample(movies_df, cosine_sim, num_movies=10):
    # Örnek tablo: İlk 'num_movies' filmi kullanıyoruz
    example_movies = movies_df.iloc[:num_movies]['title'].tolist()  # İlk 10 film
    cosine_sim_sample = cosine_sim[:num_movies, :num_movies]

    # Tabloyu DataFrame olarak oluştur
    similarity_df = pd.DataFrame(
        cosine_sim_sample,
        index=example_movies,
        columns=example_movies
    )

    # Cosine similarity değerlerini 0-1 arasında formatlamak için
    similarity_df = similarity_df.round(2)  # İki basamağa yuvarlama
    return similarity_df

# Kullanıcı-film matrisi oluşturma
def get_user_movie_matrix(ratings_df):
    user_movie_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    return user_movie_matrix

# Kullanıcıları KMeans algoritması ile kümelere ayırma
def cluster_users(ratings_df, n_clusters=5):
    user_movie_matrix = get_user_movie_matrix(ratings_df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    user_movie_matrix['cluster'] = kmeans.fit_predict(user_movie_matrix)
    return user_movie_matrix

def cluster_users_and_print(ratings_df):
    user_clusters = cluster_users(ratings_df)
    print(user_clusters.head())
    return user_clusters

if _name_ == "_main_":
    # Veriyi yükle
    ratings_df, movies_df, cosine_sim, _ = load_data()
    
    # 10 filmle cosine similarity tablosu oluştur
    similarity_df = cosine_similarity_sample(movies_df, cosine_sim, num_movies=10)

    # Tabloyu ekrana yazdır
    print("Cosine Similarity Örnek Tablosu (10 Film):")
    print(similarity_df)

    # CSV dosyasına kaydet
    similarity_df.to_csv("cosine_similarity_sample.csv", index=True)

    # Kullanıcıları kümelere ayır ve sonuçları yazdır
    user_clusters = cluster_users_and_print(ratings_df)
