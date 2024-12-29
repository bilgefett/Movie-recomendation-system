from flask import Flask, render_template, request, session
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from helperr import load_data

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Session için bir anahtar

# Veriyi yükle
ratings_df, movies_df, cosine_sim, movie_id_to_index = load_data()

# Surprise modelini eğitme işlevi
def train_model(ratings_df):
    reader = Reader(line_format='user item rating timestamp', sep=',')
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    return model, rmse, mae

# Modeli eğit
model, rmse, mae = train_model(ratings_df)

# Cosine similarity'ye dayalı öneri fonksiyonu
def get_cosine_recommendations(movie_id, cosine_sim, movies_df):
    idx = movie_id_to_index[movie_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:11]]  # En benzer 10 film
    recommended_movies = movies_df.iloc[movie_indices]
    return recommended_movies['title'].tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    global ratings_df, model, rmse, mae

    if "users" not in session:
        session["users"] = {}

    if request.method == "POST":
        user_name = request.form.get("username")
        user_movies = request.form.getlist("movies")
        user_ratings = request.form.getlist("ratings")

        if not user_name:
            return render_template("indexx.html", movies=movies_df.to_dict(orient="records"), error="Kullanıcı adınızı girin.", rmse=rmse, mae=mae)

        if user_name not in session["users"]:
            session["users"][user_name] = {"movies": [], "ratings": [], "recommendations": []}

        session["users"][user_name]["movies"].extend(user_movies)
        session["users"][user_name]["ratings"].extend(user_ratings)

        new_data = pd.DataFrame({
            'userId': [user_name] * len(user_movies),
            'movieId': user_movies,
            'rating': user_ratings
        })

        ratings_df = pd.concat([ratings_df, new_data], ignore_index=True)

        # Yeni veriyle modeli yeniden eğit
        model, rmse, mae = train_model(ratings_df)

        all_movies = movies_df['movieId'].tolist()
        # Kullanıcının izlediği filmleri int türüne dönüştürerek kontrol et
        recommendable_movies = [int(movie) for movie in all_movies if int(movie) not in [int(user_movie) for user_movie in session["users"][user_name]["movies"]]]

        recommendations = []
        for movie_id in recommendable_movies:
            pred = model.predict(user_name, movie_id)
            recommendations.append((movie_id, pred.est))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]
        session["users"][user_name]["recommendations"] = [movies_df[movies_df['movieId'] == rec[0]]["title"].values[0] for rec in recommendations]

        return render_template("indexx.html", movies=movies_df.to_dict(orient="records"), users=session["users"], error=None, rmse=rmse, mae=mae)

    return render_template("indexx.html", movies=movies_df.to_dict(orient="records"), users=session.get("users", {}), error=None, rmse=rmse, mae=mae)

if __name__ == "__main__":
    app.run(debug=True) 