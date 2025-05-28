from pymongo import MongoClient
import bcrypt
from bson.objectid import ObjectId

class Database:
    def __init__(self):
        try:
            self.client = MongoClient('mongodb://user5hg34pg6:h*45l2)f%26vb%264Gy5Uj@78.153.149.90:37694/')
            self.db = self.client['movie_diary']
            self.watchlist = self.db.watchlist
            self.users = self.db.users
            print("Connected to MongoDB")
        except Exception as e:
            print(f"Error connecting to MongoDB: {str(e)}")
            raise

    def register_user(self, login, password):
        try:
            print(f"Attempting registration: login={login}")
            if self.db.users.find_one({"login": login}):
                print(f"User already exists: login={login}")
                return False
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            result = self.db.users.insert_one({"login": login, "password": hashed_password})
            print(f"User created: login={login}, user_id={result.inserted_id}")
            return result.inserted_id is not None
        except Exception as e:
            print(f"Error during registration: {str(e)}")
            return False

    def add_review(self, user_id, movie_title, review_text, genres, sentiment, cover_data):
        try:
            print(f"Attempting to add review: user_id={user_id}, movie_title={movie_title}")
            review = {
                "user_id": user_id,
                "movie_title": movie_title.strip(),
                "review_text": review_text.strip(),
                "genres": genres,
                "sentiment": sentiment,
                "cover_data": cover_data or ""
            }
            result = self.db.reviews.insert_one(review)
            print(f"Review added: review_id={result.inserted_id}, movie_title={movie_title}")
            return result.inserted_id is not None
        except Exception as e:
            print(f"Error adding review: {str(e)}")
            return False

    def login_user(self, login, password):
        try:
            print(f"Attempting login: login={login}")
            user = self.db.users.find_one({"login": login})
            print(f"User found: {user}")
            if not user:
                print("Login failed: user not found")
                return None
            if not isinstance(user, dict):
                print(f"Login failed: user is not a dictionary, type={type(user)}")
                return None
            if 'password' not in user or 'login' not in user:
                print(f"Login failed: user missing required fields, user={user}")
                return None
            if not isinstance(user['password'], bytes):
                print(f"Login failed: password is not bytes, type={type(user['password'])}")
                return None
            password_match = bcrypt.checkpw(password.encode('utf-8'), user['password'])
            print(f"Password check: success={password_match}")
            if password_match:
                if '_id' not in user:
                    print(f"Login failed: user missing '_id' field, user={user}")
                    return None
                return user
            print("Login failed: incorrect password")
            return None
        except Exception as e:
            print(f"Error during login: {str(e)}")
            return None

    def get_all_reviews(self):
        try:
            reviews = list(self.db.reviews.find())
            return reviews
        except Exception as e:
            print(f"Error getting all reviews: {str(e)}")
            return []

    def get_user_reviews(self, user_id):
        try:
            reviews = list(self.db.reviews.find({"user_id": user_id}))
            return reviews
        except Exception as e:
            print(f"Error getting user reviews: {str(e)}")
            return []

    def get_watchlist_reviews(self, user_id):
        try:
            watchlist = list(self.db.watchlist.find({"user_id": user_id}))
            review_ids = [item['review_id'] for item in watchlist]
            reviews = list(self.db.reviews.find({"_id": {"$in": [ObjectId(rid) if isinstance(rid, str) else rid for rid in review_ids]}}))
            return reviews
        except Exception as e:
            print(f"Error getting watchlist reviews: {str(e)}")
            return []

    def add_to_watchlist(self, user_id, review_id):
        try:
            result = self.db.watchlist.insert_one({"user_id": user_id, "review_id": str(review_id)})
            print(f"Added to watchlist: user_id={user_id}, review_id={review_id}")
            return result.inserted_id is not None
        except Exception as e:
            print(f"Error adding to watchlist: {str(e)}")
            return False

    def is_in_watchlist(self, user_id, review_id):
        try:
            return self.db.watchlist.find_one({"user_id": user_id, "review_id": str(review_id)}) is not None
        except Exception as e:
            print(f"Error checking watchlist: {str(e)}")
            return False

    def delete_review(self, review_id):
        try:
            result = self.db.reviews.delete_one({"_id": review_id if isinstance(review_id, ObjectId) else ObjectId(review_id)})
            print(f"Deleted review: review_id={review_id}, deleted_count={result.deleted_count}")
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting review: {str(e)}")
            return False

    def remove_from_watchlist_all(self, review_id):
        try:
            result = self.db.watchlist.delete_many({"review_id": str(review_id)})
            print(f"Removed from all watchlists: review_id={review_id}, deleted_count={result.deleted_count}")
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error removing from all watchlists: {str(e)}")
            return False

    def remove_from_watchlist(self, user_id, review_id):
        try:
            review_id_str = str(review_id)
            print(f"Attempting to remove from watchlist: user_id={user_id}, review_id={review_id_str}")
            watchlist_entry = self.db.watchlist.find_one({"user_id": user_id, "review_id": review_id_str})
            print(f"Watchlist entry: {watchlist_entry}")
            if not watchlist_entry:
                print(f"No entry found in watchlist for user_id={user_id}, review_id={review_id_str}")
                return False
            result = self.db.watchlist.delete_one({"user_id": user_id, "review_id": review_id_str})
            print(f"Removed from watchlist: user_id={user_id}, review_id={review_id_str}, deleted_count={result.deleted_count}")
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error removing from watchlist: {str(e)}")
            return False

    def update_review(self, review_id, movie_title, review_text, genres, sentiment, cover_data):
        try:
            print(f"Attempting to update review: review_id={review_id}, movie_title={movie_title}")
            update_data = {
                "movie_title": movie_title.strip(),
                "review_text": review_text.strip(),
                "genres": genres,
                "sentiment": sentiment,
                "cover_data": cover_data or ""
            }
            result = self.db.reviews.update_one(
                {"_id": review_id if isinstance(review_id, ObjectId) else ObjectId(review_id)},
                {"$set": update_data}
            )
            print(
                f"Updated review: review_id={review_id}, matched_count={result.matched_count}, modified_count={result.modified_count}")
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating review: {str(e)}")
            return False
