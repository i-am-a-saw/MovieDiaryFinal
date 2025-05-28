import sys
import base64
from io import BytesIO
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt5.QtCore import Qt
from src.ui import main_window
from src.database.db import Database
from src.nlp.sentiment import SentimentAnalyzer
import bcrypt

class MovieDiaryApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.db = Database()
        print(f"Database instance: {type(self.db)}, module: {self.db.__module__}")
        self.sentiment_analyzer = SentimentAnalyzer()

        self.login_window = main_window.LoginWindow(db=self.db, parent=None)
        self.login_window.show()

        self.main_window = None
        self.user_id = None

        # Подключение сигналов LoginWindow
        self.login_window.login_button.clicked.connect(self.handle_login)
        self.login_window.register_button.clicked.connect(self.handle_register)
        self.login_window.register_switch_button.clicked.connect(
            lambda: self.login_window.stacked_widget.setCurrentIndex(1))
        self.login_window.login_switch_button.clicked.connect(
            lambda: self.login_window.stacked_widget.setCurrentIndex(0))

    def handle_login(self):
        login = self.login_window.login_input.text()
        password = self.login_window.password_input.text()

        if not login or not password:
            QMessageBox.warning(self.login_window, "Ошибка", "Введите логин и пароль")
            return

        try:
            print(f"Attempting login for user: {login}")
            user = self.db.login_user(login, password)
            print(f"User retrieved: {user}")

            if user:
                if bcrypt.checkpw(password.encode('utf-8'), user['password']):
                    self.user_id = str(user['_id'])
                    print(f"Login successful for user_id: {self.user_id}")

                    self.main_window = main_window.MainWindow(self.user_id, self.db)
                    self.main_window.showMaximized()

                    # Подключение сигналов для добавления/редактирования отзывов
                    self.main_window.cover_button.clicked.connect(self.select_cover)
                    self.main_window.submit_button.clicked.connect(self.handle_submit_review)

                    self.login_window.close()
                else:
                    QMessageBox.warning(self.login_window, "Ошибка", "Неверный пароль")
            else:
                QMessageBox.warning(self.login_window, "Ошибка", "Пользователь не найден")
        except Exception as e:
            print(f"Error during login: {str(e)}")
            QMessageBox.critical(self.login_window, "Ошибка", f"Произошла ошибка при входе: {str(e)}")

    def handle_register(self):
        login = self.login_window.register_login_input.text()
        password = self.login_window.register_password_input.text()

        if not login or not password:
            QMessageBox.warning(self.login_window, "Ошибка", "Введите логин и пароль")
            return

        try:
            if self.db.login_user(login, password):
                QMessageBox.warning(self.login_window, "Ошибка", "Пользователь уже существует")
                return

            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            user_id = self.db.add_user(login, hashed_password)
            if user_id:
                QMessageBox.information(self.login_window, "Успех", "Регистрация успешна")
                self.login_window.stacked_widget.setCurrentIndex(0)
            else:
                QMessageBox.warning(self.login_window, "Ошибка", "Не удалось зарегистрироваться")
        except Exception as e:
            print(f"Error during registration: {str(e)}")
            QMessageBox.critical(self.login_window, "Ошибка", f"Произошла ошибка при регистрации: {str(e)}")

    def select_cover(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Выбрать обложку", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            try:
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    # img.thumbnail((720, 400), Image.LANCZOS)
                    buffer = BytesIO()
                    img.save(buffer, format="PNG", optimize=True, quality=85)
                    self.main_window.cover_path = base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as e:
                QMessageBox.warning(self.main_window, "Ошибка", f"Не удалось загрузить обложку: {str(e)}")

    def handle_submit_review(self):
        title = self.main_window.title_input.text()
        review_text = self.main_window.review_input.toPlainText()
        genres = [self.main_window.genres_list.item(i).text()
                  for i in range(self.main_window.genres_list.count())
                  if self.main_window.genres_list.item(i).isSelected()]

        if not title or not review_text:
            QMessageBox.warning(self.main_window, "Ошибка", "Введите название и текст отзыва")
            return

        try:
            sentiment = self.sentiment_analyzer.analyze_sentiment(review_text)
            review_id = self.db.add_review(
                user_id=self.user_id,
                movie_title=title,
                review_text=review_text,
                sentiment=sentiment,
                cover_data=self.main_window.cover_path or "",
                genres=genres
            )
            if review_id:
                # Обновляем кэши
                self.main_window.reviews_cache = self.db.get_all_reviews()
                self.main_window.user_reviews_cache[self.user_id] = self.db.get_user_reviews(self.user_id)
                self.main_window.title_input.clear()
                self.main_window.review_input.clear()
                self.main_window.genres_list.clearSelection()
                self.main_window.cover_path = None
                self.main_window.show_feed()  # Возвращаемся к ленте
                QMessageBox.information(self.main_window, "Успех", "Отзыв добавлен")
            else:
                QMessageBox.warning(self.main_window, "Ошибка", "Не удалось добавить отзыв")
        except Exception as e:
            print(f"Error submitting review: {str(e)}")
            QMessageBox.warning(self.main_window, "Ошибка", f"Ошибка при добавлении отзыва: {str(e)}")

    def edit_review(self, review):
        try:
            edit_window = main_window.EditReviewWindow(
                review,
                select_cover_callback=self.select_cover_for_edit,
                save_review_callback=self.handle_edit_review,
                parent=self.main_window
            )
            edit_window.exec_()
        except Exception as e:
            print(f"Error opening edit window: {str(e)}")
            QMessageBox.warning(self.main_window, "Ошибка", f"Ошибка при открытии окна редактирования: {str(e)}")

    def select_cover_for_edit(self, edit_window):
        file_path, _ = QFileDialog.getOpenFileName(
            edit_window, "Выбрать обложку", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            try:
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    #img.thumbnail((720, 400), Image.LANCZOS)
                    buffer = BytesIO()
                    img.save(buffer, format="PNG", optimize=True, quality=85)
                    edit_window.cover_path = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    edit_window.update_cover_preview(edit_window.cover_path)
            except Exception as e:
                print(f"Error selecting cover for edit: {str(e)}")
                QMessageBox.warning(edit_window, "Ошибка", f"Не удалось загрузить обложку: {str(e)}")

    def handle_edit_review(self, edit_window, review):
        title = edit_window.title_input.text()
        review_text = edit_window.review_input.toPlainText()
        genres = [edit_window.genres_list.item(i).text()
                  for i in range(edit_window.genres_list.count())
                  if edit_window.genres_list.item(i).isSelected()]

        if not title or not review_text:
            QMessageBox.warning(edit_window, "Ошибка", "Введите название и текст отзыва")
            return

        try:
            sentiment = self.sentiment_analyzer.analyze_sentiment(review_text)
            success = self.db.update_review(
                review_id=review['_id'],
                movie_title=title,
                review_text=review_text,
                sentiment=sentiment,
                cover_data=edit_window.cover_path or review.get('cover_data', ''),
                genres=genres
            )
            if success:
                # Обновляем кэши
                self.main_window.reviews_cache = self.db.get_all_reviews()
                self.main_window.user_reviews_cache[self.user_id] = self.db.get_user_reviews(self.user_id)
                self.main_window.watchlist_cache[self.user_id] = self.db.get_watchlist_reviews(self.user_id)
                QMessageBox.information(edit_window, "Успех", "Отзыв обновлён")
                edit_window.close()  # Добавлено явное закрытие окна редактирования
            else:
                QMessageBox.warning(edit_window, "Ошибка", "Не удалось обновить отзыв")
        except Exception as e:
            print(f"Error editing review: {str(e)}")
            QMessageBox.warning(edit_window, "Ошибка", f"Ошибка при обновлении отзыва: {str(e)}")

    def delete_review(self, review):
        reply = QMessageBox.question(
            self.main_window, "Подтверждение", "Вы уверены, что хотите удалить отзыв?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                success = self.db.delete_review(review['_id'])
                if success:
                    self.db.remove_from_watchlist_all(review['_id'])
                    # Обновляем кэши
                    self.main_window.reviews_cache = self.db.get_all_reviews()
                    self.main_window.user_reviews_cache[self.user_id] = self.db.get_user_reviews(self.user_id)
                    self.main_window.watchlist_cache[self.user_id] = self.db.get_watchlist_reviews(self.user_id)
                    QMessageBox.information(self.main_window, "Успех", "Отзыв удалён")
                    self.main_window.search_reviews()
                else:
                    QMessageBox.warning(self.main_window, "Ошибка", "Не удалось удалить отзыв")
            except Exception as e:
                print(f"Error deleting review: {str(e)}")
                QMessageBox.warning(self.main_window, "Ошибка", f"Ошибка при удалении отзыва: {str(e)}")

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    app = MovieDiaryApp()
    app.run()
