from PyQt5.QtWidgets import QMainWindow, QWidget, QListWidget, QListWidgetItem, QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit, QPushButton, QScrollArea, QLabel, QFileDialog, QStackedWidget, QDialog, QGridLayout, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QIcon
from .styles import GLASSMORPH_STYLE
import base64
from io import BytesIO
from PIL import Image
import os
from bson.objectid import ObjectId

GENRES = ["Драма", "Комедия", "Фантастика", "Боевик", "Триллер", "Мелодрама", "Ужасы", "Приключения", "Фэнтези", "Документальный"]

class LoginWindow(QDialog):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Вход в КиноОтзыв")
        self.setStyleSheet(GLASSMORPH_STYLE)
        self.setObjectName("loginWindow")
        self.setFixedSize(400, 300)
        self.db = db
        self.user_id = None

        background_widget = QWidget()
        background_widget.setObjectName("background")
        background_layout = QVBoxLayout(background_widget)
        background_layout.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        self.stacked_widget = QStackedWidget()

        login_widget = QWidget()
        login_layout = QVBoxLayout(login_widget)
        login_layout.setContentsMargins(20, 20, 20, 20)
        login_layout.setSpacing(15)

        self.login_input = QLineEdit()
        self.login_input.setPlaceholderText("Логин")
        login_layout.addWidget(self.login_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Пароль")
        self.password_input.setEchoMode(QLineEdit.Password)
        login_layout.addWidget(self.password_input)

        self.login_button = QPushButton("Войти")
        self.login_button.clicked.connect(self.handle_login)
        login_layout.addWidget(self.login_button)

        self.register_switch_button = QPushButton("Зарегистрироваться")
        self.register_switch_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        login_layout.addWidget(self.register_switch_button)
        login_layout.addStretch()

        register_widget = QWidget()
        register_layout = QVBoxLayout(register_widget)
        register_layout.setContentsMargins(20, 20, 20, 20)
        register_layout.setSpacing(15)

        self.register_login_input = QLineEdit()
        self.register_login_input.setPlaceholderText("Логин")
        register_layout.addWidget(self.register_login_input)

        self.register_password_input = QLineEdit()
        self.register_password_input.setPlaceholderText("Пароль")
        self.register_password_input.setEchoMode(QLineEdit.Password)
        register_layout.addWidget(self.register_password_input)

        self.register_button = QPushButton("Зарегистрироваться")
        self.register_button.clicked.connect(self.handle_register)
        register_layout.addWidget(self.register_button)

        self.login_switch_button = QPushButton("Вернуться ко входу")
        self.login_switch_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        register_layout.addWidget(self.login_switch_button)
        register_layout.addStretch()

        self.stacked_widget.addWidget(login_widget)
        self.stacked_widget.addWidget(register_widget)

        main_layout.addWidget(self.stacked_widget)
        background_layout.addWidget(container)
        self.setLayout(background_layout)

    def handle_login(self):
        try:
            login = self.login_input.text().strip()
            password = self.password_input.text().strip()
            print(f"Login attempt: login='{login}', password='{'*' * len(password)}'")
            if not login or not password:
                print("Empty login or password")
                QMessageBox.warning(self, "Ошибка", "Введите логин и пароль")
                return
            user_id = self.db.login_user(login, password)
            print(f"Login result: user_id={user_id}")
            if user_id:
                self.user_id = user_id
                print(f"Login successful: user_id={user_id}")
                self.accept()
            else:
                print("User not found or incorrect password")
                QMessageBox.warning(self, "Ошибка", "Пользователь не найден или неверный пароль")
        except Exception as e:
            print(f"Error during login: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка входа: {str(e)}")

    def handle_register(self):
        try:
            login = self.register_login_input.text().strip()
            password = self.register_password_input.text().strip()
            print(f"Registration attempt: login='{login}', password='{'*' * len(password)}'")
            if not login or not password:
                print("Empty login or password")
                QMessageBox.warning(self, "Ошибка", "Введите логин и пароль")
                return
            if self.db.register_user(login, password):
                print(f"Registration successful: login={login}")
                QMessageBox.information(self, "Успех", "Пользователь успешно зарегистрирован")
                self.stacked_widget.setCurrentIndex(0)
                self.register_login_input.clear()
                self.register_password_input.clear()
            else:
                print("Registration failed: login already exists or error")
                QMessageBox.warning(self, "Ошибка", "Пользователь уже существует или ошибка регистрации")
        except Exception as e:
            print(f"Error during registration: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка регистрации: {str(e)}")

    def closeEvent(self, event):
        print("Closing LoginWindow")
        event.accept()


class EditReviewWindow(QDialog):
    def __init__(self, review, select_cover_callback=None, save_review_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Редактировать отзыв")
        self.setStyleSheet(GLASSMORPH_STYLE)
        self.setObjectName("editWindow")
        self.setFixedSize(500, 600)  # Увеличенная высота для размещения всех элементов
        self.review = review
        self.cover_path = None
        self.select_cover_callback = select_cover_callback
        self.save_review_callback = save_review_callback

        # Создаём фоновый виджет и компоновку
        background_widget = QWidget()
        background_widget.setObjectName("background")
        background_layout = QVBoxLayout(background_widget)
        background_layout.setContentsMargins(0, 0, 0, 0)

        # Контейнер для содержимого
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)  # Уменьшенное расстояние между элементами

        # Поле ввода названия фильма
        self.title_input = QLineEdit()
        self.title_input.setText(review.get("movie_title", ""))
        self.title_input.setPlaceholderText("Название фильма")
        layout.addWidget(self.title_input)

        # Поле ввода текста отзыва
        self.review_input = QTextEdit()
        self.review_input.setText(review.get("review_text", ""))
        self.review_input.setPlaceholderText("Ваш отзыв")
        self.review_input.setFixedHeight(150)
        layout.addWidget(self.review_input)

        # Список жанров с множественным выбором
        self.genres_list = QListWidget()
        self.genres_list.setFixedHeight(100)
        self.genres_list.setSelectionMode(QListWidget.MultiSelection)
        for genre in GENRES:
            item = QListWidgetItem(genre)
            self.genres_list.addItem(item)
            if genre in review.get("genres", []):
                item.setSelected(True)
        layout.addWidget(self.genres_list)

        # Кнопка выбора обложки
        self.cover_button = QPushButton("Выбрать новую обложку")
        if self.select_cover_callback:
            self.cover_button.clicked.connect(lambda: self.select_cover_callback(self))
        layout.addWidget(self.cover_button)

        # Метка для предпросмотра обложки
        self.cover_label = QLabel()
        self.cover_label.setFixedSize(100, 150)
        self.cover_label.setAlignment(Qt.AlignCenter)
        self.cover_label.setStyleSheet("background-color: rgba(255, 255, 255, 0.1); border-radius: 5px;")
        try:
            self.update_cover_preview(review.get("cover_data", ""))
        except Exception:
            self.cover_label.setText("Нет обложки")
        layout.addWidget(self.cover_label, Qt.AlignCenter)

        # Небольшой разделитель перед кнопкой "Сохранить"
        layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # Кнопка сохранения
        self.save_button = QPushButton("Сохранить")
        if self.save_review_callback:
            self.save_button.clicked.connect(lambda: self.save_review_callback(self, self.review))
        layout.addWidget(self.save_button)

        # Растяжка для выравнивания содержимого сверху
        layout.addStretch()

        background_layout.addWidget(container)
        self.setLayout(background_layout)

    def update_cover_preview(self, cover_data):
        if not cover_data:
            self.cover_label.setText("Нет обложки")
            return
        try:
            img = Image.open(BytesIO(base64.b64decode(cover_data)))
            img.thumbnail((100, 150), Image.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="PNG", optimize=True, quality=85)
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            self.cover_label.setPixmap(pixmap)
        except Exception as e:
            self.cover_label.setText("Ошибка предпросмотра")
            raise Exception(f"Не удалось загрузить предпросмотр: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self, user_id, db):
        super().__init__()
        print("Starting MainWindow initialization")
        self.setWindowTitle("КиноОтзыв")
        self.setStyleSheet(GLASSMORPH_STYLE)
        self.user_id = user_id
        self.db = db
        self.is_watchlist_mode = False
        self.is_my_reviews_mode = False
        self.sentiment_filter = None
        self.reviews_cache = []
        self.user_reviews_cache = {}
        self.watchlist_cache = {}
        self.users_cache = {}
        self.cover_cache = {}  # Кэш обложек
        self.selected_genres = []  # Список выбранных жанров
        self.cache_valid = {"reviews": False, "user_reviews": False, "watchlist": False}  # Флаги актуальности кэшей

        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._debounced_search)

        background_widget = QWidget()
        background_widget.setObjectName("background")

        central_container = QWidget()
        central_container.setObjectName("centralContainer")
        central_layout = QHBoxLayout(central_container)
        central_layout.setContentsMargins(20, 20, 20, 20)
        central_layout.setSpacing(10)

        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)

        base_path = os.path.dirname(os.path.abspath(__file__))
        icons_path = os.path.join(base_path, "icons")
        print(f"Icons path: {icons_path}")

        self.feed_button = QPushButton("Лента")
        self.feed_button.setObjectName("tabButton")
        self.feed_button.setCheckable(True)
        self.feed_button.setChecked(True)
        icon_path = os.path.join(icons_path, "movie.png")
        if os.path.exists(icon_path):
            self.feed_button.setIcon(QIcon(icon_path))
            self.feed_button.setIconSize(QSize(24, 24))

        self.add_review_button = QPushButton("Добавить отзыв")
        self.add_review_button.setObjectName("tabButton")
        self.add_review_button.setCheckable(True)
        icon_path = os.path.join(icons_path, "edit_note.png")
        if os.path.exists(icon_path):
            self.add_review_button.setIcon(QIcon(icon_path))
            self.add_review_button.setIconSize(QSize(24, 24))

        self.my_reviews_button = QPushButton("Мои отзывы")
        self.my_reviews_button.setObjectName("tabButton")
        self.my_reviews_button.setCheckable(True)
        icon_path = os.path.join(icons_path, "rate_review.png")
        if os.path.exists(icon_path):
            self.my_reviews_button.setIcon(QIcon(icon_path))
            self.my_reviews_button.setIconSize(QSize(24, 24))

        self.watchlist_button = QPushButton("Хочу посмотреть")
        self.watchlist_button.setObjectName("tabButton")
        self.watchlist_button.setCheckable(True)
        icon_path = os.path.join(icons_path, "bookmark_border.png")
        if os.path.exists(icon_path):
            self.watchlist_button.setIcon(QIcon(icon_path))
            self.watchlist_button.setIconSize(QSize(24, 24))

        self.genres_label_container = QWidget()
        genres_label_layout = QHBoxLayout(self.genres_label_container)
        genres_label_layout.setContentsMargins(0, 0, 0, 0)
        genres_label_layout.addStretch()
        self.genres_label = QLabel("Жанры:")
        self.genres_label.setStyleSheet("font-size: 28px; color: #ffffff;")
        genres_label_layout.addWidget(self.genres_label)
        genres_label_layout.addStretch()
        self.genres_label_container.hide()

        self.genres_buttons_container = QWidget()
        self.genres_buttons_container.setFixedWidth(200)
        genres_buttons_layout = QVBoxLayout(self.genres_buttons_container)
        genres_buttons_layout.setContentsMargins(0, 0, 0, 0)
        genres_buttons_layout.setSpacing(5)
        self.genre_buttons = {}
        for genre in GENRES:
            button = QPushButton(genre)
            button.setCheckable(True)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #D3D3D3;
                    color: #000000;
                    border: none;
                    border-radius: 5px;
                    padding: 5px;
                    text-align: left;
                }
                QPushButton:checked {
                    background-color: #A9A9A9;
                    color: #000000;
                }
                QPushButton:hover {
                    background-color: #C0C0C0;
                }
            """)
            button.clicked.connect(self._update_genre_selection)
            self.genre_buttons[genre] = button
            genres_buttons_layout.addWidget(button)
        genres_buttons_layout.addStretch()
        self.genres_buttons_container.hide()

        print("Creating logout_button")
        try:
            self.logout_button = QPushButton("Выйти")
            self.logout_button.setObjectName("tabButton")
            icon_path = os.path.join(icons_path, "exit.png")
            if os.path.exists(icon_path):
                self.logout_button.setIcon(QIcon(icon_path))
                self.logout_button.setIconSize(QSize(24, 24))
            self.logout_button.clicked.connect(self.logout)
            print("logout_button created successfully")
        except Exception as e:
            print(f"Error creating logout_button: {str(e)}")
            raise

        sidebar_layout.addWidget(self.feed_button)
        sidebar_layout.addWidget(self.add_review_button)
        sidebar_layout.addWidget(self.my_reviews_button)
        sidebar_layout.addWidget(self.watchlist_button)
        sidebar_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        sidebar_layout.addWidget(self.genres_label_container)
        sidebar_layout.addWidget(self.genres_buttons_container)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.logout_button)

        self.stacked_widget = QStackedWidget()

        feed_widget = QWidget()
        feed_layout = QVBoxLayout(feed_widget)
        feed_layout.setContentsMargins(0, 0, 0, 0)

        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 10)
        self.search_input = QLineEdit()
        self.search_input.setObjectName("searchInput")
        self.search_input.setPlaceholderText("Поиск по названию")
        self.search_input.textChanged.connect(self._start_search_timer)
        self.filter_positive = QPushButton("Положительные")
        self.filter_positive.setObjectName("filterButton")
        self.filter_positive.clicked.connect(lambda: self.apply_filter("Положительный"))
        self.filter_negative = QPushButton("Отрицательные")
        self.filter_negative.setObjectName("filterButton")
        self.filter_negative.clicked.connect(lambda: self.apply_filter("Отрицательный"))
        filter_layout.addWidget(self.search_input, stretch=4)
        filter_layout.addWidget(self.filter_positive, stretch=1)
        filter_layout.addWidget(self.filter_negative, stretch=1)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll_layout.setSpacing(10)
        self.scroll_layout.setColumnStretch(0, 1)
        self.scroll_layout.setColumnStretch(1, 1)
        self.scroll_layout.setColumnStretch(2, 1)
        self.scroll_area.setWidget(self.scroll_content)

        self.empty_label = QLabel("Пока тут пусто...")
        self.empty_label.setObjectName("emptyLabel")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.hide()
        self.scroll_layout.addWidget(self.empty_label, 0, 0, 1, 3)

        self.nothing_label = QLabel("Ничего не нашли :(")
        self.nothing_label.setObjectName("emptyLabel")
        self.nothing_label.setAlignment(Qt.AlignCenter)
        self.nothing_label.hide()
        self.scroll_layout.addWidget(self.nothing_label, 0, 0, 1, 3)

        feed_layout.addWidget(filter_widget)
        feed_layout.addWidget(self.scroll_area)

        add_review_widget = QWidget()
        add_review_layout = QVBoxLayout(add_review_widget)
        add_review_layout.setAlignment(Qt.AlignTop)
        add_review_layout.setContentsMargins(0, 0, 0, 0)
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Название фильма")
        self.review_input = QTextEdit()
        self.review_input.setPlaceholderText("Ваш отзыв")
        self.review_input.setFixedHeight(200)
        self.genres_list = QListWidget()
        self.genres_list.setFixedHeight(100)
        self.genres_list.setSelectionMode(QListWidget.MultiSelection)
        for genre in GENRES:
            self.genres_list.addItem(QListWidgetItem(genre))
        self.cover_button = QPushButton("Выбрать обложку")
        self.submit_button = QPushButton("Добавить отзыв")
        add_review_layout.addWidget(self.title_input)
        add_review_layout.addWidget(self.review_input)
        add_review_layout.addWidget(self.genres_list)
        add_review_layout.addWidget(self.cover_button)
        add_review_layout.addWidget(self.submit_button)
        add_review_layout.addStretch()

        self.stacked_widget.addWidget(feed_widget)
        self.stacked_widget.addWidget(add_review_widget)

        central_layout.addWidget(sidebar)
        central_layout.addWidget(self.stacked_widget, stretch=1)

        background_layout = QHBoxLayout(background_widget)
        background_layout.addWidget(central_container)
        self.setCentralWidget(background_widget)

        self.cover_path = None
        self.card_count = 0

        button_style = """
            QPushButton#tabButton {
                text-align: left;
                padding-left: 10px;
            }
            QPushButton#tabButton::icon {
                margin-right: 8px;
            }
        """
        self.setStyleSheet(GLASSMORPH_STYLE + button_style)

        self.feed_button.clicked.connect(self.show_feed)
        self.add_review_button.clicked.connect(self.show_add_review)
        self.my_reviews_button.clicked.connect(self.load_my_reviews)
        self.watchlist_button.clicked.connect(self.load_watchlist)

        self._initialize_cache()
        self.show_feed()

        print(f"feed_button receivers: {self.feed_button.receivers(self.feed_button.clicked)}")
        print(f"add_review_button receivers: {self.add_review_button.receivers(self.add_review_button.clicked)}")
        print(f"my_reviews_button receivers: {self.my_reviews_button.receivers(self.my_reviews_button.clicked)}")
        print(f"watchlist_button receivers: {self.watchlist_button.receivers(self.watchlist_button.clicked)}")
        print("MainWindow initialization completed")

    def _initialize_cache(self):
        try:
            print("Initializing cache")
            self.reviews_cache = self.db.get_all_reviews()
            self.user_reviews_cache[self.user_id] = self.db.get_user_reviews(self.user_id)
            self.watchlist_cache[self.user_id] = self.db.get_watchlist_reviews(self.user_id)
            self.cache_valid["reviews"] = True
            self.cache_valid["user_reviews"] = True
            self.cache_valid["watchlist"] = True
            print(f"Cache initialized: reviews_cache={len(self.reviews_cache)}, "
                  f"user_reviews_cache={len(self.user_reviews_cache[self.user_id])}, "
                  f"watchlist_cache={len(self.watchlist_cache[self.user_id])}")
        except Exception as e:
            print(f"Error initializing cache: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при инициализации кэша: {str(e)}")

    def _start_search_timer(self):
        try:
            print("Starting search timer")
            self.search_timer.start(300)
        except Exception as e:
            print(f"Error starting search timer: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось запустить таймер поиска: {str(e)}")

    def _update_genre_selection(self):
        try:
            print("Updating genre selection")
            self.selected_genres = [genre for genre, button in self.genre_buttons.items() if button.isChecked()]
            print(f"Selected genres: {self.selected_genres}")
            self._start_search_timer()
        except Exception as e:
            print(f"Error updating genre selection: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при обновлении жанров: {str(e)}")

    def _debounced_search(self):
        try:
            print("Performing debounced search")
            self.search_reviews()
        except Exception as e:
            print(f"Error in debounced search: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка поиска: {str(e)}")

    def invalidate_cache(self, cache_type):
        """Помечает указанный кэш как неактуальный."""
        if cache_type in self.cache_valid:
            self.cache_valid[cache_type] = False
            print(f"Cache invalidated: {cache_type}")

    def logout(self):
        try:
            print("Logging out")
            if not hasattr(self, 'logout_button'):
                print("Warning: logout_button not found")
                QMessageBox.warning(self, "Ошибка", "Кнопка выхода не инициализирована")
                return
            self.clear_scroll_layout()
            self.hide()
            login_window = LoginWindow(self.db)
            if login_window.exec_():
                self.close()
            else:
                self.show()
        except Exception as e:
            print(f"Error during logout: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при выходе: {str(e)}")

    def add_review_card(self, review, edit_callback, delete_callback):
        try:
            review_id = str(review.get("_id"))
            print(f"add_review_card: review_id={review_id}, title={review.get('movie_title')}, user_id={review.get('user_id')}")
            card = QWidget()
            card.setObjectName("card")
            card.setFixedSize(470, 640)
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(8)
            card_layout.setContentsMargins(0, 0, 15, 15)

            cover_label = QLabel()
            cover_label.setObjectName("cover")
            cover_label.setAlignment(Qt.AlignCenter)

            if review_id in self.cover_cache:
                cover_label.setPixmap(self.cover_cache[review_id])
            else:
                cover_data = review.get("cover_data", "")
                if cover_data:
                    try:
                        img = Image.open(BytesIO(base64.b64decode(cover_data)))
                        print("------SIZE:", img.size)
                        img = img.resize((320, 430), Image.LANCZOS)
                        buffer = BytesIO()
                        img.save(buffer, format="PNG", optimize=True, quality=100)
                        pixmap = QPixmap()
                        pixmap.loadFromData(buffer.getvalue())
                        scaled_pixmap = pixmap.scaled(720, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        cover_label.setPixmap(pixmap)
                        self.cover_cache[review_id] = pixmap
                    except Exception as e:
                        print(f"Error loading cover: {str(e)}")
                        cover_label.setText("Ошибка обложки")
                else:
                    cover_label.setText("Нет обложки")
            card_layout.addWidget(cover_label, alignment=Qt.AlignCenter)

            sentiment = "Рекомендую" if review.get("sentiment", "") == "Положительный" else "Не рекомендуют"
            sentiment_label = QLabel(sentiment)
            sentiment_label.setObjectName("positiveSentiment" if sentiment == "Рекомендую" else "negativeSentiment")
            sentiment_label.setWordWrap(True)
            sentiment_label.setFixedWidth(440)
            sentiment_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(sentiment_label, alignment=Qt.AlignCenter)

            genres = review.get("genres", [])
            genres_text = "Жанры: " + ", ".join(genres) if genres else "Жанры: не указаны"
            genres_label = QLabel(genres_text)
            genres_label.setWordWrap(True)
            genres_label.setStyleSheet("font-size: 12px; color: #ffffff;")
            genres_label.setMaximumHeight(50)
            genres_label.setMinimumWidth(440)
            card_layout.addWidget(genres_label, alignment=Qt.AlignCenter)

            review_text = review.get("review_text", "")
            words = review_text.split()
            short_text = " ".join(words[:15]) + ("..." if len(words) > 15 else "")
            review_label = QLabel(short_text)
            review_label.setWordWrap(True)
            review_label.setMaximumHeight(60)
            review_label.setMinimumWidth(440)
            card_layout.addWidget(review_label, alignment=Qt.AlignCenter)

            full_review_container = QWidget()
            full_review_container.setMinimumWidth(440)
            full_review_layout = QVBoxLayout(full_review_container)
            full_review_layout.setContentsMargins(0, 0, 0, 0)
            full_review_text = QTextEdit()
            full_review_text.setText(review_text)
            full_review_text.setReadOnly(True)
            full_review_text.setStyleSheet("background-color: rgba(255, 255, 255, 0.1); border: none; color: #ffffff; font-size: 12px;")
            full_review_layout.addWidget(full_review_text)
            full_review_container.hide()

            more_button = QPushButton("Больше...")
            more_button.setObjectName("filterButton")
            less_button = QPushButton("Меньше")
            less_button.setObjectName("filterButton")
            less_button.hide()

            more_button_container = QWidget()
            more_button_container.setMinimumWidth(440)
            more_button_layout = QHBoxLayout(more_button_container)
            more_button_layout.setContentsMargins(0, 0, 0, 0)
            more_button_layout.addStretch()
            more_button_layout.addWidget(more_button)
            more_button_layout.addWidget(less_button)
            more_button_layout.addStretch()
            if len(words) <= 15:
                more_button_container.hide()

            card_layout.addWidget(more_button_container, alignment=Qt.AlignCenter)
            card_layout.addWidget(full_review_container, alignment=Qt.AlignCenter)

            def show_full_review():
                review_label.hide()
                more_button.hide()
                full_review_container.show()
                less_button.show()

            def hide_full_review():
                full_review_container.hide()
                less_button.hide()
                review_label.show()
                more_button.show()

            more_button.clicked.connect(show_full_review)
            less_button.clicked.connect(hide_full_review)

            if self.is_my_reviews_mode or self.is_watchlist_mode:
                buttons_widget = QWidget()
                buttons_widget.setObjectName("buttonsWidget")
                buttons_widget.setMinimumWidth(440)
                buttons_layout = QHBoxLayout(buttons_widget)
                buttons_layout.setContentsMargins(0, 0, 0, 0)
                buttons_layout.setSpacing(10)

                if self.is_watchlist_mode:
                    remove_button = QPushButton("Удалить из списка")
                    remove_button.setObjectName("filterButton")
                    remove_button.setFixedWidth(440)
                    remove_button.clicked.connect(lambda: self.remove_from_watchlist(review))
                    buttons_layout.addWidget(remove_button)
                elif self.is_my_reviews_mode:
                    if str(review.get("user_id")) == self.user_id:
                        edit_button = QPushButton("Изменить")
                        edit_button.setObjectName("filterButton")
                        edit_button.setMinimumWidth(215)
                        edit_button.clicked.connect(lambda: edit_callback(review))
                        buttons_layout.addWidget(edit_button)

                    delete_button = QPushButton("Удалить")
                    delete_button.setObjectName("filterButton")
                    delete_button.setMinimumWidth(215)
                    delete_button.clicked.connect(lambda: delete_callback(review))
                    buttons_layout.addWidget(delete_button)

                card_layout.addWidget(buttons_widget, alignment=Qt.AlignCenter)

            if not (self.is_my_reviews_mode or self.is_watchlist_mode) and str(review.get("user_id")) != self.user_id:
                watch_button_container = QWidget()
                watch_button_container.setMinimumWidth(440)
                watch_button_layout = QHBoxLayout(watch_button_container)
                watch_button_layout.setContentsMargins(0, 0, 0, 0)
                watch_button_layout.addStretch()

                watch_button = QPushButton("Хочу посмотреть!")
                watch_button.setObjectName("filterButton")
                watch_button.setFixedWidth(430)
                if not self.db.is_in_watchlist(self.user_id, review["_id"]):
                    watch_button_layout.addWidget(watch_button)
                watch_button_layout.addStretch()
                card_layout.addWidget(watch_button_container, alignment=Qt.AlignCenter)

                def toggle_watchlist():
                    try:
                        if self.db.add_to_watchlist(self.user_id, review["_id"]):
                            watch_button.hide()
                            print(f"Added to watchlist: review_id={review['_id']}, user_id={self.user_id}")
                            self.watchlist_cache[self.user_id] = self.db.get_watchlist_reviews(self.user_id)
                            self.cache_valid["watchlist"] = True
                        else:
                            QMessageBox.warning(self, "Ошибка", "Не удалось добавить в список Хочу посмотреть")
                    except Exception as e:
                        print(f"Error toggling watchlist: {str(e)}")
                        QMessageBox.warning(self, "Ошибка", f"Не удалось обновить список: {str(e)}")

                watch_button.clicked.connect(toggle_watchlist)

            if not self.is_my_reviews_mode:
                user_id = str(review.get("user_id"))
                user = self.users_cache.get(user_id, {})
                if not user and user_id:
                    try:
                        user = self.db.users.find_one({"_id": ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id})
                        self.users_cache[user_id] = user or {}
                    except AttributeError as e:
                        print(f"Error accessing users collection: {str(e)}")
                        self.users_cache[user_id] = {}
                author_text = f"Автор: {user.get('login')}"
                author_label = QLabel(author_text)
                author_label.setStyleSheet("font-size: 16px; color: #ffffff;")
                author_container = QWidget()
                author_container.setMinimumWidth(440)
                author_layout = QHBoxLayout(author_container)
                author_layout.setContentsMargins(0, 0, 10, 10)
                author_layout.addStretch()
                author_layout.addWidget(author_label)
                card_layout.addWidget(author_container, alignment=Qt.AlignRight)

            card_layout.addStretch()

            row = self.card_count // 3
            col = self.card_count % 3
            self.scroll_layout.addWidget(card, row, col)
            self.card_count += 1
            self.empty_label.hide()
            self.nothing_label.hide()
            print(f"add_review_card: added card at row={row}, col={col}, card_count={self.card_count}")

        except Exception as e:
            print(f"Error creating review card: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при создании карточки: {str(e)}")

    def clear_scroll_layout(self):
        try:
            print("Clearing scroll layout")
            widgets_to_remove = []
            while self.scroll_layout.count():
                item = self.scroll_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widgets_to_remove.append(widget)

            for widget in widgets_to_remove:
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()

            self.empty_label = QLabel("Пока тут пусто...")
            self.empty_label.setObjectName("emptyLabel")
            self.empty_label.setAlignment(Qt.AlignCenter)
            self.empty_label.hide()
            self.scroll_layout.addWidget(self.empty_label, 0, 0, 1, 3)

            self.nothing_label = QLabel("Ничего не нашли :(")
            self.nothing_label.setObjectName("emptyLabel")
            self.nothing_label.setAlignment(Qt.AlignCenter)
            self.nothing_label.hide()
            self.scroll_layout.addWidget(self.nothing_label, 0, 0, 1, 3)

            self.card_count = 0
            self.empty_label.show()
            self.nothing_label.hide()
            print("clear_scroll_layout: cleared layout")
            QTimer.singleShot(50, self.scroll_area.update)
        except Exception as e:
            print(f"Error clearing scroll layout: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при очистке ленты: {str(e)}")

    def show_empty_label(self):
        try:
            print("Showing empty label")
            if self.card_count == 0:
                self.empty_label.show()
                self.nothing_label.hide()
            else:
                self.empty_label.hide()
                self.nothing_label.hide()
            print(f"show_empty_label: card_count={self.card_count}, empty_label_visible={self.empty_label.isVisible()}")
        except Exception as e:
            print(f"Error showing empty label: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при отображении метки: {str(e)}")

    def search_reviews(self):
        try:
            print("Searching reviews")
            print(f"Cache state: reviews_cache={len(self.reviews_cache)} reviews, "
                  f"user_reviews_cache={len(self.user_reviews_cache.get(self.user_id, []))}, "
                  f"watchlist_cache={len(self.watchlist_cache.get(self.user_id, []))}")
            search_text = self.search_input.text().strip().lower()
            selected_genres = self.selected_genres

            if self.is_watchlist_mode:
                reviews = self.watchlist_cache.get(self.user_id, [])
            elif self.is_my_reviews_mode:
                reviews = self.user_reviews_cache.get(self.user_id, [])
            else:
                reviews = self.reviews_cache

            print(f"search_reviews: mode={'watchlist' if self.is_watchlist_mode else 'my_reviews' if self.is_my_reviews_mode else 'feed'}, "
                  f"reviews_count={len(reviews)}")

            filtered_reviews = []
            for review in reviews:
                if not isinstance(review, dict) or "_id" not in review or "movie_title" not in review:
                    print(f"Skipping invalid review: {review}")
                    continue
                if (not search_text or search_text in review.get("movie_title", "").lower()) and \
                   (not selected_genres or any(genre in review.get("genres", []) for genre in selected_genres)) and \
                   (self.sentiment_filter is None or review.get("sentiment", "") == self.sentiment_filter):
                    filtered_reviews.append(review)

            print(f"search_reviews: search_text='{search_text}', genres={selected_genres}, "
                  f"sentiment_filter={self.sentiment_filter}, filtered={len(filtered_reviews)} "
                  f"(titles={[r.get('movie_title', 'Unknown') for r in filtered_reviews[:5]]})")

            self.clear_scroll_layout()
            if not filtered_reviews:
                self.nothing_label.show()
                self.empty_label.hide()
            else:
                # Ленивая загрузка: рендерим первые 9 карточек сразу, остальные порциями
                self.pending_reviews = filtered_reviews[:30]
                self.rendered_reviews = 0
                self.batch_size = 9
                self.render_review_batch()

        except Exception as e:
            print(f"Error during search: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка поиска: {str(e)}")

    def render_review_batch(self):
        """Рендерит следующую порцию карточек отзывов."""
        try:
            start = self.rendered_reviews
            end = min(start + self.batch_size, len(self.pending_reviews))
            print(f"Rendering review batch: start={start}, end={end}, total={len(self.pending_reviews)}")

            for review in self.pending_reviews[start:end]:
                self.add_review_card(
                    review,
                    edit_callback=self.edit_review,
                    delete_callback=self.delete_review
                )
            self.rendered_reviews = end

            if self.rendered_reviews < len(self.pending_reviews):
                # Планируем следующую порцию через QTimer
                QTimer.singleShot(50, self.render_review_batch)
            else:
                self.nothing_label.hide()
                self.empty_label.hide()
                print("Finished rendering all reviews")
        except Exception as e:
            print(f"Error rendering review batch: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка рендеринга карточек: {str(e)}")

    def apply_filter(self, sentiment):
        try:
            print("Applying filter: " + sentiment)
            self.sentiment_filter = sentiment
            self.search_reviews()
        except Exception as e:
            print(f"Error applying filter: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при применении фильтра: {str(e)}")

    def show_feed(self):
        try:
            print("Entering show_feed")
            print(f"stacked_widget index: {self.stacked_widget.currentIndex()}, count: {self.stacked_widget.count()}")
            self.feed_button.blockSignals(True)
            self.add_review_button.blockSignals(True)
            self.my_reviews_button.blockSignals(True)
            self.watchlist_button.blockSignals(True)
            self.feed_button.setChecked(True)
            self.add_review_button.setChecked(False)
            self.my_reviews_button.setChecked(False)
            self.watchlist_button.setChecked(False)
            self.is_my_reviews_mode = False
            self.is_watchlist_mode = False
            self.sentiment_filter = None
            self.search_input.clear()
            self.selected_genres = []
            for button in self.genre_buttons.values():
                button.setChecked(False)
            self.genres_label_container.show()
            self.genres_buttons_container.show()
            self.search_input.show()
            self.filter_positive.show()
            self.filter_negative.show()
            # Обновляем кэш только если он неактуален или пуст
            if not self.cache_valid["reviews"] or not self.reviews_cache:
                print("Updating reviews_cache")
                self.reviews_cache = self.db.get_all_reviews()
                self.cache_valid["reviews"] = True
                print(f"Updated reviews_cache: count={len(self.reviews_cache)}")
            QTimer.singleShot(0, lambda: [
                self.stacked_widget.setCurrentIndex(0),
                self.clear_scroll_layout(),
                self.search_reviews(),
                print(f"After QTimer: current stacked_widget index: {self.stacked_widget.currentIndex()}")
            ])
            print("Completed show_feed")
            self.feed_button.blockSignals(False)
            self.add_review_button.blockSignals(False)
            self.my_reviews_button.blockSignals(False)
            self.watchlist_button.blockSignals(False)
        except Exception as e:
            print(f"Error showing feed: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при отображении ленты: {str(e)}")

    def show_add_review(self):
        try:
            print("Entering show_add_review")
            print(f"stacked_widget index: {self.stacked_widget.currentIndex()}, count: {self.stacked_widget.count()}")
            self.feed_button.blockSignals(True)
            self.add_review_button.blockSignals(True)
            self.my_reviews_button.blockSignals(True)
            self.watchlist_button.blockSignals(True)
            self.feed_button.setChecked(False)
            self.add_review_button.setChecked(True)
            self.my_reviews_button.setChecked(False)
            self.watchlist_button.setChecked(False)
            self.is_my_reviews_mode = False
            self.is_watchlist_mode = False
            self.sentiment_filter = None
            self.search_input.clear()
            self.selected_genres = []
            for button in self.genre_buttons.values():
                button.setChecked(False)
            self.title_input.clear()
            self.review_input.clear()
            self.genres_list.clearSelection()
            self.cover_path = None
            self.genres_label_container.hide()
            self.genres_buttons_container.hide()
            self.search_input.hide()
            self.filter_positive.hide()
            self.filter_negative.hide()
            QTimer.singleShot(0, lambda: [
                self.stacked_widget.setCurrentIndex(1),
                print(f"After QTimer: current stacked_widget index: {self.stacked_widget.currentIndex()}")
            ])
            print("Completed show_add_review")
            self.feed_button.blockSignals(False)
            self.add_review_button.blockSignals(False)
            self.my_reviews_button.blockSignals(False)
            self.watchlist_button.blockSignals(False)
        except Exception as e:
            print(f"Error showing add review: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при отображении добавления отзыва: {str(e)}")

    def load_my_reviews(self):
        try:
            print("Entering load_my_reviews")
            print(f"stacked_widget index: {self.stacked_widget.currentIndex()}, count: {self.stacked_widget.count()}")
            self.feed_button.blockSignals(True)
            self.add_review_button.blockSignals(True)
            self.my_reviews_button.blockSignals(True)
            self.watchlist_button.blockSignals(True)
            self.feed_button.setChecked(False)
            self.add_review_button.setChecked(False)
            self.my_reviews_button.setChecked(True)
            self.watchlist_button.setChecked(False)
            self.is_my_reviews_mode = True
            self.is_watchlist_mode = False
            self.sentiment_filter = None
            self.search_input.clear()
            self.selected_genres = []
            for button in self.genre_buttons.values():
                button.setChecked(False)
            self.genres_label_container.hide()
            self.genres_buttons_container.hide()
            self.search_input.show()
            self.filter_positive.show()
            self.filter_negative.show()
            # Обновляем кэш только если он неактуален или отсутствует
            if not self.cache_valid["user_reviews"] or self.user_id not in self.user_reviews_cache:
                print("Updating user_reviews_cache")
                self.user_reviews_cache[self.user_id] = self.db.get_user_reviews(self.user_id)
                self.cache_valid["user_reviews"] = True
                print(f"Updated user_reviews_cache: count={len(self.user_reviews_cache[self.user_id])}")
            QTimer.singleShot(0, lambda: [
                self.stacked_widget.setCurrentIndex(0),
                self.clear_scroll_layout(),
                self.search_reviews(),
                print(f"After QTimer: current stacked_widget index: {self.stacked_widget.currentIndex()}")
            ])
            print("Completed load_my_reviews")
            self.feed_button.blockSignals(False)
            self.add_review_button.blockSignals(False)
            self.my_reviews_button.blockSignals(False)
            self.watchlist_button.blockSignals(False)
        except Exception as e:
            print(f"Error loading my reviews: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при загрузке моих отзывов: {str(e)}")

    def load_watchlist(self):
        try:
            print("Entering load_watchlist")
            print(f"stacked_widget index: {self.stacked_widget.currentIndex()}, count: {self.stacked_widget.count()}")
            self.feed_button.blockSignals(True)
            self.add_review_button.blockSignals(True)
            self.my_reviews_button.blockSignals(True)
            self.watchlist_button.blockSignals(True)
            self.feed_button.setChecked(False)
            self.add_review_button.setChecked(False)
            self.my_reviews_button.setChecked(False)
            self.watchlist_button.setChecked(True)
            self.is_my_reviews_mode = False
            self.is_watchlist_mode = True
            self.sentiment_filter = None
            self.search_input.clear()
            self.selected_genres = []
            for button in self.genre_buttons.values():
                button.setChecked(False)
            self.genres_label_container.hide()
            self.genres_buttons_container.hide()
            self.search_input.show()
            self.filter_positive.show()
            self.filter_negative.show()
            if not self.cache_valid["watchlist"] or self.user_id not in self.watchlist_cache:
                print("Updating watchlist_cache")
                self.watchlist_cache[self.user_id] = self.db.get_watchlist_reviews(self.user_id)
                self.cache_valid["watchlist"] = True
                print(f"Updated watchlist_cache: count={len(self.watchlist_cache[self.user_id])}")
            QTimer.singleShot(0, lambda: [
                self.stacked_widget.setCurrentIndex(0),
                self.clear_scroll_layout(),
                self.search_reviews(),
                print(f"After QTimer: current stacked_widget index: {self.stacked_widget.currentIndex()}")
            ])
            print("Completed load_watchlist")
            self.feed_button.blockSignals(False)
            self.add_review_button.blockSignals(False)
            self.my_reviews_button.blockSignals(False)
            self.watchlist_button.blockSignals(False)
        except Exception as e:
            print(f"Error loading watchlist: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при загрузке списка желаемого: {str(e)}")

    def remove_from_watchlist(self, review):
        try:
            print(f"Removing from watchlist: user_id={self.user_id}, review_id={review.get('_id')}")
            if not review.get('_id'):
                print("Error: Review ID is missing")
                QMessageBox.warning(self, "Ошибка", "Отзыв не содержит идентификатор")
                return
            review_id = str(review['_id'])
            watchlist_entry = self.db.watchlist.find_one({"user_id": self.user_id, "review_id": review_id})
            print(f"Watchlist entry for user_id={self.user_id}, review_id={review_id}: {watchlist_entry}")
            reply = QMessageBox.question(
                self, "Подтверждение", "Вы уверены, что хотите убрать фильм из списка Хочу посмотреть?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            print(f"QMessageBox reply: {reply}")
            if reply == QMessageBox.Yes:
                print("Calling db.remove_from_watchlist")
                success = self.db.remove_from_watchlist(self.user_id, review_id)
                print(f"Remove from watchlist result: success={success}")
                if success:
                    print("Updating watchlist_cache")
                    self.watchlist_cache[self.user_id] = self.db.get_watchlist_reviews(self.user_id)
                    self.cache_valid["watchlist"] = True
                    print(f"Updated watchlist_cache: count={len(self.watchlist_cache[self.user_id])}")
                    QMessageBox.information(self, "Успех", "Фильм убран из списка Хочу посмотреть")
                    print("Calling search_reviews")
                    self.search_reviews()
                else:
                    print("Failed to remove from watchlist")
                    watchlist_contents = list(self.db.watchlist.find({"user_id": self.user_id}))
                    print(f"Watchlist contents for user_id={self.user_id}: {watchlist_contents}")
                    QMessageBox.warning(self, "Ошибка", "Не удалось убрать фильм из списка. Проверьте логи для деталей.")
        except Exception as e:
            print(f"Error removing from watchlist: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при удалении из списка: {str(e)}")

    def edit_review(self, review):
        try:
            print("Opening edit review window")
            edit_window = EditReviewWindow(review, parent=self)
            edit_window.exec_()
            self.user_reviews_cache[self.user_id] = self.db.get_user_reviews(self.user_id)
            self.reviews_cache = self.db.get_all_reviews()
            self.cache_valid["user_reviews"] = True
            self.cache_valid["reviews"] = True
            self.search_reviews()
        except Exception as e:
            print(f"Error opening edit review window: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при открытии редактирования: {str(e)}")

    def delete_review(self, review):
        try:
            print("Deleting review")
            reply = QMessageBox.question(
                self, "Подтверждение", "Вы уверены, что хотите удалить отзыв?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                success = self.db.delete_review(review['_id'])
                if success:
                    self.db.remove_from_watchlist_all(review['_id'])
                    self.user_reviews_cache[self.user_id] = self.db.get_user_reviews(self.user_id)
                    self.watchlist_cache[self.user_id] = self.db.get_watchlist_reviews(self.user_id)
                    self.reviews_cache = self.db.get_all_reviews()
                    self.cache_valid["user_reviews"] = True
                    self.cache_valid["watchlist"] = True
                    self.cache_valid["reviews"] = True
                    QMessageBox.information(self, "Успех", "Отзыв удалён")
                    self.search_reviews()
                else:
                    QMessageBox.warning(self, "Ошибка", "Не удалось удалить отзыв")
        except Exception as e:
            print(f"Error deleting review: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при удалении: {str(e)}")

    def closeEvent(self, event):
        try:
            print("Closing MainWindow")
            event.accept()
        except Exception as e:
            print(f"Error closing MainWindow: {str(e)}")