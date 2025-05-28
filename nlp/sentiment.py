from transformers import pipeline
from .use_model import scan

class SentimentAnalyzer:
    def __init__(self):
        pass

    def analyze_sentiment(self, text):
        if not text or not isinstance(text, str):
            return "Отрицательный"  # Значение по умолчанию для пустого текста
        if not self.analyzer:
            return "Отрицательный"  # Значение по умолчанию, если модель не загрузилась

        try:
            # Анализ тональности текста
            result = scan(text)

        except Exception as e:
            print(f"Ошибка анализа тональности: {str(e)}")
            return "Отрицательный"  # Значение по умолчанию при ошибке