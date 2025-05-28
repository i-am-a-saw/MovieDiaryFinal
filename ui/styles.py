GLASSMORPH_STYLE = """
QWidget {
    background-color: transparent;
    color: #ffffff;
    font-family: 'Montserrat', sans-serif;
    font-size: 16px;
}
QWidget#centralContainer {
    /* max-width удален для полной ширины */
}
QWidget#background {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
        stop:0 #1a1a2a, 
        stop:0.2 #2a1a3a, 
        stop:0.4 #3a1a4a, 
        stop:0.6 #4a1a5a, 
        stop:0.8 #5a1a6a, 
        stop:1 #6a1a7a);
}
QWidget#editWindow, QWidget#loginWindow {
    background: rgba(30, 30, 50, 0.95);
    border-radius: 15px;
}
QLineEdit, QTextEdit {
    background-color: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    padding: 10px;
    color: #ffffff;
    font-size: 16px;
}
QLineEdit#searchInput {
    min-width: 600px;
}
QLineEdit#disabled {
    background-color: rgba(255, 255, 255, 0.05);
    color: #cccccc;
}
QPushButton {
    background-color: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    padding: 10px;
    color: #ffffff;
    font-size: 16px;
}
QPushButton:hover {
    background-color: rgba(255, 255, 255, 0.2);
}
QPushButton#tabButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3f1980, stop:1 #5d0ef5);
    border: none;
    border-radius: 5px;
    padding: 14px;
    text-align: left;
    font-size: 18px;
    color: #f0f8ff;
}
QPushButton#tabButton:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #250261, stop:1 #20009e);
}
QPushButton#tabButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4b2a85, stop:1 #712cf2);
}
QPushButton#filterButton {
    min-width: 130px;
}
QWidget[objectName="buttonsWidget"] {
    min-width: 280px;
}
QScrollArea {
    background-color: transparent;
    border: none;
}
QWidget[objectName="card"] {
    background-color: rgba(200, 200, 200, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 15px;
    margin: 10px;
    padding: 15px;
    width: 470px;
    min-width: 470px;
    max-width: 470px;
    height: 640px;
    min-height: 640px;
    max-height: 640px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
QLabel[objectName="cover"] {
    max-width: 470px;
    min-width: 470px;
    scaledContents: true;
    margin-top: 50px;
}
QLabel#title {
    font-size: 20px;
    font-weight: bold;
}
QLabel#positiveSentiment {
    background-color: rgba(144, 238, 144, 0.75);
    border-radius: 5px;
    padding: 5px;
}
QLabel#negativeSentiment {
    background-color: rgba(255, 99, 71, 0.75);
    border-radius: 5px;
    padding: 5px;
}
QLabel#neutralSentiment {
    background-color: rgba(128,128,128, 0.55);
    border-radius: 5px;
    padding: 5px;
}
QLabel#emptyLabel {
    font-size: 24px;
    color: rgba(255, 255, 255, 0.7);
}
"""
