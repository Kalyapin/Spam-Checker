import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.naive_bayes import MultinomialNB
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QListWidget, QTextEdit, \
    QLineEdit, QLabel, QMessageBox, QFileDialog

app = QApplication([])
main_window = QWidget()
main_window.setWindowTitle("Spam Checker")
main_window.setFixedSize(800, 600)
write_message = QTextEdit()
write_message.setPlaceholderText("Write a message...")
checker = QLabel()
checker.setText("Spam/Not spam")
timer = QTimer()

main_layout = QHBoxLayout()
layout1 = QVBoxLayout()
layout2 = QVBoxLayout()

main_window.setLayout(main_layout)
main_layout.addLayout(layout1)
main_layout.addLayout(layout2)

layout1.addWidget(write_message)
layout2.addWidget(checker)

is_updated = False


def text_updater():
    global is_updated
    is_updated = True


with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


def update_spam_label():
    global is_updated
    if is_updated:
        text = write_message.toPlainText()
        vectorized_text = vectorizer.transform([text])

        prediction = model.predict(vectorized_text)
        if prediction[0] == 0:
            checker.setText("Not spam")
        else:
            checker.setText("Spam")
        is_updated = False


write_message.textChanged.connect(text_updater)
timer.timeout.connect(update_spam_label)
timer.start(500)

main_window.show()
app.exec()
