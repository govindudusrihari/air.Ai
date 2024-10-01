import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os 
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, 
                             QFileDialog, 
                             QProgressBar, 
                             QMessageBox,
                             QComboBox, 
                             QGridLayout,
                             )
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor
from PIL import Image
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal

# --- Core Model and Preprocessing ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # ... (Add other convolutional layers, pooling, fully connected layers) 

    def forward(self, x):
        # ... (Perform computations based on your CNN architecture)
        return x

def load_and_preprocess_image(image_path):
    """Loads an image from a path and prepares it for your model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB') 
    return transform(image).unsqueeze(0) # Add a batch dimension (1, C, H, W)

def predict_image_class(image_tensor, model):
    """Feeds image into your model and returns predicted class."""
    with torch.no_grad():
        output = model(image_tensor)
    # ( ... (Determine your predicted class label and score using your model's output) ... )
    # For this example, let's assume output has shape [1, num_classes] 
    # Find the class with the highest probability:
    predicted_class_idx = torch.argmax(output)
    confidence_score = torch.nn.functional.softmax(output, dim=1)[0, predicted_class_idx].item()
    #  ( ... (Replace this part with your classification logic, returning predicted class and confidence score) ... )
    return predicted_class_idx.item(), confidence_score 

# --- Worker Thread ---

class Worker(QThread):  
    """Worker thread for background processing."""

    finished = pyqtSignal()
    progress = pyqtSignal(int) 
    result = pyqtSignal(int, float)  
    error = pyqtSignal(str) 

    def __init__(self, image_path, model):
        super().__init__()
        self.image_path = image_path
        self.model = model

    def run(self):
        try:
            self.progress.emit(20) 
            image_tensor = load_and_preprocess_image(self.image_path)
            self.progress.emit(50)
            predicted_class_idx, confidence_score = predict_image_class(image_tensor, self.model)
            self.progress.emit(80) 
            self.result.emit(predicted_class_idx, confidence_score) 
        except Exception as e:
            self.error.emit(str(e)) 
        finally:
            self.finished.emit() 

# --- UI (PyQt) --- 

class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier App")
        self.model = SimpleCNN() 
        self.model.load_state_dict(torch.load("path/to/your/trained/model.pth")) 
        self.model.eval()

        # --- UI Setup --- 

        self.image_label = QLabel()
        self.result_label = QLabel()
        self.additional_info_label = QLabel("")
        self.additional_info_label.setWordWrap(True)  
        self.additional_info_label.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar() 
        self.progress_bar.setRange(0, 100)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image) 

        self.model_selection = QComboBox() 
        self.model_selection.addItems(["SimpleCNN"]) #  For now, just the 'SimpleCNN'
        self.model_selection.currentIndexChanged.connect(self.change_model)  

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.progress_bar) 
        layout.addWidget(self.result_label) 
        layout.addWidget(self.model_selection) 
        layout.addWidget(self.additional_info_label)
        self.setLayout(layout) 

    # --- Functions ---

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            try:
                self.image_path = file_path # Set for the Worker thread
                self.display_image(file_path) 
            except (FileNotFoundError, OSError) as e:
                QMessageBox.warning(self, "Error", f"Unable to load the image: {e}")

    def display_image(self, file_path):
        image = QPixmap(file_path).scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(image)

    def classify_and_display(self): 
        if not hasattr(self, 'worker') or self.worker is None: 
            self.worker = Worker(self.image_path, self.model) 
            self.worker.finished.connect(self.on_thread_finished)
            self.worker.progress.connect(self.update_progress) 
            self.worker.result.connect(self.update_result)
            self.worker.error.connect(self.handle_error) 
            self.worker.start()

    def on_thread_finished(self):
        self.worker = None # Release worker after its task

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_result(self, predicted_class_idx, confidence_score):
        self.result_label.setText(f"Predicted Class: {predicted_class_idx}")
        self.additional_info_label.setText(f"Confidence Score: {confidence_score:.2f}")
        self.confidence_score = confidence_score
        self.update() # Refresh to repaint

    def handle_error(self, error_message):
        QMessageBox.warning(self, "Error", f"Classification failed: {error_message}")

    def change_model(self):
        #  You may want to add more models later 
        # (For now, SimpleCNN is the only option)
        model_name = self.model_selection.currentText()
        if model_name == "SimpleCNN":
            self.model = SimpleCNN()
            self.model.load_state_dict(torch.load("path/to/your/trained/model.pth"))
            self.model.eval()

    def paintEvent(self, event):
        """Handles drawing the confidence indicator (repainting the UI)."""
        super().paintEvent(event)
        painter = QPainter(self)
        confidence_bar_height = 20 
        confidence_bar_width = self.width() - 50  
        confidence_bar_x = 25 
        confidence_bar_y = self.result_label.y() + self.result_label.height() + 10

        painter.fillRect(confidence_bar_x, confidence_bar_y, int(confidence_bar_width * self.confidence_score), confidence_bar_height, QColor(0, 255, 0))  
        painter.fillRect(int(confidence_bar_x + confidence_bar_width * self.confidence_score), confidence_bar_y, int(confidence_bar_width * (1 - self.confidence_score)), confidence_bar_height, QColor(255, 0, 0)) # Red for less confident

if __name__ == "__main__":
    app = QApplication([])
    window = ImageClassifierApp()
    window.show()
    app.exec_() 
