from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(300, 200)
        
        self.layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        
        self.camera_source_input = QLineEdit("0")
        self.form_layout.addRow("Camera Source (ID or Path):", self.camera_source_input)
        
        self.conf_input = QLineEdit("0.20")
        self.form_layout.addRow("Detection Confidence (0.0-1.0):", self.conf_input)
        
        self.overlap_input = QLineEdit("0.15")
        self.form_layout.addRow("Overlap Threshold (0.0-1.0):", self.overlap_input)
        
        self.layout.addLayout(self.form_layout)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.save_btn)

    def get_settings(self):
        source = self.camera_source_input.text()
        # Try to convert to int if it's a number
        if source.isdigit():
            source = int(source)
            
        try:
            conf = float(self.conf_input.text())
        except ValueError:
            conf = 0.20
            
        try:
            overlap = float(self.overlap_input.text())
        except ValueError:
            overlap = 0.15
            
        return {"source": source, "conf": conf, "overlap": overlap}
