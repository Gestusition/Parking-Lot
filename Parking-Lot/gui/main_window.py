from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Parking System")
        self.setGeometry(100, 100, 1280, 720)
        
        # Main layout container
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Header
        self.header_label = QLabel("Parking Management System")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        self.layout.addWidget(self.header_label)
        
        # Video Display Area
        self.video_label = QLabel("Video Feed Loading...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #333;")
        self.video_label.setFixedSize(800, 450) # Fixed size to match video resolution
        self.video_label.installEventFilter(self)
        self.layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Stats Area
        self.stats_layout = QHBoxLayout()
        self.total_spots_label = QLabel("Total Spots: 0")
        self.occupied_spots_label = QLabel("Occupied: 0")
        self.free_spots_label = QLabel("Free: 0")
        
        for label in [self.total_spots_label, self.occupied_spots_label, self.free_spots_label]:
            label.setStyleSheet("font-size: 16px; padding: 5px;")
            self.stats_layout.addWidget(label)
            
        self.layout.addLayout(self.stats_layout)
        
        # Controls
        self.controls_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.settings_btn = QPushButton("Settings")
        
        self.controls_layout.addWidget(self.start_btn)
        self.controls_layout.addWidget(self.stop_btn)
        self.controls_layout.addWidget(self.settings_btn)
        
        # ROI Selection Controls
        self.add_spot_btn = QPushButton("Add Spot")
        self.save_spots_btn = QPushButton("Save Spots")
        self.calibrate_btn = QPushButton("Calibrate Empty")
        self.controls_layout.addWidget(self.add_spot_btn)
        self.delete_spot_btn = QPushButton("Delete Spot")
        self.controls_layout.addWidget(self.delete_spot_btn)
        self.controls_layout.addWidget(self.save_spots_btn)
        self.controls_layout.addWidget(self.calibrate_btn)
        
        self.layout.addLayout(self.controls_layout)

        # Placeholder for video thread
        self.thread = None
        
        # Logic components
        from core.parking_manager import ParkingManager
        self.manager = ParkingManager()
        self.manager.load_from_json()
        
        # Drawing state
        self.drawing_mode = False
        self.delete_mode = False
        self.current_points = []
        
        # Connect signals
        self.start_btn.clicked.connect(self.start_video)
        self.stop_btn.clicked.connect(self.stop_video)
        self.add_spot_btn.clicked.connect(self.toggle_drawing_mode)
        self.delete_spot_btn.clicked.connect(self.toggle_delete_mode)
        self.save_spots_btn.clicked.connect(self.save_spots)
        self.calibrate_btn.clicked.connect(self.calibrate)
        self.settings_btn.clicked.connect(self.open_settings)
        
        # Default settings
        self.video_source = 0
        self.conf = 0.20
        self.overlap = 0.15
        
    def open_settings(self):
        from gui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self)
        if dialog.exec():
            settings = dialog.get_settings()
            self.video_source = settings["source"]
            self.conf = settings["conf"]
            self.overlap = settings["overlap"]
            print(f"Settings updated: Source={self.video_source}, Conf={self.conf}, Overlap={self.overlap}")
            
            # If thread is running, update it
            if self.thread and self.thread.isRunning():
                self.thread.update_settings(self.video_source, self.conf, self.overlap)

    def start_video(self):
        if self.thread and self.thread.isRunning():
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        from gui.video_thread import VideoThread
        # Pass manager to thread
        self.thread = VideoThread(self.manager, self.video_source, self.conf, self.overlap)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.stats_signal.connect(self.update_stats)
        self.thread.start()

    def update_stats(self, total, occupied, free):
        self.total_spots_label.setText(f"Total Spots: {total}")
        self.occupied_spots_label.setText(f"Occupied: {occupied}")
        self.free_spots_label.setText(f"Free: {free}")

    def stop_video(self):
        if self.thread:
            self.thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("Camera Stopped")
            
    def toggle_drawing_mode(self):
        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self.delete_mode = False # Mutually exclusive
            self.add_spot_btn.setText("Cancel Drawing")
            self.delete_spot_btn.setText("Delete Spot")
            self.current_points = []
        else:
            self.add_spot_btn.setText("Add Spot")

    def toggle_delete_mode(self):
        self.delete_mode = not self.delete_mode
        if self.delete_mode:
            self.drawing_mode = False # Mutually exclusive
            self.delete_spot_btn.setText("Done Deleting")
            self.add_spot_btn.setText("Add Spot")
            self.current_points = []
        else:
            self.delete_spot_btn.setText("Delete Spot")

    def save_spots(self):
        self.manager.save_to_json()
        print("Spots saved!")

    def calibrate(self):
        if self.thread and self.thread.isRunning():
            self.thread.calibrate_trigger = True
            print("Calibration requested...")
        else:
            print("Camera not running!")

    def eventFilter(self, source, event):
        from PyQt6.QtCore import QEvent
        if source == self.video_label and event.type() == QEvent.Type.MouseButtonPress:
            x = event.pos().x()
            y = event.pos().y()

            if self.drawing_mode:
                self.current_points.append((x, y))
                
                if len(self.current_points) == 4:
                    self.manager.add_spot(self.current_points)
                    self.current_points = []
                    self.drawing_mode = False
                    self.add_spot_btn.setText("Add Spot")
            
            elif self.delete_mode:
                # Check if click is inside any spot
                for i, spot in enumerate(self.manager.parking_spots):
                    pts = np.array(spot, np.int32)
                    # measureDist=False returns +1 if inside, -1 if outside, 0 if on edge
                    if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                        self.manager.delete_spot(i)
                        print(f"Deleted spot {i}")
                        break

            return True
        return super().eventFilter(source, event)

    def update_image(self, cv_img):
        """Updates the video_label with a new opencv image"""
        # Thread already resizes to 800x450
        
        # Draw existing spots
        from utils.utils import draw_parking_spots
        cv_img = draw_parking_spots(cv_img, self.manager.parking_spots, self.manager.occupancy_status, self.manager.plates)
        
        # Draw points being clicked
        for p in self.current_points:
            cv2.circle(cv_img, p, 5, (0, 255, 255), -1)
            # Draw lines between points
            if len(self.current_points) > 1:
                cv2.polylines(cv_img, [np.array(self.current_points, np.int32)], False, (0, 255, 255), 2)
        
        # Draw instructions if in drawing mode
        if self.drawing_mode:
            cv2.putText(cv_img, f"Click to define point {len(self.current_points)+1}/4", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif self.delete_mode:
            cv2.putText(cv_img, "Click on a spot to delete it", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # p = convert_to_Qt_format.scaled(800, 450, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(convert_to_Qt_format)
