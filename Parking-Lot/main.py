import sys
import os

# --- FORCE CPU MODE & OPTIMIZATIONS ---
# CRITICAL: Set this BEFORE importing torch/ultralytics to avoid any GPU checks
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Silence YOLO and Ultralytics
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_QUIET"] = "true"
os.environ["ULTRALYTICS_NO_AUTOINSTALL"] = "True" # Prevent auto-updates/installs

# Prevent "RAM usage" / Library errors (common in PyTorch/Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Optimize CPU usage
os.environ["OMP_NUM_THREADS"] = "auto"

print(">> SYSTEM: Forcing CPU mode for maximum stability and speed.")

from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
