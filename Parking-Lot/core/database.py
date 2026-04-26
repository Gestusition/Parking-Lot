import csv
import os
import datetime

class DatabaseManager:
    def __init__(self, db_name="parking_logs.csv"):
        self.filename = db_name
        self.create_table()

    def create_table(self):
        # Check if file exists, if not create with headers
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "timestamp", "spot_id", "plate", "action"])

    def log_event(self, spot_id, plate, action):
        """
        Log an event (Entry/Exit) to CSV.
        action: 'ENTRY' or 'EXIT'
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate a simple ID based on line count (not perfect but sufficient for logs)
        log_id = 1
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                log_id = sum(1 for line in f) # Header is line 1, so id starts at 1
        
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([log_id, timestamp, spot_id, plate, action])

    def close(self):
        pass
