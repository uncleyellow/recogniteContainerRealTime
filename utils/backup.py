import os
import shutil
from datetime import datetime

class BackupManager:
    def __init__(self):
        self.backup_dir = "backups"
        self.max_backups = 7
        
    def create_backup(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
        
        # Backup configuration
        shutil.copy2("config.py", backup_path)
        
        # Backup detection stats
        shutil.copy2("storage/stats/detection_stats.json", backup_path)
        
        # Backup recent images
        self.backup_recent_images(backup_path)
        
    def restore_from_backup(self, backup_date):
        backup_path = os.path.join(self.backup_dir, f"backup_{backup_date}")
        if os.path.exists(backup_path):
            shutil.copy2(os.path.join(backup_path, "config.py"), "config.py")
            shutil.copy2(os.path.join(backup_path, "detection_stats.json"), 
                        "storage/stats/detection_stats.json")
            return True
        return False
