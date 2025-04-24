from datetime import datetime
from app.detect import detection_stats

def get_total_containers():
    return detection_stats.stats["total_detections"]

class ReportGenerator:
    def generate_daily_report(self):
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_detections": get_total_containers(),
            "accuracy_stats": self.calculate_accuracy(),
            "performance_metrics": self.get_performance_metrics(),
            "system_health": self.check_system_health()
        }
        return self.format_report(report)
