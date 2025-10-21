"""
Alert System Module
Manages disaster alerts with visual, audio, and logging capabilities
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import csv


class DisasterAlertSystem:
    """Manages multi-disaster alert generation and logging"""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        self.alert_log_path = self.reports_dir / "alert_log.csv"
        self.active_alerts = {}
        
        # Load configuration
        with open("configs/thresholds.json", 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Initialize alert log
        self._initialize_log()
        
        print("✅ Alert System initialized")
    
    def _initialize_log(self):
        """Initialize CSV log file"""
        if not self.alert_log_path.exists():
            with open(self.alert_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'disaster_type', 'risk_level', 
                    'risk_score', 'alert_message', 'acknowledged'
                ])
    
    def get_alert_emoji(self, disaster_type: str, risk_level: str) -> str:
        """Get emoji for alert visualization"""
        emoji_map = {
            'flood': {'SAFE': '🟢', 'LOW': '🟡', 'MEDIUM': '🟠', 'HIGH': '🔴', 'CRITICAL': '🚨'},
            'earthquake': {'SAFE': '🟢', 'LOW': '🟡', 'MEDIUM': '🟠', 'HIGH': '🔴', 'CRITICAL': '🚨'},
            'cyclone': {'SAFE': '🟢', 'LOW': '🟡', 'MEDIUM': '🟠', 'HIGH': '🔴', 'CRITICAL': '🚨'}
        }
        return emoji_map.get(disaster_type, {}).get(risk_level, '⚪')
    
    def get_alert_message(self, disaster_type: str, risk_level: str, risk_score: float) -> str:
        """Generate human-readable alert message"""
        messages = {
            'flood': {
                'SAFE': f"No flood risk detected (Score: {risk_score:.2f})",
                'LOW': f"Low flood risk - Monitor conditions (Score: {risk_score:.2f})",
                'MEDIUM': f"⚠️ Moderate flood risk - Prepare emergency supplies (Score: {risk_score:.2f})",
                'HIGH': f"🚨 HIGH FLOOD RISK - Evacuate low-lying areas (Score: {risk_score:.2f})",
                'CRITICAL': f"🚨🚨 CRITICAL FLOOD ALERT - IMMEDIATE EVACUATION REQUIRED (Score: {risk_score:.2f})"
            },
            'earthquake': {
                'SAFE': f"No significant seismic activity (Score: {risk_score:.2f})",
                'LOW': f"Minor seismic activity detected (Score: {risk_score:.2f})",
                'MEDIUM': f"⚠️ Moderate earthquake risk - Secure loose objects (Score: {risk_score:.2f})",
                'HIGH': f"🚨 HIGH EARTHQUAKE RISK - Take cover immediately (Score: {risk_score:.2f})",
                'CRITICAL': f"🚨🚨 CRITICAL EARTHQUAKE ALERT - DROP, COVER, HOLD ON (Score: {risk_score:.2f})"
            },
            'cyclone': {
                'SAFE': f"Weather conditions normal (Score: {risk_score:.2f})",
                'LOW': f"Low cyclone risk - Monitor weather updates (Score: {risk_score:.2f})",
                'MEDIUM': f"⚠️ Cyclone watch - Secure outdoor items (Score: {risk_score:.2f})",
                'HIGH': f"🚨 HIGH CYCLONE RISK - Seek shelter indoors (Score: {risk_score:.2f})",
                'CRITICAL': f"🚨🚨 CRITICAL CYCLONE ALERT - IMMEDIATE SHELTER REQUIRED (Score: {risk_score:.2f})"
            }
        }
        return messages.get(disaster_type, {}).get(risk_level, f"Alert: {risk_level}")
    
    def should_trigger_alert(self, disaster_type: str, risk_level: str) -> bool:
        """Determine if alert should be triggered"""
        # Only alert on MEDIUM, HIGH, or CRITICAL
        return risk_level in ['MEDIUM', 'HIGH', 'CRITICAL']
    
    def log_alert(self, disaster_type: str, risk_level: str, risk_score: float, 
                  alert_message: str, acknowledged: bool = False):
        """Log alert to CSV file"""
        with open(self.alert_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                disaster_type,
                risk_level,
                f"{risk_score:.4f}",
                alert_message,
                acknowledged
            ])
    
    def create_alert(self, disaster_type: str, risk_score: float, risk_level: str) -> Optional[Dict]:
        """
        Create alert if conditions are met
        
        Returns:
            Alert dictionary if alert should be triggered, None otherwise
        """
        if not self.should_trigger_alert(disaster_type, risk_level):
            return None
        
        # Generate alert
        emoji = self.get_alert_emoji(disaster_type, risk_level)
        message = self.get_alert_message(disaster_type, risk_level, risk_score)
        
        alert = {
            'disaster_type': disaster_type,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'emoji': emoji,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False
        }
        
        # Log the alert
        self.log_alert(disaster_type, risk_level, risk_score, message)
        
        # Store as active alert
        self.active_alerts[disaster_type] = alert
        
        return alert
    
    def acknowledge_alert(self, disaster_type: str):
        """Acknowledge an active alert"""
        if disaster_type in self.active_alerts:
            self.active_alerts[disaster_type]['acknowledged'] = True
    
    def get_active_alerts(self) -> Dict:
        """Get all active alerts"""
        return {k: v for k, v in self.active_alerts.items() if not v['acknowledged']}
    
    def clear_all_alerts(self):
        """Clear all active alerts"""
        self.active_alerts.clear()
    
    def get_alert_statistics(self) -> Dict:
        """Get statistics from alert log"""
        if not self.alert_log_path.exists():
            return {}
        
        stats = {
            'total_alerts': 0,
            'by_disaster': {'flood': 0, 'earthquake': 0, 'cyclone': 0},
            'by_level': {'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        }
        
        with open(self.alert_log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats['total_alerts'] += 1
                if row['disaster_type'] in stats['by_disaster']:
                    stats['by_disaster'][row['disaster_type']] += 1
                if row['risk_level'] in stats['by_level']:
                    stats['by_level'][row['risk_level']] += 1
        
        return stats


if __name__ == "__main__":
    # Test alert system
    alert_system = DisasterAlertSystem()
    
    # Simulate some alerts
    alert1 = alert_system.create_alert('flood', 0.75, 'HIGH')
    if alert1:
        print(f"\n{alert1['emoji']} {alert1['message']}")
    
    alert2 = alert_system.create_alert('earthquake', 0.92, 'CRITICAL')
    if alert2:
        print(f"\n{alert2['emoji']} {alert2['message']}")
    
    # Get statistics
    stats = alert_system.get_alert_statistics()
    print(f"\n📊 Alert Statistics: {stats}")
