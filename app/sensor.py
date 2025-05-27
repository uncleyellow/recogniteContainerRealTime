#import RPi.GPIO as GPIO
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class LightSensor:
    def __init__(self, pin: int, callback: Callable = None):
        """
        Initialize light sensor
        :param pin: GPIO pin number for the sensor
        :param callback: Function to call when motion is detected
        """
        self.pin = pin
        self.callback = callback
        self.is_detecting = False
        
        # Setup GPIO
        #GPIO.setmode(GPIO.BCM)
        #GPIO.setup(self.pin, GPIO.IN)
        
        # Add event detection
        #GPIO.add_event_detect(self.pin, GPIO.RISING, 
        #                    callback=self._motion_detected,
        #                    bouncetime=300)
        
        logger.info(f"Light sensor initialized on pin {pin}")

    def _motion_detected(self, channel):
        """Callback when motion is detected"""
        if not self.is_detecting and self.callback:
            self.is_detecting = True
            try:
                self.callback()
            except Exception as e:
                logger.error(f"Error in light sensor callback: {str(e)}")
            finally:
                self.is_detecting = False

    def cleanup(self):
        """Cleanup GPIO resources"""
        #GPIO.remove_event_detect(self.pin)
        #GPIO.cleanup(self.pin)
        logger.info("Light sensor cleaned up")

class SensorManager:
    def __init__(self):
        self.sensors = {}
        self.is_running = False

    def add_sensor(self, name: str, pin: int, callback: Callable):
        """Add a new light sensor"""
        if name in self.sensors:
            logger.warning(f"Sensor {name} already exists")
            return False
            
        try:
            sensor = LightSensor(pin, callback)
            self.sensors[name] = sensor
            logger.info(f"Added sensor {name} on pin {pin}")
            return True
        except Exception as e:
            logger.error(f"Failed to add sensor {name}: {str(e)}")
            return False

    def remove_sensor(self, name: str):
        """Remove a light sensor"""
        if name in self.sensors:
            self.sensors[name].cleanup()
            del self.sensors[name]
            logger.info(f"Removed sensor {name}")
            return True
        return False

    def cleanup(self):
        """Cleanup all sensors"""
        for name, sensor in self.sensors.items():
            try:
                sensor.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up sensor {name}: {str(e)}")
        self.sensors.clear()
        logger.info("All sensors cleaned up")

# Singleton instance
_sensor_manager = None

def get_sensor_manager() -> SensorManager:
    """Get the singleton sensor manager instance"""
    global _sensor_manager
    if _sensor_manager is None:
        _sensor_manager = SensorManager()
    return _sensor_manager
