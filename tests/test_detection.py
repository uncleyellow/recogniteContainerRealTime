import unittest

class TestContainerDetection(unittest.TestCase):
    def setUp(self):
        self.detector = ContainerDetector()
        
    def test_valid_container_code(self):
        valid_codes = ["ABCD1234567", "WXYZ7654321"]
        for code in valid_codes:
            self.assertTrue(self.detector.is_valid_container_code(code))
            
    def test_invalid_container_code(self):
        invalid_codes = ["ABC123", "12345", "ABCD12345678"]
        for code in invalid_codes:
            self.assertFalse(self.detector.is_valid_container_code(code))
