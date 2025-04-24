from fastapi import HTTPException
from datetime import datetime
import logging
import os

class RateLimiter:
    def __init__(self, limit=100, window=60):
        self.requests = {}
        self.limit = limit
        self.window = window

    def check_limit(self, client_ip):
        now = datetime.now()
        if client_ip not in self.requests:
            self.requests[client_ip] = [now]
            return True
        
        self.requests[client_ip] = [t for t in self.requests[client_ip] 
                                  if (now - t).seconds < self.window]
        if len(self.requests[client_ip]) < self.limit:
            self.requests[client_ip].append(now)
            return True
        return False

class AccessLogger:
    def __init__(self):
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Setup logger
        logger = logging.getLogger('access_log')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler('logs/access.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        return logger

    def log_access(self, request):
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ip': request.client.host,
            'method': request.method,
            'url': str(request.url),
            'user_agent': request.headers.get('user-agent', 'Unknown')
        }
        
        self.logger.info(
            f"Access: {log_data['ip']} - {log_data['method']} {log_data['url']} "
            f"- {log_data['user_agent']}"
        )

class SecurityManager:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.access_logger = AccessLogger()
        
    def validate_request(self, request):
        if not self.rate_limiter.check_limit(request.client.host):
            raise HTTPException(status_code=429)
        self.access_logger.log_access(request)
