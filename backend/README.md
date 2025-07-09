# Drowsiness Detection Backend API

A Flask-based REST API backend for the Drowsiness Detection System. This backend provides data storage, analytics, and real-time monitoring capabilities for the detection scripts.

## Features

- RESTful API: Complete REST API for event logging and retrieval
- SQLite Database: Lightweight database for storing detection events and sessions
- Real-time Monitoring: Session tracking and event broadcasting
- Configuration Management: Dynamic system configuration updates
- Analytics: Statistical analysis and reporting
- CORS Support: Cross-origin request support for web applications
- Health Monitoring: API health checks and status monitoring

## Quick Start

### 1. Install Dependencies

```bash
# From project root
pip install -r requirements.txt

# Or install backend-specific dependencies
pip install -r backend/requirements.txt
```

### 2. Start the Backend

```bash
# Option 1: Use the startup script
python start_backend.py

# Option 2: Start directly
cd backend
python app.py
```

### 3. Verify Installation

Visit http://localhost:8080 to see the API documentation.

## API Endpoints

### Health & Status
- `GET /api/health` - Check API health and database connection

### Event Management
- `POST /api/events` - Log detection events
- `GET /api/events` - Retrieve events with filtering

### Session Management
- `POST /api/sessions` - Start monitoring session
- `GET /api/sessions` - Get user sessions
- `PUT /api/sessions/{id}` - End session

### Configuration
- `GET /api/config` - Get all configuration values
- `GET /api/config/{key}` - Get specific configuration
- `PUT /api/config/{key}` - Update configuration

### Analytics
- `GET /api/stats` - Get system statistics

### Real-time Alerts
- `POST /api/alerts/register` - Register for alerts

## Database Schema

### DetectionEvent
```sql
CREATE TABLE detection_event (
    id INTEGER PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    duration FLOAT DEFAULT 0.0,
    details TEXT,
    user_id VARCHAR(50) DEFAULT 'default_user'
);
```

### UserSession
```sql
CREATE TABLE user_session (
    id INTEGER PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,
    total_events INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active'
);
```

### SystemConfig
```sql
CREATE TABLE system_config (
    id INTEGER PRIMARY KEY,
    key VARCHAR(100) UNIQUE NOT NULL,
    value TEXT NOT NULL,
    description VARCHAR(200),
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Integration with Detection Scripts

### Using the Backend Client

```python
from backend_client import DrowsinessDetectionClient

# Create client
client = DrowsinessDetectionClient(user_id="driver_001")

# Start session
client.start_session()

# Log events
client.log_drowsiness(ear=0.15, frame_count=20, duration=5.2)
client.log_distraction(direction="left", duration=2.1)
client.log_yawn(mar=35.5, duration=1.8)

# Get statistics
stats = client.get_statistics(days=7)

# End session
client.end_session()
```

### Direct API Calls

```python
import requests

# Log drowsiness event
response = requests.post("http://localhost:5000/api/events", json={
    "event_type": "drowsiness",
    "severity": "high",
    "duration": 5.2,
    "details": {"ear": 0.15, "frame_count": 20},
    "user_id": "driver_001"
})

# Get events
events = requests.get("http://localhost:5000/api/events?event_type=drowsiness&limit=10")
```

## Configuration

### Default Configuration Values

| Key | Default Value | Description |
|-----|---------------|-------------|
| `EYE_AR_THRESH` | `0.25` | Eye Aspect Ratio threshold |
| `EYE_AR_CONSEC_FRAMES` | `4` | Consecutive frames threshold |
| `DISC_COUNT_THRES` | `7` | Distraction detection threshold |
| `MAR_THRES` | `29` | Mouth Aspect Ratio threshold |
| `ALERT_ENABLED` | `true` | Enable audio alerts |
| `LOG_ENABLED` | `true` | Enable event logging |
| `API_ENABLED` | `true` | Enable API endpoints |

### Updating Configuration

```python
# Via API
requests.put("http://localhost:5000/api/config/EYE_AR_THRESH", json={
    "value": "0.30",
    "description": "Updated threshold for better sensitivity"
})

# Via client
client.update_config("EYE_AR_THRESH", "0.30", "Updated threshold")
```

## Development

### Project Structure
```
backend/
├── app.py              # Main Flask application
├── requirements.txt    # Backend dependencies
├── README.md          # This file
└── templates/
    └── index.html     # API documentation page
```

### Running in Development Mode

```bash
cd backend
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Database Management

The backend uses SQLite for simplicity. The database file `drowsiness_detection.db` is created automatically in the backend directory.

To reset the database:
```bash
cd backend
rm drowsiness_detection.db
python app.py  # Database will be recreated
```

## Deployment

### Production Considerations

1. **Use a Production WSGI Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Environment Variables**
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secret-key
   ```

3. **Database Migration**
   - Consider using Flask-Migrate for database migrations
   - Use PostgreSQL for production databases

4. **Security**
   - Enable HTTPS
   - Implement authentication
   - Rate limiting
   - Input validation

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Monitoring and Logging

### Health Checks

```bash
# Check API health
curl http://localhost:5000/api/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "version": "1.0.0",
  "database": "connected"
}
```

### Logging

The backend logs to stdout. For production, consider using a proper logging framework:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 5000
   lsof -i :5000
   # Kill process
   kill -9 <PID>
   ```

2. **Database Locked**
   ```bash
   # Remove database file and restart
   rm backend/drowsiness_detection.db
   python app.py
   ```

3. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r backend/requirements.txt
   ```

4. **CORS Issues**
   - Check CORS configuration in app.py
   - Verify frontend origin is allowed

### Debug Mode

Enable debug mode for detailed error messages:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## API Documentation

Visit http://localhost:5000 for interactive API documentation with examples and testing interface.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the API endpoints
5. Submit a pull request

## License

This project is licensed under the MIT License. 