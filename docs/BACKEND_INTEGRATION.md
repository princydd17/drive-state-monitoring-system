# Backend Integration Guide

Complete Backend System Added!

Your drowsiness detection project now has a fully compatible backend API that works seamlessly with your existing detection scripts.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DROWSINESS DETECTION SYSTEM              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Detection     │    │   Backend       │                │
│  │   Scripts       │◄──►│   API           │                │
│  │                 │    │                 │                │
│  │ • driver_alert.py│    │ • Flask Server  │                │
│  │ • driver_eye_only.py│ │ • SQLite DB     │                │
│  │ • distraction_detection.py│ │ • REST API     │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Backend       │    │   Web           │                │
│  │   Client        │    │   Dashboard     │                │
│  │                 │    │                 │                │
│  │ • API Client    │    │ • API Docs      │                │
│  │ • Event Logging │    │ • Statistics    │                │
│  │ • Session Mgmt  │    │ • Real-time     │                │
│  └─────────────────┘    └─────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. **Start the Backend**
```bash
# Option 1: Use startup script
python start_backend.py

# Option 2: Start directly
cd backend
python app.py
```

### 2. **Access the API**
- **API Documentation**: http://localhost:5000
- **Health Check**: http://localhost:5000/api/health
- **Database**: `backend/drowsiness_detection.db`

### 3. **Test Integration**
```bash
# Test backend client
python backend_client.py

# Run detection with backend logging
python driver_alert.py
```

## New Files Added

### Backend System
- `backend/app.py` - Flask API server
- `backend/requirements.txt` - Backend dependencies
- `backend/templates/index.html` - API documentation
- `backend/README.md` - Backend documentation

### Integration
- `backend_client.py` - Python client for API integration
- `start_backend.py` - Backend startup script
- `BACKEND_INTEGRATION.md` - This guide

### Updated Files
- `requirements.txt` - Added backend dependencies
- `README.md` - Updated with backend information

## Integration Options

### Option 1: **Automatic Integration** (Recommended)
The backend client automatically connects when available:

```python
# Your existing detection scripts work as-is
# Backend logging happens automatically when connected
python driver_alert.py
```

### Option 2: **Manual Integration**
Add backend logging to your scripts:

```python
from backend_client import log_drowsiness_event, log_distraction_event

# In your detection code
if ear < EYE_AR_THRESH:
    log_drowsiness_event(ear=ear, frame_count=COUNTER, duration=duration)

if distraction_detected:
    log_distraction_event(direction="left", duration=duration)
```

### Option 3: **Direct API Calls**
Make direct HTTP requests:

```python
import requests

# Log event
requests.post("http://localhost:5000/api/events", json={
    "event_type": "drowsiness",
    "severity": "high",
    "duration": 5.2,
    "details": {"ear": 0.15},
    "user_id": "driver_001"
})
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/events` | Log detection event |
| `GET` | `/api/events` | Get events with filtering |
| `POST` | `/api/sessions` | Start monitoring session |
| `GET` | `/api/sessions` | Get user sessions |
| `PUT` | `/api/sessions/{id}` | End session |
| `GET` | `/api/config` | Get configuration |
| `PUT` | `/api/config/{key}` | Update configuration |
| `GET` | `/api/stats` | Get statistics |

## Database Schema

### Detection Events
```sql
- id (Primary Key)
- event_type (drowsiness, distraction, yawn)
- severity (low, medium, high)
- timestamp
- duration
- details (JSON)
- user_id
```

### User Sessions
```sql
- id (Primary Key)
- user_id
- start_time
- end_time
- total_events
- status (active, completed, paused)
```

### System Configuration
```sql
- id (Primary Key)
- key
- value
- description
- updated_at
```

## Features Added

### ✅ **Data Storage**
- Persistent event logging
- Session tracking
- Configuration management

### ✅ **Analytics**
- Event statistics
- Time-based analysis
- Severity tracking

### ✅ **Real-time Monitoring**
- Live session tracking
- Event broadcasting
- Health monitoring

### ✅ **API Management**
- RESTful endpoints
- CORS support
- Error handling

### ✅ **Configuration**
- Dynamic threshold updates
- System settings
- User preferences

## Migration from CSV to Database

Your existing CSV logging still works! The backend adds database storage alongside CSV:

```python
# Existing CSV logging (still works)
df.to_csv('{}.csv'.format(NAME), mode='a', header=False)

# New database logging (automatic when backend is running)
client.log_event("drowsiness", "high", 5.2, {"ear": 0.15})
```

## Development Workflow

### 1. **Start Backend**
```bash
python start_backend.py
```

### 2. **Run Detection**
```bash
python driver_alert.py
# or
python driver_eye_only.py
```

### 3. **Monitor Results**
- Visit http://localhost:5000 for API dashboard
- Check database: `backend/drowsiness_detection.db`
- View CSV logs: `Driver.csv`

### 4. **Analyze Data**
```python
from backend_client import DrowsinessDetectionClient

client = DrowsinessDetectionClient()
stats = client.get_statistics(days=7)
events = client.get_events(event_type="drowsiness")
```

## Configuration Management

### Update Detection Thresholds
```python
# Via API
requests.put("http://localhost:5000/api/config/EYE_AR_THRESH", json={
    "value": "0.30"
})

# Via client
client.update_config("EYE_AR_THRESH", "0.30")
```

### Get Current Settings
```python
config = client.get_config()
print(f"Eye threshold: {config['EYE_AR_THRESH']}")
```

## Troubleshooting

### Backend Won't Start
```bash
# Check dependencies
pip install -r requirements.txt

# Check port availability
lsof -i :5000

# Start with debug
cd backend && python app.py
```

### Client Connection Issues
```bash
# Test backend health
curl http://localhost:5000/api/health

# Check firewall settings
# Ensure backend is running
```

### Database Issues
```bash
# Reset database
rm backend/drowsiness_detection.db
cd backend && python app.py
```

## Future Enhancements

### Web Dashboard
- Real-time monitoring interface
- Charts and graphs
- User management

### Mobile App
- iOS/Android companion app
- Push notifications
- Offline support

### Cloud Integration
- AWS/Google Cloud deployment
- Multi-user support
- Advanced analytics

### Machine Learning
- Predictive analytics
- Pattern recognition
- Personalized thresholds

## Next Steps

1. **Start the backend**: `python start_backend.py`
2. **Test integration**: `python backend_client.py`
3. **Run detection**: `python driver_alert.py`
4. **Monitor results**: Visit http://localhost:5000
5. **Explore API**: Use the interactive documentation

## Support

- **API Documentation**: http://localhost:5000
- **Backend Docs**: `backend/README.md`
- **Integration Guide**: This file
- **Main README**: `README.md`

---

**🎉 Congratulations!** Your drowsiness detection system now has a powerful backend that provides data storage, analytics, and real-time monitoring while maintaining full compatibility with your existing detection scripts. 