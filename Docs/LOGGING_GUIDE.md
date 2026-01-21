# Logging System Guide

## Overview

The Voice Agent now has a comprehensive logging system that separates session-specific logs from runtime logs, making it easy to monitor and debug your application.

## Log Files Created

### Per-Session Logs
- **`storage/logs/session_YYYY-MM-DD_HH-MM-SS.log`** - Human-readable conversation transcript
- **`storage/logs/session_YYYY-MM-DD_HH-MM-SS.json`** - Structured JSON data

### Unified Logs
- **`storage/logs/runtime.log`** - All runtime events across all sessions
- **`storage/logs/sessions.log`** - All conversation activity across all sessions

## Real-Time Log Monitoring

Monitor logs in real-time using `tail -f`:

```bash
# Monitor all runtime events (initialization, errors, processing)
tail -f storage/logs/runtime.log

# Monitor all conversation activity
tail -f storage/logs/sessions.log

# Monitor both at once (split screen recommended)
tail -f storage/logs/runtime.log storage/logs/sessions.log
```

## Log Format Examples

### Runtime Log (`runtime.log`)
```
2025-11-25 12:30:45 [INFO] [Session: 2025-11-25_12-30-45] Voice Agent initialization started
2025-11-25 12:30:45 [INFO] [Session: 2025-11-25_12-30-45] Initializing Speech-to-Text component
2025-11-25 12:30:45 [INFO] [Session: 2025-11-25_12-30-45] Initializing Text-to-Speech component
2025-11-25 12:30:45 [INFO] [Session: 2025-11-25_12-30-45] Initializing LLM Agent component
2025-11-25 12:30:45 [INFO] [Session: 2025-11-25_12-30-45] Voice Agent initialization complete
2025-11-25 12:30:46 [INFO] [Session: 2025-11-25_12-30-45] Starting Speech-to-Text stream
2025-11-25 12:30:46 [INFO] [Session: 2025-11-25_12-30-45] Voice Agent ready - Conversation loop started
2025-11-25 12:30:52 [INFO] [Session: 2025-11-25_12-30-45] Processing user input: I need pickup from Mumbai...
2025-11-25 12:30:52 [INFO] [Session: 2025-11-25_12-30-45] USER: I need pickup from Mumbai
2025-11-25 12:30:52 [INFO] [Session: 2025-11-25_12-30-45] ASSISTANT: Great! And where would you like the goods delivered to?
2025-11-25 12:30:53 [INFO] [Session: 2025-11-25_12-30-45] TTS synthesis and playback successful
2025-11-25 12:31:05 [INFO] [Session: 2025-11-25_12-30-45] BOOKING UPDATE: pickup_location = Mumbai
2025-11-25 12:31:20 [INFO] [Session: 2025-11-25_12-30-45] Call ended by user (KeyboardInterrupt)
2025-11-25 12:31:20 [INFO] [Session: 2025-11-25_12-30-45] Shutdown initiated
2025-11-25 12:31:20 [INFO] [Session: 2025-11-25_12-30-45] Session ended - Duration: 35s, Turns: 4
2025-11-25 12:31:20 [INFO] [Session: 2025-11-25_12-30-45] Shutdown complete
```

### Sessions Log (`sessions.log`)
```
[2025-11-25 12:30:45] [Session: 2025-11-25_12-30-45] SESSION STARTED: 2025-11-25_12-30-45
[2025-11-25 12:30:52] [Session: 2025-11-25_12-30-45] USER: I need pickup from Mumbai
[2025-11-25 12:30:52] [Session: 2025-11-25_12-30-45] ASSISTANT: Great! And where would you like the goods delivered to?
[2025-11-25 12:31:00] [Session: 2025-11-25_12-30-45] USER: To Pune
[2025-11-25 12:31:00] [Session: 2025-11-25_12-30-45] ASSISTANT: Perfect. What type of vehicle do you need?
[2025-11-25 12:31:05] [Session: 2025-11-25_12-30-45] BOOKING UPDATE: pickup_location = Mumbai
[2025-11-25 12:31:05] [Session: 2025-11-25_12-30-45] BOOKING UPDATE: drop_location = Pune
[2025-11-25 12:31:20] [Session: 2025-11-25_12-30-45] CONFIRMATION STATUS: confirmed
[2025-11-25 12:31:20] [Session: 2025-11-25_12-30-45] SESSION ENDED - Duration: 35s, Turns: 4
[2025-11-25 12:31:20] [Session: 2025-11-25_12-30-45] ======================================================================
```

### Session-Specific Log (`session_*.log`)
```
======================================================================
DROPTRUCK AI SALES AGENT - CONVERSATION LOG
======================================================================
Session ID: 2025-11-25_12-30-45
Started: 2025-11-25 12:30:45
======================================================================

[12:30:52] USER: I need pickup from Mumbai
[12:30:52] ASSISTANT: Great! And where would you like the goods delivered to?
----------------------------------------------------------------------
[12:31:00] USER: To Pune
[12:31:00] ASSISTANT: Perfect. What type of vehicle do you need?
----------------------------------------------------------------------
[12:31:05] BOOKING UPDATE: pickup_location = Mumbai
[12:31:05] BOOKING UPDATE: drop_location = Pune

======================================================================
SESSION SUMMARY
======================================================================
Session ended: 2025-11-25 12:31:20
Duration: 35 seconds
Conversation turns: 4

FINAL BOOKING DATA:
----------------------------------------------------------------------
pickup_location................ Mumbai
drop_location.................. Pune
vehicle_type................... Truck
body_type...................... Container
goods_material................. FMCG
trip_date...................... Tomorrow
======================================================================
```

## Testing the Logging System

Run the test script:

```bash
python test_logging.py
```

This will create sample logs demonstrating all logging features.

## Advanced Usage

### Multiple Terminal Windows

For real-time monitoring during development:

```bash
# Terminal 1: Run the voice agent
python main.py

# Terminal 2: Monitor runtime logs
tail -f storage/logs/runtime.log

# Terminal 3: Monitor sessions logs
tail -f storage/logs/sessions.log
```

### Filtering Logs

Filter logs for specific events:

```bash
# Show only errors
grep ERROR storage/logs/runtime.log

# Show only booking updates
grep "BOOKING UPDATE" storage/logs/sessions.log

# Show logs from a specific session
grep "2025-11-25_12-30-45" storage/logs/runtime.log

# Show only conversation turns
grep -E "(USER:|ASSISTANT:)" storage/logs/sessions.log
```

### Log Analysis

Analyze session statistics:

```bash
# Count total sessions
grep "SESSION STARTED" storage/logs/sessions.log | wc -l

# Count booking updates
grep "BOOKING UPDATE" storage/logs/sessions.log | wc -l

# Find sessions with errors
grep -B 5 ERROR storage/logs/runtime.log
```

## Log Management

### Cleanup Old Session Logs

```bash
# Remove session logs older than 7 days
find storage/logs/session_*.log -mtime +7 -delete
find storage/logs/session_*.json -mtime +7 -delete
```

### Archive Logs

```bash
# Create dated archive
tar -czf logs_archive_$(date +%Y%m%d).tar.gz storage/logs/session_*.log storage/logs/session_*.json

# Move archive to backup location
mv logs_archive_*.tar.gz ~/backups/
```

### Rotate Unified Logs

For production use, consider setting up logrotate for `runtime.log` and `sessions.log`:

```bash
# Example logrotate config (save as /etc/logrotate.d/voice_agent)
/path/to/Voice_Agent/storage/logs/runtime.log /path/to/Voice_Agent/storage/logs/sessions.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 root root
}
```

## Programmatic Access

Access log files from your code:

```python
from core.logger import WorkflowLogger

logger = WorkflowLogger(logs_dir="storage/logs")

# Get log paths
session_log = logger.get_log_path()
json_log = logger.get_json_log_path()
runtime_log = logger.get_runtime_log_path()
sessions_log = logger.get_sessions_log_path()

# Log custom events
logger.log_info("Custom info message")
logger.log_warning("Custom warning message")
logger.log_error("Custom error message")
```

## Troubleshooting

### Logs Not Being Created

Check that the `storage/logs/` directory exists and is writable:
```bash
ls -la storage/logs/
chmod 755 storage/logs/
```

### Permission Denied Errors

Ensure the application has write permissions:
```bash
chmod -R 755 storage/logs/
```

### Logs Too Large

Implement log rotation or cleanup scripts as shown above.

## Benefits

✅ **Separate concerns**: Session logs are isolated, unified logs provide overview  
✅ **Real-time monitoring**: Use `tail -f` to watch logs live  
✅ **Easy debugging**: Runtime logs show initialization, errors, and warnings  
✅ **Conversation tracking**: Sessions log provides complete conversation history  
✅ **Structured data**: JSON logs enable programmatic analysis  
✅ **Session correlation**: All logs include session ID for tracking  

## Next Steps

1. Run `python test_logging.py` to verify the setup
2. Start the voice agent: `python main.py`
3. In another terminal, run: `tail -f storage/logs/runtime.log`
4. Have a conversation and watch the logs update in real-time!

