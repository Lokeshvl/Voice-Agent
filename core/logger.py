import os
import json
from datetime import datetime
from typing import Dict, Any
import logging


class WorkflowLogger:
    """Logs conversation workflow and booking information."""
    
    def __init__(self, logs_dir: str = "logs"):
        """
        Initialize the workflow logger.
        
        Args:
            logs_dir: Directory to store log files
        """
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate session ID with timestamp
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_start = datetime.now()
        
        # Log file paths
        self.text_log_path = os.path.join(logs_dir, f"session_{self.session_id}.log")
        self.json_log_path = os.path.join(logs_dir, f"session_{self.session_id}.json")
        
        # Unified log file paths
        self.runtime_log_path = os.path.join(logs_dir, "runtime.log")
        self.sessions_log_path = os.path.join(logs_dir, "sessions.log")

        # Conversation data
        self.conversation_turns = []
        self.booking_updates = []
        
        # Initialize log files
        self._init_logs()
        self._setup_runtime_logger()

    def _init_logs(self):
        """Initialize log files with session header."""
        # Text log header
        with open(self.text_log_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DROPTRUCK AI SALES AGENT - CONVERSATION LOG\n")
            f.write("="*70 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
        
        # JSON log initialization
        self.json_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "conversation": [],
            "booking_data": {},
            "session_end": None,
            "duration_seconds": None
        }
        
        print(f"📝 Logging to: {self.text_log_path}")
    
    def _setup_runtime_logger(self):
        """Setup unified runtime logger for tail -f functionality."""
        # Configure runtime logger
        self.runtime_logger = logging.getLogger('droptruck_runtime')
        self.runtime_logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        self.runtime_logger.handlers.clear()

        # Create file handler for runtime logs
        runtime_handler = logging.FileHandler(self.runtime_log_path)
        runtime_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [Session: %(session_id)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        runtime_handler.setFormatter(formatter)

        # Add handler to logger
        self.runtime_logger.addHandler(runtime_handler)

        # Log session start to runtime log
        self.runtime_logger.info(
            f"Session started",
            extra={'session_id': self.session_id}
        )

        # Also log to unified sessions log
        self._log_to_sessions_file(f"SESSION STARTED: {self.session_id}")

    def _log_to_sessions_file(self, message: str):
        """
        Append a message to the unified sessions log file.

        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.sessions_log_path, 'a') as f:
            f.write(f"[{timestamp}] [Session: {self.session_id}] {message}\n")

    def log_conversation_turn(self, user_input: str, assistant_response: str):
        """
        Log a conversation turn (user input + assistant response).
        
        Args:
            user_input: What the user said
            assistant_response: What the assistant responded
        """
        timestamp = datetime.now()
        
        # Add to conversation turns
        turn = {
            "timestamp": timestamp.isoformat(),
            "user": user_input,
            "assistant": assistant_response
        }
        self.conversation_turns.append(turn)
        self.json_data["conversation"].append(turn)
        
        # Write to text log
        with open(self.text_log_path, 'a') as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] USER: {user_input}\n")
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] ASSISTANT: {assistant_response}\n")
            f.write("-" * 70 + "\n")

        # Log to runtime logger
        self.runtime_logger.info(
            f"USER: {user_input}",
            extra={'session_id': self.session_id}
        )
        self.runtime_logger.info(
            f"ASSISTANT: {assistant_response}",
            extra={'session_id': self.session_id}
        )

        # Log to unified sessions log
        self._log_to_sessions_file(f"USER: {user_input}")
        self._log_to_sessions_file(f"ASSISTANT: {assistant_response}")

    def log_booking_update(self, field: str, value: str):
        """
        Log a booking field update.
        
        Args:
            field: Field name that was updated
            value: New value
        """
        timestamp = datetime.now()
        
        update = {
            "timestamp": timestamp.isoformat(),
            "field": field,
            "value": value
        }
        self.booking_updates.append(update)
        
        # Write to text log
        with open(self.text_log_path, 'a') as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] BOOKING UPDATE: {field} = {value}\n")

        # Log to runtime logger
        self.runtime_logger.info(
            f"BOOKING UPDATE: {field} = {value}",
            extra={'session_id': self.session_id}
        )

        # Log to unified sessions log
        self._log_to_sessions_file(f"BOOKING UPDATE: {field} = {value}")

    def log_confirmation_status(self, status: str):
        """
        Log confirmation status change.
        
        Args:
            status: New confirmation status (confirmed, not_interested, pending)
        """
        timestamp = datetime.now()
        
        with open(self.text_log_path, 'a') as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] CONFIRMATION STATUS: {status}\n")
            f.write("-" * 70 + "\n")
        
        self.json_data["confirmation_status"] = status

        # Log to runtime logger
        self.runtime_logger.info(
            f"CONFIRMATION STATUS: {status}",
            extra={'session_id': self.session_id}
        )

        # Log to unified sessions log
        self._log_to_sessions_file(f"CONFIRMATION STATUS: {status}")

    def log_session_end(self, booking_data: Dict[str, Any]):
        """
        Log session end with final booking summary.
        
        Args:
            booking_data: Final booking data dictionary
        """
        session_end = datetime.now()
        duration = (session_end - self.session_start).total_seconds()
        
        # Update JSON data
        self.json_data["session_end"] = session_end.isoformat()
        self.json_data["duration_seconds"] = duration
        self.json_data["booking_data"] = booking_data
        self.json_data["booking_updates"] = self.booking_updates
        
        # Write to text log
        with open(self.text_log_path, 'a') as f:
            f.write("\n" + "="*70 + "\n")
            f.write("SESSION SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Session ended: {session_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {int(duration)} seconds\n")
            f.write(f"Conversation turns: {len(self.conversation_turns)}\n")
            f.write("\nFINAL BOOKING DATA:\n")
            f.write("-" * 70 + "\n")
            for key, value in booking_data.items():
                status = value if value else "[NOT PROVIDED]"
                f.write(f"{key:.<30} {status}\n")
            f.write("="*70 + "\n")
        
        # Write JSON log
        with open(self.json_log_path, 'w') as f:
            json.dump(self.json_data, f, indent=2)
        
        # Log to runtime logger
        self.runtime_logger.info(
            f"Session ended - Duration: {int(duration)}s, Turns: {len(self.conversation_turns)}",
            extra={'session_id': self.session_id}
        )

        # Log session summary to unified sessions log
        self._log_to_sessions_file(f"SESSION ENDED - Duration: {int(duration)}s, Turns: {len(self.conversation_turns)}")
        self._log_to_sessions_file("=" * 70)

        print(f"✅ Session log saved: {self.text_log_path}")
        print(f"✅ JSON log saved: {self.json_log_path}")
    
    def log_info(self, message: str):
        """
        Log an informational runtime message.

        Args:
            message: Message to log
        """
        self.runtime_logger.info(message, extra={'session_id': self.session_id})

    def log_warning(self, message: str):
        """
        Log a warning runtime message.

        Args:
            message: Warning message to log
        """
        self.runtime_logger.warning(message, extra={'session_id': self.session_id})

    def log_error(self, message: str):
        """
        Log an error runtime message.

        Args:
            message: Error message to log
        """
        self.runtime_logger.error(message, extra={'session_id': self.session_id})

    def get_log_path(self) -> str:
        """Get the text log file path."""
        return self.text_log_path
    
    def get_json_log_path(self) -> str:
        """Get the JSON log file path."""
        return self.json_log_path

    def get_runtime_log_path(self) -> str:
        """Get the unified runtime log file path."""
        return self.runtime_log_path

    def get_sessions_log_path(self) -> str:
        """Get the unified sessions log file path."""
        return self.sessions_log_path
