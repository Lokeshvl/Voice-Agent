import os
import threading
import sounddevice as sd
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV2SocketClientResponse, SpeakV1SocketClientResponse, \
    SpeakV1ControlMessage, ListenV2MediaMessage, SpeakV1TextMessage


class SpeechToText:
    """Manages real-time speech-to-text using Deepgram."""
    
    def __init__(self, api_key: str = None):
        """Initialize STT with Deepgram API key."""
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY is required. Set it in environment or .env file."
            )
        
        self.client = DeepgramClient(api_key=self.api_key)
        self.connection = None
        self.conn_cm = None  # Connection context manager
        self.listener_thread = None
        
        # Callback handlers
        self.on_transcript_callback = None
        self.on_final_callback = None
        self.on_error_callback = None
    
    def _attempt_connection(self):
        """Connect to Deepgram with working parameters."""
        # Keywords to boost recognition (Indian cities and truck terms)
        keywords = [
            # Major Indian cities
            "Mumbai", "Delhi", "Bangalore", "Bengaluru", "Chennai", "Kolkata", 
            "Hyderabad", "Pune", "Ahmedabad", "Surat", "Jaipur", "Lucknow",
            "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam",
            "Pimpri", "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra",
            "Nashik", "Faridabad", "Meerut", "Rajkot", "Kalyan", "Vasai",
            "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar",
            "Navi Mumbai", "Allahabad", "Ranchi", "Howrah", "Coimbatore",
            "Jabalpur", "Gwalior", "Vijayawada", "Jodhpur", "Madurai",
            "Raipur", "Kota", "Guwahati", "Chandigarh", "Solapur",
            # Truck types
            "Tata", "Ace", "Dost", "Bolero", "Ashok", "Leyland", "Eicher",
            # Common terms
            "pickup", "drop", "container", "open", "truck", "vehicle"
        ]


        params = {
            "model": "nova-2-general",  # v2 works best with nova-2
            "encoding": "linear16",
            "sample_rate": 16000,
            "endpointing": 1000,  # 1 second of silence before finalizing
            "interim_results": True,  # Get real-time transcription
            "vad_events": True,  # Voice Activity Detection for noise suppression
            "smart_format": True,  # Better formatting and punctuation
        }

        print(f"[STT] Connecting with Deepgram v2 API")
        print(f"[STT] Model: {params['model']}")
        print(f"[STT] Endpointing: {params['endpointing']}ms")
        print(f"[STT] VAD enabled for noise suppression")
        try:
            conn = self.client.listen.v1.connect(
            model="nova-3",
            language="en-US",
            # Apply smart formatting to the output
            smart_format=True,
            # Raw audio format details
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            # To get UtteranceEnd, the following must be set:
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            # Time in milliseconds of silence to wait for before finalizing speech
            endpointing=300)
            connection = conn.__enter__()
            print(f"✅ STT Connected successfully")
            return conn, connection, params
        except Exception as e:
            print(f"❌ Failed to connect to Deepgram: {repr(e)}")
            raise RuntimeError(f"STT connection failed: {repr(e)}")
    
    def _on_open(self, event):
        """Handle connection open event."""
        print("🟢 STT connection opened. Listening...")
    
    def _on_message(self, message):
        """Handle incoming transcription messages."""
        if hasattr(message, "channel") and hasattr(message.channel, "alternatives"):
            alt = message.channel.alternatives[0]
            transcript = alt.transcript
            
            if transcript:
                # Call the transcript callback for partial results
                if self.on_transcript_callback:
                    self.on_transcript_callback(transcript)
                
                # Check for final transcription
                if hasattr(message, 'is_final') and message.is_final:
                    if self.on_final_callback:
                        self.on_final_callback(transcript)
        
        # Check for VAD (voice activity detection) final event
        if hasattr(message, 'speech_final') and message.speech_final:
            if self.on_final_callback:
                self.on_final_callback("")
    
    def _on_close(self, event):
        """Handle connection close event."""
        print("🔴 STT connection closed")
    
    def _on_error(self, err):
        """Handle connection errors."""
        print("❌ STT Error:", err)
        if self.on_error_callback:
            self.on_error_callback(err)
    
    def start(self, on_transcript=None, on_final=None, on_error=None):
        """
        Start the STT connection and audio streaming.
        
        Args:
            on_transcript: Callback for partial transcripts (receives text)
            on_final: Callback for final transcripts (receives text)
            on_error: Callback for errors (receives error object)
        """
        # Set callbacks
        self.on_transcript_callback = on_transcript
        self.on_final_callback = on_final
        self.on_error_callback = on_error
        
        # Establish connection
        self.conn_cm, self.connection, self.used_params = self._attempt_connection()
        
        # Register event handlers
        self.connection.on(EventType.OPEN, self._on_open)
        self.connection.on(EventType.MESSAGE, self._on_message)
        self.connection.on(EventType.CLOSE, self._on_close)
        self.connection.on(EventType.ERROR, self._on_error)
        
        # Start listening thread
        def listener_thread():
            self.connection.start_listening()
        
        self.listener_thread = threading.Thread(target=listener_thread, daemon=True)
        self.listener_thread.start()
        
        print(f"🎤 STT started with model: {self.used_params['model']}, "
              f"sample_rate: {self.used_params['sample_rate']}")
    
    def get_audio_callback(self):
        """
        Returns a callback function for sounddevice audio streaming.
        Use this with sd.InputStream(callback=stt.get_audio_callback(), ...)
        """
        def audio_callback(indata, frames, time_info, status):
            if status:
                print("Audio status:", status)
            try:
                if self.connection:
                    self.connection.send_media(indata.tobytes())
            except Exception as e:
                print("Failed sending audio:", e)
        
        return audio_callback
    
    def stop(self):
        """Stop the STT connection and cleanup."""
        try:
            if self.connection:
                # Close the connection properly
                if hasattr(self.connection, 'close'):
                    self.connection.close()
            if self.conn_cm:
                # Exit the context manager
                self.conn_cm.__exit__(None, None, None)
            if self.listener_thread:
                self.listener_thread.join(timeout=2)
            print("🛑 STT stopped")
        except Exception as e:
            print(f"⚠️  Error stopping STT: {e}")
            # Continue anyway to allow shutdown to proceed
