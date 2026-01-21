# DropTruck Voice Agent - Technical Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Main Components](#main-components)
3. [Token Usage & Costs](#token-usage--costs)
4. [Detailed Code Explanation](#detailed-code-explanation)
5. [Audio Recording System](#audio-recording-system)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                    (VoiceAgent Class)                       │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │   STT    │  │   LLM    │  │   TTS    │  │  Logger  │     │
│  │ (Deepgram│  │ (OpenAI) │  │ (Deepgram│  │          │     │
│  │ Real-time│  │          │  │ Aura)    │  │          │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│       │             │              │              │         │
│       │             │              │              │         │
└───────┼─────────────┼──────────────┼──────────────┼─────────┘
        │             │              │              │
        ▼             ▼              ▼              ▼
   Microphone    Conversation    Speaker       Log Files
   (User Input)   Management    (Audio Output)  (.log/.json)
```

### Data Flow

1. **User speaks** → Microphone captures audio
2. **STT (Deepgram)** → Converts speech to text in real-time
3. **LLM (OpenAI)** → Processes text, generates response
4. **Logger** → Records conversation turn
5. **TTS (Deepgram)** → Converts response to speech
6. **Speaker** → Plays audio to user
7. **Loop** → Return to step 1

---

## Main Components

### 1. main.py - Voice Agent Orchestrator

**Purpose**: Coordinates all components and manages the conversation loop.

**Key Classes & Methods**:

```python
class VoiceAgent:
    def __init__(self):
        # Initializes all components (STT, TTS, LLM, Logger)
        # Sets up conversation state tracking
        
    def on_transcript(self, text: str):
        # Callback for partial transcription results
        # Accumulates text segments
        
    def on_final(self, text: str):
        # Callback when transcription segment is complete
        # Triggers processing
        
    def on_error(self, error):
        # Handles STT errors
        
    def process_user_input(self, user_text: str):
        # Main processing pipeline:
        # 1. User text → LLM
        # 2. LLM response → TTS
        # 3. Play audio
        
    def run(self):
        # Main conversation loop
        # Continuously listens and processes
        
    def shutdown(self):
        # Clean shutdown
        # Saves logs and cleans up resources
```

**State Management**:
- `current_transcript`: List of partial transcription segments
- `segment_ready`: Boolean flag indicating when to process input
- `audio_stream`: Audio input stream reference

---

### 2. app/stt.py - Speech-to-Text

**Purpose**: Real-time audio capture and transcription using Deepgram.

**Key Features**:
- Streams audio from microphone to Deepgram
- Handles partial and final transcription events
- Connection fallback (tries multiple model configurations)
- VAD (Voice Activity Detection) for natural conversation breaks

**API Usage**:
```python
# Deepgram STT API
model: "nova-2-multilingual" or "nova-3"
encoding: "pcm_s16le"
sample_rate: 16000 Hz
vad_events: True  # Voice Activity Detection
punctuate: True   # Automatic punctuation
```

**Cost**: Deepgram charges per minute of audio
- Pay-as-you-go: ~$0.0125/minute
- Usage-based plans available

**No Token Limit**: Deepgram processes audio streams without token limits.

---

### 3. app/llm.py - Language Model Integration

**Purpose**: Conversation management and booking data extraction using OpenAI.

**Key Features**:
- Maintains conversation history
- Manages system prompt (DropTruck call script)
- Extracts booking information
- Detects confirmation/rejection keywords
- Logs all interactions

**API Usage**:
```python
# OpenAI Chat Completions API
model: "gpt-4o-mini" (default)
temperature: 0.7
max_tokens: 256 per response
```

**Token Usage Calculation**:

Each API call includes:
1. **System Prompt**: ~450 tokens (DropTruck call script)
2. **Conversation History**: Variable (keeps last 10 exchanges = ~20 messages)
3. **Current User Input**: Variable (typically 10-50 tokens)
4. **Response**: Max 256 tokens

**Example Token Breakdown** (6-turn conversation):

| Turn | System | History | User Input | Response | Total Input | Total Output |
|------|--------|---------|------------|----------|-------------|--------------|
| 1    | 450    | 0       | 20         | 40       | 470         | 40           |
| 2    | 450    | 60      | 25         | 35       | 535         | 35           |
| 3    | 450    | 130     | 15         | 30       | 595         | 30           |
| 4    | 450    | 190     | 30         | 45       | 670         | 45           |
| 5    | 450    | 280     | 10         | 50       | 740         | 50           |
| 6    | 450    | 380     | 5          | 35       | 835         | 35           |

**Total**: ~4,000 input tokens + ~235 output tokens = ~4,235 tokens

**Cost Estimate** (gpt-4o-mini):
- Input: $0.150 per 1M tokens
- Output: $0.600 per 1M tokens
- Per conversation (6 turns): ~$0.0006 input + $0.00014 output = **$0.00074** (~$0.001 per call)

**Optimization**:
- History pruning: Keeps only last 10 exchanges (20 messages)
- Max tokens: Limited to 256 per response
- Temperature: 0.7 for consistent but natural responses

---

### 4. app/tts.py - Text-to-Speech

**Purpose**: Synthesizes assistant responses to natural-sounding speech using Deepgram.

**Key Features**:
- Converts text to MP3 audio files
- Cross-platform audio playback (macOS, Linux, Windows)
- Automatic cleanup of old audio files
- Saves audio to `audio_output/` directory

**API Usage**:
```python
# Deepgram TTS API
model: "aura-asteria-en"
output_format: MP3
```

**Cost**: Deepgram TTS charges per character
- Pay-as-you-go: ~$0.015 per 1,000 characters
- Average response: ~100 characters
- **Per response**: ~$0.0015

**No Token Limit**: Character-based pricing.

---

### 5. logs/logger.py - Workflow Logger

**Purpose**: Comprehensive conversation and runtime logging.

**Architecture**:
```python
class WorkflowLogger:
    def __init__(self):
        # Creates 4 log files:
        # 1. session_TIMESTAMP.log (text)
        # 2. session_TIMESTAMP.json (JSON)
        # 3. runtime.log (unified runtime events)
        # 4. sessions.log (unified conversations)
        
    def log_conversation_turn(self, user_input, assistant_response):
        # Logs to session log AND sessions.log
        
    def log_booking_update(self, field, value):
        # Tracks booking data collection
        
    def log_confirmation_status(self, status):
        # Records confirmation/rejection
        
    def log_info(self, message):
        # Runtime info logging
        
    def log_warning(self, message):
        # Runtime warnings
        
    def log_error(self, message):
        # Runtime errors
        
    def log_session_end(self, booking_data):
        # Saves final session summary
```

**Log File Structure**:

1. **Session Logs** (`session_*.log`):
   - Conversation transcript
   - Booking updates
   - Confirmation status
   - Session summary

2. **Session JSON** (`session_*.json`):
   - Structured data
   - Timestamps (ISO format)
   - Complete conversation array
   - Booking updates array
   - Metadata

3. **Runtime Log** (`runtime.log`):
   - Component initialization
   - Processing events
   - Errors and warnings
   - Shutdown events

4. **Sessions Log** (`sessions.log`):
   - All conversations from all sessions
   - Unified view for analytics

**Log Entry Format** (runtime.log):
```
YYYY-MM-DD HH:MM:SS [LEVEL] [Session: SESSION_ID] Message
```

**No Additional Cost**: Logging is local, no API usage.

---

## Token Usage & Costs

### Summary Per Call (6-turn conversation)

| Component | Usage | Cost |
|-----------|-------|------|
| **STT (Deepgram)** | ~2 minutes audio | $0.025 |
| **LLM (OpenAI)** | ~4,235 tokens | $0.001 |
| **TTS (Deepgram)** | ~600 characters (6 responses) | $0.009 |
| **Logging** | Local files | $0.000 |
| **Total** | - | **$0.035** |

### Monthly Cost Estimates

Assuming 100 calls/day:

| Calls/Month | STT | LLM | TTS | Total/Month |
|-------------|-----|-----|-----|-------------|
| 3,000 | $75 | $3 | $27 | **$105** |
| 10,000 | $250 | $10 | $90 | **$350** |
| 30,000 | $750 | $30 | $270 | **$1,050** |

### Cost Optimization Tips

1. **STT**:
   - Use VAD to reduce dead air processing
   - Already optimized with real-time streaming

2. **LLM**:
   - ✅ History pruning (keeps last 10 exchanges)
   - ✅ Max tokens: 256
   - ✅ Using gpt-4o-mini (cheapest model)
   - Consider: Shorter system prompt (currently ~450 tokens)

3. **TTS**:
   - ✅ Only synthesizes assistant responses
   - ✅ Automatic cleanup of old audio files
   - Consider: Shorter responses (currently avg 100 chars)

---

## Detailed Code Explanation

### main.py Flow

```python
# 1. INITIALIZATION
def __init__(self):
    load_dotenv()  # Load API keys from .env
    
    # Initialize logger FIRST
    self.logger = WorkflowLogger()
    
    # Initialize components
    self.stt = SpeechToText()  # Deepgram STT
    self.tts = TextToSpeech()  # Deepgram TTS
    self.llm = LLMAgent(logger=self.logger)  # OpenAI with logger
    
    # State tracking
    self.current_transcript = []  # Accumulates partial transcripts
    self.segment_ready = False    # Flag for processing

# 2. CONVERSATION LOOP
def run(self):
    # Start STT with callbacks
    self.stt.start(
        on_transcript=self.on_transcript,  # Partial results
        on_final=self.on_final,            # Final results
        on_error=self.on_error             # Error handling
    )
    
    # Audio stream (16kHz, 16-bit, mono)
    with sd.InputStream(callback=self.stt.get_audio_callback(), ...):
        while True:
            if self.segment_ready:  # User finished speaking
                user_text = " ".join(self.current_transcript)
                self.process_user_input(user_text)
                self.current_transcript.clear()
                self.segment_ready = False

# 3. PROCESSING
def process_user_input(self, user_text):
    # LLM generates response
    response = self.llm.generate_response(user_text)
    
    # TTS synthesizes and plays
    audio_path = self.tts.synthesize(response, play=True)

# 4. SHUTDOWN
def shutdown(self):
    self.stt.stop()  # Stop audio stream
    booking_data = self.llm.get_booking_data()
    self.logger.log_session_end(booking_data.to_dict())  # Save logs
    self.tts.cleanup_old_files()  # Clean temp files
```

### logger.py Unified Logging

```python
class WorkflowLogger:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Session-specific logs
        self.text_log_path = f"logs/session_{self.session_id}.log"
        self.json_log_path = f"logs/session_{self.session_id}.json"

        # Unified logs
        self.runtime_log_path = "../storage/logs/runtime.log"
        self.sessions_log_path = "../storage/logs/sessions.log"

    def log_conversation_turn(self, user_input, assistant_response):
        # Log to session file
        with open(self.text_log_path, 'a') as f:
            f.write(f"[{timestamp}] USER: {user_input}\n")
            f.write(f"[{timestamp}] ASSISTANT: {assistant_response}\n")

        # Log to unified sessions.log
        with open(self.sessions_log_path, 'a') as f:
            f.write(f"[{timestamp}] [Session: {self.session_id}] USER: {user_input}\n")
            f.write(f"[{timestamp}] [Session: {self.session_id}] ASSISTANT: {assistant_response}\n")

        # Add to JSON data
        self.json_data["conversation"].append({
            "timestamp": timestamp.isoformat(),
            "user": user_input,
            "assistant": assistant_response
        })

    def log_info(self, message):
        # Runtime logging with session ID
        log_entry = f"{datetime.now()} [INFO] [Session: {self.session_id}] {message}\n"
        with open(self.runtime_log_path, 'a') as f:
            f.write(log_entry)
```

---

## Audio Recording System

### Current Implementation

Currently, the system saves:
- ✅ **Assistant audio** (TTS responses as MP3 files)
- ❌ **User audio** (NOT saved, only transcribed)

### Full Conversation Audio Recording

To save the complete conversation audio (both user and assistant), we need to:

1. **Record microphone input** to a WAV file
2. **Save assistant TTS responses** to WAV files
3. **Merge both** into a single conversation audio file

**Implementation**: See `audio_recorder.py` (created separately)

### Audio File Locations

```
audio_output/
├── session_2025-11-25_12-30-45/
│   ├── user_input.wav          # Raw microphone recording
│   ├── assistant_01.mp3        # TTS response 1
│   ├── assistant_02.mp3        # TTS response 2
│   ├── ...
│   └── full_conversation.wav   # Merged audio (user + assistant)
```

### Audio Quality Settings

- **Microphone**: 16kHz, 16-bit, mono (WAV)
- **TTS Output**: Variable bitrate MP3
- **Merged**: 16kHz, 16-bit, mono (WAV)

---

## Performance Characteristics

### Latency Breakdown (per turn)

| Component | Average Latency |
|-----------|----------------|
| User speaks | 2-5 seconds |
| STT transcription | ~300-500ms (real-time) |
| LLM processing | ~800-1500ms |
| TTS synthesis | ~500-800ms |
| Audio playback | 2-4 seconds |
| **Total response time** | **~4-6 seconds** |

### Resource Usage

| Component | CPU | Memory | Network |
|-----------|-----|--------|---------|
| STT (streaming) | Low | ~50MB | ~128 kbps upload |
| LLM | Minimal | ~10MB | ~1KB/request |
| TTS | Minimal | ~20MB | ~10KB/response |
| Logger | Minimal | ~5MB | None |
| **Total** | **Low** | **~100MB** | **~128 kbps** |

---

## Configuration Files

### .env
```bash
DEEPGRAM_API_KEY=your_deepgram_key
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini  # or gpt-4o for better quality
```

### prompt.py
- System prompt: ~450 tokens
- Call script outline
- Booking data structure

---

## Error Handling

### STT Errors
- Connection fallback (tries multiple models)
- Logged to `runtime.log`
- Callback: `on_error()`

### LLM Errors
- API timeout: 30 seconds
- Retry logic: None (returns error message)
- Logged to `runtime.log`

### TTS Errors
- Synthesis failure: Continues conversation
- Playback failure: Logged warning
- Audio file cleanup on error

---

## Monitoring & Debugging

### Real-Time Monitoring

```bash
# Watch runtime events
tail -f logs/runtime.log

# Watch all conversations
tail -f logs/sessions.log

# Watch current session
tail -f logs/session_$(date +%Y-%m-%d)*.log
```

### Debug Mode

Set in code:
```python
# In main.py
DEBUG = True  # Enables verbose logging
```

---

## Scalability Considerations

### Single Instance
- Handles 1 concurrent conversation
- ~100MB memory
- ~128 kbps upload bandwidth

### Multiple Instances
- Can run multiple processes
- Each handles 1 conversation
- Separate log files per session

### Production Deployment
- Use process manager (PM2, systemd)
- Implement queue for incoming calls
- Monitor log file sizes
- Implement log rotation

---

## Security Notes

1. **API Keys**: Stored in `.env` (gitignored)
2. **Logs**: May contain PII (customer names, locations)
3. **Audio**: Temporary files cleaned up
4. **Network**: All API calls over HTTPS

---

## Future Enhancements

1. **Full audio recording** (user + assistant) ✅ Implemented
2. **Better field extraction** (NER/LLM-based)
3. **Multi-language support** (Deepgram supports 36 languages)
4. **Webhook integration** (send booking data to backend)
5. **Real-time dashboard** (monitor active calls)
6. **Call analytics** (average duration, success rate, etc.)

---

**Last Updated**: 2025-11-25
**Version**: 1.0
