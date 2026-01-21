"""
Main orchestration module for the voice agent.
Coordinates STT, LLM, and TTS components to manage booking conversations.
"""

import os
import sys
import time
import sounddevice as sd
from dotenv import load_dotenv

from app.stt import SpeechToText
from app.tts import TextToSpeech
from app.llm import LLMAgent
from core.logger import WorkflowLogger
from core.audio_recorder import ConversationRecorder
from core.api_client import DropTruckAPIClient


class VoiceAgent:
    """Main voice agent orchestrator."""
    
    def __init__(self):
        """Initialize the voice agent with all components."""
        # Load environment variables
        load_dotenv()
        
        # Validate required API keys
        if not os.getenv("DEEPGRAM_API_KEY"):
            raise SystemExit(
                "❌ DEEPGRAM_API_KEY is not set.\n"
                "Please set it in your .env file or environment:\n"
                "  export DEEPGRAM_API_KEY=your_key_here\n"
            )
        
        # Initialize components
        print("🚀 Initializing DropTruck AI Sales Agent...")
        
        # Initialize logger with storage/logs directory
        self.logger = WorkflowLogger(logs_dir="storage/logs")
        self.logger.log_info("Voice Agent initialization started")

        # Initialize audio recorder
        self.audio_recorder = ConversationRecorder(self.logger.session_id)
        self.audio_recorder.start_recording()
        self.logger.log_info(f"Audio recording started: session_{self.logger.session_id}")

        # Initialize STT, TTS, and LLM with logger
        self.logger.log_info("Initializing Speech-to-Text component")
        self.stt = SpeechToText()

        self.logger.log_info("Initializing Text-to-Speech component")
        # Pass session directory to TTS for saving audio files
        session_audio_dir = f"storage/audio_output/session_{self.logger.session_id}"
        self.tts = TextToSpeech(output_dir=session_audio_dir)

        self.logger.log_info("Initializing LLM Agent component")
        self.llm = LLMAgent(logger=self.logger)
        
        # Conversation state
        self.current_transcript = []
        self.segment_ready = False
        self.audio_stream = None
        self.call_complete = False  # Flag to trigger shutdown
        
        self.logger.log_info("Voice Agent initialization complete")
        print("✅ DropTruck AI Sales Agent initialized")
        print(f"📼 Recording conversation to: audio_output/session_{self.logger.session_id}/")
    
    def on_final_transcript(self, transcript: str):
        """
        Handle final transcription from STT.
        Filters out noise, duplicates, and partial words.
        """
        if not transcript or not transcript.strip():
            return
        
        # Clean up transcript
        transcript = transcript.strip()
        
        # Filter out very short transcripts (likely noise)
        if len(transcript) < 3:
            print(f"⚠️  Filtered out noise: '{transcript}'")
            return
        
        # Filter out single-word duplicates
        if len(transcript.split()) == 1:
            # Check if this word was just said
            if self.current_transcript and transcript.lower() in ' '.join(self.current_transcript).lower():
                print(f"⚠️  Filtered duplicate: '{transcript}'")
                return
        
        # Add to current transcript buffer
        self.current_transcript.append(transcript)
        print(f"You: {transcript}")
        
        # Mark segment as ready for processing
        self.segment_ready = True
    
    def get_complete_transcript(self) -> str:
        """
        Get the complete transcript, removing duplicates and cleaning up.
        """
        if not self.current_transcript:
            return ""
        
        # Join all parts
        full_text = " ".join(self.current_transcript)
        
        # Remove consecutive duplicate words
        words = full_text.split()
        cleaned_words = []
        prev_word = None
        
        for word in words:
            # Skip if same as previous word (case-insensitive)
            if prev_word and word.lower() == prev_word.lower():
                continue
            cleaned_words.append(word)
            prev_word = word
        
        return " ".join(cleaned_words)

    def on_error(self, error):
        """Handle STT errors."""
        error_msg = f"STT Error: {error}"
        print(f"⚠️  {error_msg}")
        self.logger.log_error(error_msg)

    def process_user_input(self, user_text: str):
        """
        Process user input through LLM and generate TTS response.
        
        Args:
            user_text: Final transcribed user text
        """
        if not user_text.strip():
            return
        
        print(f"\n📝 User (final): {user_text}")
        self.logger.log_info(f"Processing user input: {user_text[:50]}...")

        # Save user audio segment for this turn
        self.audio_recorder.save_user_segment()

        # Generate LLM response
        assistant_response = self.llm.generate_response(user_text)
        print(f"💬 Assistant: {assistant_response}")
        
        # Synthesize and play response (with mic muting)
        audio_path = self.tts.synthesize(
            assistant_response, 
            play=True,
            audio_recorder=self.audio_recorder
        )
        
        # If audio was saved to file, record it for conversation merge
        if audio_path:
            self.audio_recorder.add_assistant_response(audio_path)

        self.logger.log_info("TTS synthesis and playback successful")
        print("🎵 Response complete\n")

        # Check if call should be completed
        if self.llm.is_call_complete():
            print("\n✅ Call completed - closing conversation")
            self.logger.log_info("Call auto-completed based on closing phrase")
            self.call_complete = True
    
    def run(self):
        """Run the main conversation loop."""
        try:
            # Start STT
            self.logger.log_info("Starting Speech-to-Text stream")
            self.stt.start(
                on_transcript=lambda text: None,  # Ignore partial transcripts
                on_final=self.on_final_transcript,  # Use our filtered method
                on_error=self.on_error
            )
            
            # Start audio stream
            print("\n" + "="*60)
            print("🎤 DROPTRUCK AI SALES AGENT READY")
            print("Press ENTER or Ctrl+C to end the call")
            print("="*60 + "\n")
            
            self.logger.log_info("Voice Agent ready - Conversation loop started")

            # Custom audio callback that records AND sends to STT
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print("Audio status:", status)
                
                # Record user audio for conversation merge (respects pause flag)
                self.audio_recorder.add_audio_chunk(indata.tobytes())
                
                # Send to STT (send silence during TTS to keep connection alive)
                try:
                    if self.stt.connection:
                        if self.audio_recorder.recording_paused:
                            # Send silence to prevent Deepgram timeout during TTS
                            silence = bytes(len(indata.tobytes()))
                            self.stt.connection.send_media(silence)
                        else:
                            # Send actual audio when not paused
                            self.stt.connection.send_media(indata.tobytes())
                except Exception as e:
                    print("Failed sending audio:", e)

            with sd.InputStream(
                callback=audio_callback,  # Use our custom callback
                channels=1,
                samplerate=16000,
                dtype='int16'
            ):
                # Wait 1 second for audio stream to stabilize
                time.sleep(1)
                
                # Trigger initial greeting from AI
                print("🤖 AI Agent starting conversation...\n")
                initial_greeting = "Hello, this is the DropTruck AI sales agent. I’m calling regarding your enquiry. May I know your name and mobile number to assist you better?"
                print(f"💬 Assistant: {initial_greeting}")
                
                # Synthesize and play greeting (with mic muting)
                audio_path = self.tts.synthesize(
                    initial_greeting, 
                    play=True,
                    audio_recorder=self.audio_recorder
                )
                if audio_path:
                    self.audio_recorder.add_assistant_response(audio_path)
                
                # Add greeting to LLM conversation history (so it doesn't repeat)
                # This tells the LLM it already said the greeting
                self.llm.conversation_history.append({
                    "role": "assistant",
                    "content": initial_greeting
                })
                
                # Log the greeting
                self.logger.log_info("Initial greeting delivered")
                print("🎵 Greeting complete - waiting for customer response...\n")
                
                # Main conversation loop
                while True:
                    # Check if call is complete
                    if self.call_complete:
                        print("\n📞 Call ending gracefully...")
                        break
                    
                    # Process accumulated transcript
                    if self.segment_ready and self.current_transcript:
                        # Get cleaned transcript (removes duplicates)
                        user_text = self.get_complete_transcript()
                        
                        if user_text:
                            self.process_user_input(user_text)
                        
                        # Reset for next segment
                        self.current_transcript = []
                        self.segment_ready = False
                    
                    # Small sleep to prevent busy waiting
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Call ended by user (Ctrl+C)")
            self.logger.log_info("Call ended by user (KeyboardInterrupt)")

        except Exception as e:
            error_msg = f"Runtime error: {e}"
            print(f"\n❌ {error_msg}")
            self.logger.log_error(error_msg)
            import traceback
            traceback.print_exc()
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all components."""
        print("\n🛑 Shutting down...")
        self.logger.log_info("Shutdown initiated")

        # Stop STT
        self.stt.stop()
        
        # Stop audio recording
        self.audio_recorder.stop_recording()
        
        # Small delay to ensure TTS playback completes and files are written
        print("⏳ Waiting for audio processing to complete...")
        time.sleep(2)
        
        # Print collected booking information
        self.print_booking_summary()
        
        # Save session logs
        booking_data = self.llm.get_booking_data()
        self.logger.log_session_end(booking_data.to_dict())
        
        # Merge conversation audio into single file
        print("\n🎬 Creating full conversation audio...")
        conversation_path = self.audio_recorder.merge_conversation()
        
        if conversation_path:
            print(f"✅ Full conversation audio saved: {conversation_path}")
            self.logger.log_info(f"Full conversation audio: {conversation_path}")
        
        # Get recording stats
        stats = self.audio_recorder.get_recording_stats()
        if stats.get('conversation_exists'):
            print(f"\n📊 Audio Recording Stats:")
            print(f"   Assistant responses recorded: {stats['assistant_responses']}")
            if stats.get('conversation_size_mb'):
                print(f"   Full conversation: {stats.get('conversation_size_mb', 0):.2f} MB")
        
        # Cleanup old audio files (files older than 1 hour in base audio_output dir)
        self.tts.cleanup_old_files()
        
        # Send booking data to API if confirmed
        if self.llm.booking_data.confirmation_status == "confirmed":
            print("\n📡 Sending booking to DropTruck API...")
            api_client = DropTruckAPIClient()
            success = api_client.send_booking(booking_data.to_dict())
            if success:
                self.logger.log_info("Booking data sent to API successfully")
            else:
                self.logger.log_warning("Failed to send booking data to API")
        else:
            print(f"\n⚠️  Booking not confirmed (status: {self.llm.booking_data.confirmation_status}), skipping API submission")
        
        self.logger.log_info("Shutdown complete")
        print("\n✅ Shutdown complete")
        print(f"\n📄 Logs saved to:")
        print(f"   Session (text): {self.logger.get_log_path()}")
        print(f"   Session (JSON): {self.logger.get_json_log_path()}")
        print(f"\n📝 Unified logs:")
        print(f"   Runtime log: {self.logger.get_runtime_log_path()}")
        print(f"   Sessions log: {self.logger.get_sessions_log_path()}")
        print(f"\n💡 To tail runtime logs, run:")
        print(f"   tail -f {self.logger.get_runtime_log_path()}")
        print(f"\n💡 To tail sessions logs, run:")
        print(f"   tail -f {self.logger.get_sessions_log_path()}")

    def print_booking_summary(self):
        """Print the collected booking information after call ends."""
        booking_data = self.llm.get_booking_data()
        print(booking_data)
        
        # Additional conversation stats
        print(self.llm.get_conversation_summary())


def main():
    """Entry point for the voice agent."""
    try:
        agent = VoiceAgent()
        agent.run()
    except Exception as e:
        print(f"\n❌ Failed to start voice agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
