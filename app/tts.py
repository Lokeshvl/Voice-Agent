"""
Text-to-Speech module using Deepgram.
Converts assistant responses to natural-sounding speech.
"""

import os
import time
import platform
from deepgram import DeepgramClient


class TextToSpeech:
    """Manages text-to-speech synthesis using Deepgram."""
    
    def __init__(self, api_key: str = None, output_dir: str = "audio_output", model: str = "aura-2-selene-en"):
        """
        Initialize TTS with Deepgram API key.
        
        Args:
            api_key: Deepgram API key (reads from env if not provided)
            output_dir: Directory to save audio files
            model: Deepgram TTS model to use
        """
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY is required. Set it in environment or .env file."
            )
        
        self.client = DeepgramClient(api_key=self.api_key)
        self.model = model
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def synthesize(self, text: str, play: bool = True, audio_recorder=None) -> str:
        """
        Synthesize text to speech and optionally play it.
        
        Args:
            text: Text to synthesize
            play: Whether to play the audio after synthesis
            audio_recorder: Optional ConversationRecorder to pause during playback
            
        Returns:
            Path to saved audio file, or empty string on failure
        """
        if not text or not text.strip():
            print("⚠️  Empty text provided, skipping TTS")
            return ""
        
        # Generate unique filename
        timestamp = int(time.time() * 1000)
        filename = f"assistant_reply_{timestamp}.mp3"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            print("🔊 Synthesizing speech...")
            
            # Generate speech using Deepgram
            speak_stream = self.client.speak.v1.audio.generate(
                text=text,
                model=self.model,
            )
            
            # Write audio to file
            with open(filepath, "wb") as f:
                for chunk in speak_stream:
                    f.write(chunk)
            
            print(f"✅ Audio saved to: {filepath}")
            
            # Play audio if requested
            if play:
                # Pause mic during playback to prevent echo
                if audio_recorder:
                    audio_recorder.pause_recording()
                
                self._play_audio(filepath)
                
                # Resume mic after playback
                if audio_recorder:
                    audio_recorder.resume_recording()
            
            return filepath
            
        except Exception as e:
            print(f"❌ TTS synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _play_audio(self, filepath: str):
        """
        Play audio file using platform-specific command.
        
        Args:
            filepath: Path to audio file to play
        """
        if not os.path.exists(filepath):
            print(f"⚠️  Audio file not found: {filepath}")
            return
        
        try:
            print("▶️  Playing audio...")
            
            system = platform.system()
            if system == "Darwin":  # macOS
                os.system(f"afplay '{filepath}'")
            elif system == "Linux":
                # Try common Linux audio players
                if os.system("which aplay > /dev/null 2>&1") == 0:
                    os.system(f"aplay '{filepath}'")
                elif os.system("which mpg123 > /dev/null 2>&1") == 0:
                    os.system(f"mpg123 '{filepath}'")
                else:
                    print("⚠️  No audio player found (aplay or mpg123)")
            elif system == "Windows":
                os.system(f"start {filepath}")
            else:
                print(f"⚠️  Unsupported platform for audio playback: {system}")
                return
            
            print("✅ Playback complete")
            
        except Exception as e:
            print(f"❌ Audio playback failed: {e}")
    
    def save_conversation_audio(self, output_path: str = None) -> str:
        """
        Save all audio responses to a single merged audio file.

        Args:
            output_path: Path to save the merged audio (optional)

        Returns:
            Path to the saved audio file
        """
        if not self.audio_responses:
            print("⚠️  No audio responses to save")
            return ""

        try:
            from pydub import AudioSegment

            # If no output path specified, create one in the output directory
            if not output_path:
                os.makedirs(self.output_dir, exist_ok=True)
                timestamp = int(time.time() * 1000)
                output_path = os.path.join(self.output_dir, f"conversation_{timestamp}.mp3")

            print(f"💾 Saving conversation audio ({len(self.audio_responses)} responses)...")

            # Merge all audio responses with silence between them
            conversation = AudioSegment.silent(duration=500)  # Start with 500ms silence

            for i, audio_data in enumerate(self.audio_responses):
                # Save audio data to temporary file to load with pydub
                import tempfile
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.mp3', delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_path = tmp_file.name

                # Load and add to conversation
                audio_segment = AudioSegment.from_mp3(tmp_path)
                if i > 0:
                    conversation += AudioSegment.silent(duration=300)  # 300ms pause between responses
                conversation += audio_segment

                # Clean up temp file
                try:
                    os.remove(tmp_path)
                except:
                    pass

            # Add final silence
            conversation += AudioSegment.silent(duration=500)

            # Export the merged audio
            conversation.export(output_path, format="mp3")

            print(f"✅ Conversation audio saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Failed to save conversation audio: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def cleanup_old_files(self, max_age_seconds: int = 3600):
        """
        Remove old audio files from output directory.
        
        Args:
            max_age_seconds: Maximum age of files to keep (default: 1 hour)
        """
        try:
            current_time = time.time()
            for filename in os.listdir(self.output_dir):
                filepath = os.path.join(self.output_dir, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > max_age_seconds:
                        os.remove(filepath)
                        print(f"🗑️  Removed old audio file: {filename}")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")
