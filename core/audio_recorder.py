"""
Audio Recording Module for Voice Agent.
Records full conversation audio (both user input and assistant responses).
"""

import os
import wave
import threading
import numpy as np
from datetime import datetime
from pydub import AudioSegment


class ConversationRecorder:
    """Records complete conversation audio including user and assistant."""
    
    def __init__(self, session_id: str, output_dir: str = "storage/audio_output"):
        """
        Initialize conversation recorder.
        
        Args:
            session_id: Unique session identifier
            output_dir: Base directory for audio files
        """
        self.session_id = session_id
        self.session_dir = os.path.join(output_dir, f"session_{session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Audio buffers
        self.user_audio_buffer = []
        self.recording_active = False
        self.recording_paused = False  # For muting mic during TTS
        self.lock = threading.Lock()
        
        # File paths
        self.user_audio_path = os.path.join(self.session_dir, "user_input.wav")
        self.conversation_path = os.path.join(self.session_dir, "full_conversation.wav")
        
        # Track audio segments for chronological merging
        self.user_segments = []  # List of user audio file paths
        self.assistant_responses = []  # List of assistant audio file paths
        self.current_turn = 0  # Track conversation turns
        
        # Audio settings (must match STT settings)
        self.sample_rate = 16000
        self.channels = 1
        self.sample_width = 2  # 16-bit = 2 bytes
        
        print(f"📼 Audio recording to: {self.session_dir}")
    
    def start_recording(self):
        """Start recording user audio."""
        with self.lock:
            self.recording_active = True
            self.user_audio_buffer.clear()
        print("🔴 Recording started")
    
    def add_audio_chunk(self, audio_data: bytes):
        """
        Add audio chunk from microphone to buffer.
        This should be called from the sounddevice callback.
        Only captures audio if recording is active AND not paused.
        
        Args:
            audio_data: Raw audio bytes from microphone
        """
        if self.recording_active and not self.recording_paused:
            with self.lock:
                self.user_audio_buffer.append(audio_data)
    
    def stop_recording(self):
        """Stop recording and save user audio to file."""
        with self.lock:
            self.recording_active = False
        
        if self.user_audio_buffer:
            self._save_user_audio()
            print(f"✅ User audio saved: {self.user_audio_path}")
    
    def pause_recording(self):
        """
        Pause audio capture temporarily (e.g., during TTS playback).
        Prevents echo and feedback by muting the mic.
        """
        with self.lock:
            self.recording_paused = True
        print("🔇 Mic muted (TTS playing)")
    
    def resume_recording(self):
        """Resume audio capture after pause."""
        with self.lock:
            self.recording_paused = False
        print("🎤 Mic unmuted (ready for input)")
    
    def _save_user_audio(self):
        """Save accumulated user audio to WAV file."""
        if not self.user_audio_buffer:
            return
        
        # Combine all audio chunks
        audio_data = b''.join(self.user_audio_buffer)
        
        # Write to WAV file
        with wave.open(self.user_audio_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
    
    def add_assistant_response(self, audio_file_path: str):
        """
        Record path of assistant's TTS audio file.
        
        Args:
            audio_file_path: Path to assistant's MP3 audio file
        """
        self.assistant_responses.append(audio_file_path)
        print(f"📝 Assistant response {len(self.assistant_responses)} recorded")
    
    def save_user_segment(self):
        """Save current user audio buffer as a segment for this turn."""
        if not self.user_audio_buffer:
            return None
        
        # Save this turn's audio
        segment_path = os.path.join(self.session_dir, f"user_turn_{self.current_turn}.wav")
        
        # Combine audio chunks for this turn
        audio_data = b''.join(self.user_audio_buffer)
        
        # Write to WAV file
        with wave.open(segment_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        
        self.user_segments.append(segment_path)
        self.user_audio_buffer.clear()  # Clear for next turn
        self.current_turn += 1
        
        return segment_path
    
    def merge_conversation(self):
        """
        Merge user audio segments and assistant responses in chronological order.
        Format: Assistant greeting → User 1 → Assistant 1 → User 2 → Assistant 2...
        Automatically cleans up intermediate files after merging.
        """
        print("🎬 Merging conversation audio...")
        print(f"   User segments: {len(self.user_segments)}")
        print(f"   Assistant responses: {len(self.assistant_responses)}")
        
        try:
            # Start with silence
            conversation = AudioSegment.silent(duration=500)  # 500ms silence
            
            # Interleave assistant and user audio
            # Pattern: Assistant greeting, User 1, Assistant 1, User 2, Assistant 2, etc.
            max_turns = max(len(self.assistant_responses), len(self.user_segments))
            
            for i in range(max_turns):
                # Add assistant response if available
                if i < len(self.assistant_responses):
                    response_path = self.assistant_responses[i]
                    print(f"   Checking assistant {i+1}: {response_path}")
                    if os.path.exists(response_path):
                        file_size = os.path.getsize(response_path)
                        print(f"   ✓ File exists ({file_size} bytes)")
                        conversation += AudioSegment.silent(duration=300)  # 300ms pause
                        assistant_audio = AudioSegment.from_mp3(response_path)
                        conversation += assistant_audio
                        print(f"   ✓ Merged assistant response {i+1} ({len(assistant_audio)/1000:.1f}s)")
                    else:
                        print(f"   ✗ File NOT FOUND: {response_path}")
                
                # Add user segment if available
                if i < len(self.user_segments):
                    segment_path = self.user_segments[i]
                    print(f"   Checking user {i+1}: {segment_path}")
                    if os.path.exists(segment_path):
                        conversation += AudioSegment.silent(duration=300)  # 300ms pause
                        user_audio = AudioSegment.from_wav(segment_path)
                        conversation += user_audio
                        print(f"   ✓ Merged user segment {i+1} ({len(user_audio)/1000:.1f}s)")
                    else:
                        print(f"   ✗ File NOT FOUND: {segment_path}")
            
            # Add final silence
            conversation += AudioSegment.silent(duration=500)
            
            # Export as WAV
            conversation.export(
                self.conversation_path,
                format="wav",
                parameters=["-ar", str(self.sample_rate)]
            )
            
            print(f"✅ Full conversation saved: {self.conversation_path}")
            print(f"   Total duration: {len(conversation)/1000:.1f} seconds")
            print(f"   User audio + {len(self.assistant_responses)} AI responses merged")
            
            # Clean up intermediate files
            self._cleanup_intermediate_files()
            
            return self.conversation_path
            
        except Exception as e:
            print(f"❌ Error merging conversation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _cleanup_intermediate_files(self):
        """
        Delete intermediate files (user_input.wav and assistant MP3s).
        Keeps only the final merged conversation.
        """
        try:
            # Delete user audio file
            if os.path.exists(self.user_audio_path):
                os.remove(self.user_audio_path)
                print(f"🗑️  Removed: {os.path.basename(self.user_audio_path)}")
            
            # Delete all assistant response files
            for response_path in self.assistant_responses:
                if os.path.exists(response_path):
                    os.remove(response_path)
                    print(f"🗑️  Removed: {os.path.basename(response_path)}")
            
            print("✅ Intermediate audio files cleaned up")
            
        except Exception as e:
            print(f"⚠️  Error cleaning up intermediate files: {e}")
    
    def get_recording_stats(self):
        """Get statistics about recorded audio."""
        stats = {
            "session_id": self.session_id,
            "user_audio_exists": os.path.exists(self.user_audio_path),
            "assistant_responses": len(self.assistant_responses),
            "conversation_exists": os.path.exists(self.conversation_path)
        }
        
        # Get file sizes
        if stats["user_audio_exists"]:
            stats["user_audio_size_mb"] = os.path.getsize(self.user_audio_path) / (1024 * 1024)
        
        if stats["conversation_exists"]:
            stats["conversation_size_mb"] = os.path.getsize(self.conversation_path) / (1024 * 1024)
        
        return stats


# Integration helper functions

def create_recorder(session_id: str) -> ConversationRecorder:
    """
    Factory function to create a conversation recorder.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        ConversationRecorder instance
    """
    return ConversationRecorder(session_id)


def audio_callback_with_recording(recorder: ConversationRecorder, original_callback):
    """
    Wraps the original audio callback to include recording.
    
    Args:
        recorder: ConversationRecorder instance
        original_callback: Original STT audio callback
        
    Returns:
        Wrapped callback function
    """
    def wrapped_callback(indata, frames, time_info, status):
        # Record audio
        recorder.add_audio_chunk(indata.tobytes())
        
        # Call original STT callback
        original_callback(indata, frames, time_info, status)
    
    return wrapped_callback
