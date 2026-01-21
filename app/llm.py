"""
LLM integration module for the voice agent.
Manages conversation with OpenAI API and extracts booking information.
"""

import os
import requests
from openai import OpenAI
from core.prompt import SYSTEM_PROMPT, BookingData


class LLMAgent:
    """Manages LLM interactions and conversation state."""
    
    def __init__(self, api_key: str = None, model: str = None, logger=None):
        """
        Initialize LLM agent.
        
        Args:
            api_key: OpenAI API key (reads from env if not provided)
            model: OpenAI model to use (default: gpt-4o-mini)
            logger: WorkflowLogger instance for logging conversations
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.logger = logger
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            print("⚠️  WARNING: OPENAI_API_KEY not set. Agent will use echo mode.")
        
        self.conversation_history = []
        self.booking_data = BookingData()
        
        # Initialize vehicle keywords with hardcoded defaults
        self.vehicle_keywords = {
            # Basic trucks
            "tata ace": "Tata Ace",
            "tata ac": "Tata Ace",
            "ace": "Tata Ace",
            "dost": "Dost",
            "bada dost": "Bada Dost",
            "bolero": "Bolero",
            "bolero pickup": "Bolero",
            "407": "407",
            "eicher": "Eicher",
            "ashok leyland": "Ashok Leyland",
            
            # Feet-based trucks
            "12 feet": "12 Feet",
            "14 feet": "14 Feet",
            "17 feet": "17 Feet",
            "19 feet": "19 Feet",
            "20 feet": "20 Feet",
            "22 feet": "22 Feet",
            "24 feet": "24 Feet",
            "32 feet": "32 Feet Multi-Axle",
            "32 feet multi-axle": "32 Feet Multi-Axle",
            "32 feet multi axle": "32 Feet Multi-Axle",
            
            # Trailers
            "trailer": "Trailer",
            "20 feet trailer": "20 Feet Trailer",
            "24 feet trailer": "24 Feet Trailer",
            "40 feet trailer": "40 Feet Trailer",
            "low-bed": "Low-Bed Trailer",
            "low bed": "Low-Bed Trailer",
            "semi-bed": "Semi-Bed Trailer",
            "semi bed": "Semi-Bed Trailer",
            "high-bed": "High-Bed Trailer",
            "high bed": "High-Bed Trailer",
            
            # Wheel configurations
            "6-wheel": "6-Wheel Truck",
            "6 wheel": "6-Wheel Truck",
            "10-wheel": "10-Wheel Truck",
            "10 wheel": "10-Wheel Truck",
            "12-wheel": "12-Wheel Truck",
            "12 wheel": "12-Wheel Truck",
            "14-wheel": "14-Wheel Truck",
            "14 wheel": "14-Wheel Truck",
            "16-wheel": "16-Wheel Truck",
            "16 wheel": "16-Wheel Truck",
            
            # Special types
            "car-carrier": "Car-Carrier",
            "car carrier": "Car-Carrier",
            "part-load": "Part-Load",
            "part load": "Part-Load",
        }
        
        # Initialize body types
        self.body_types = {"open": "Open", "container": "Container"}

        # Fetch valid types from DB if available
        try:
            from core.db_client import DBClient
            db = DBClient()
            
            # Fetch trucks
            db_trucks = db.get_truck_types()
            if db_trucks:
                for truck in db_trucks:
                    name = truck['name']
                    self.vehicle_keywords[name.lower()] = name
                    self.vehicle_keywords[name.lower().replace('-', ' ')] = name
                if self.logger:
                    self.logger.log_info(f"Loaded {len(db_trucks)} truck types from database")
            
            # Fetch bodies
            db_bodies = db.get_body_types()
            if db_bodies:
                for body in db_bodies:
                    name = body['name']
                    self.body_types[name.lower()] = name
                    
        except Exception as e:
            print(f"⚠️ Failed to load types from DB: {e}")
        
        # Initialize conversation with system prompt
        if self.api_key:
            self.conversation_history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
    
    def generate_response(self, user_text: str) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_text: User's spoken text
            
        Returns:
            Assistant's response text
        """
        if not user_text or not user_text.strip():
            return "I didn't catch that. Could you please repeat?"
        
        # Extract booking information from user text
        self._extract_booking_info(user_text)
        
        # Detect confirmation status
        self._detect_confirmation(user_text)
        
        # If no API key, use echo mode
        if not self.api_key:
            response = f"[Echo Mode] You said: {user_text}"
            if self.logger:
                self.logger.log_conversation_turn(user_text, response)
            return response
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
        # Keep only recent messages to control token usage
        recent_messages = self._get_recent_messages()
        
        # Call OpenAI API
        try:
            response_text = self._call_openai(recent_messages)
            
            # Extract from LLM response (captures locations from confirmations)
            self._extract_booking_info(response_text)
            
            # Also extract name/contact from confirmation format
            # Pattern: "Name X, mobile Y, pickup P..."
            self._extract_from_confirmation(response_text)
            
            # Check if response contains BOOKING_CONFIRMED marker
            if self.check_booking_confirmed_marker(response_text):
                self.booking_data.confirmation_status = "confirmed"
                if self.logger:
                    self.logger.log_confirmation_status("confirmed")
                    self.logger.log_info("BOOKING_CONFIRMED marker detected in LLM response")
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Log conversation turn
            if self.logger:
                self.logger.log_conversation_turn(user_text, response_text)
            
            return response_text
            
        except Exception as e:
            print(f"❌ LLM error: {e}")
            error_response = "I'm having trouble processing that right now. Could you try again?"
            if self.logger:
                self.logger.log_conversation_turn(user_text, error_response)
            return error_response
    
    def is_call_complete(self) -> bool:
        """
        Check if the call should be ended based on assistant's last response.
        Returns True if the assistant has said goodbye or confirmed booking.
        """
        if not self.conversation_history:
            return False
        
        # Get the last assistant message
        last_message = None
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                last_message = msg["content"].lower()
                break
        
        if not last_message:
            return False
        
        # Check for BOOKING_CONFIRMED marker (highest priority)
        if "booking_confirmed" in last_message:
            return True
        
        # Check for closing phrases
        closing_phrases = [
            "have a great day",
            "have a good day", 
            "thank you for your time",
            "goodbye",
            "bye",
            "our sales person will contact you soon",
            "you can contact droptruck anytime"
        ]
        
        return any(phrase in last_message for phrase in closing_phrases)
    
    def _call_openai(self, messages: list) -> str:
        """
        Call OpenAI API with conversation messages.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Assistant's response text
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 256
        }
        
        print("🤖 Calling LLM...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"OpenAI API error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            raise Exception(f"OpenAI API returned status {response.status_code}")
        
        data = response.json()
        return data['choices'][0]['message']['content']
    
    def _get_recent_messages(self, max_exchanges: int = 10) -> list:
        """
        Get recent conversation messages to control token usage.
        Args:
            max_exchanges: Maximum number of user-assistant exchanges to keep
        Returns:
            List of recent messages including system prompt
        """
        # Always include system prompt
        system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
        
        # Get recent user/assistant messages
        other_messages = [msg for msg in self.conversation_history if msg["role"] != "system"]
        recent_other = other_messages[-(max_exchanges * 2):]
        
        return system_messages + recent_other
    
    def _extract_from_confirmation(self, text: str):
        """
        Extract booking info from AI confirmation messages.
        Pattern: "Name [name], mobile [number], pickup [p], drop [d]..."
        """
        import re
        text_lower = text.lower()
        
        # Extract name from confirmation
        if not self.booking_data.customer_name:
            name_match = re.search(r'name\s+([a-zA-Z\s]+?)(?:,|\s+mobile)', text_lower)
            if name_match:
                name = name_match.group(1).strip().title()
                # Filter out city names
                if name.lower() not in ['chennai', 'bangalore', 'mumbai', 'delhi', 'pune', 'hyderabad', 'goa', 'kolkata']:
                    self.booking_data.customer_name = name
                    if self.logger:
                        self.logger.log_booking_update("customer_name", name)
        
        # Extract phone from confirmation
        if not self.booking_data.contact:
            # Match patterns like "mobile 9066542031" or "mobile (906) 654-2031"
            phone_patterns = [
                r'mobile\s+([6-9]\d{9})',  # mobile 9066542031
                r'mobile\s+\((\d{3})\)\s*(\d{3})-(\d{4})',  # mobile (906) 654-2031
                r'number\s+([6-9]\d{9})',  # number 9066542031
            ]
            
            for pattern in phone_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    if len(match.groups()) == 1:
                        phone = match.group(1)
                    else:
                        # Formatted number like (906) 654-2031
                        phone = ''.join(match.groups())
                    
                    if len(phone) == 10:
                        self.booking_data.contact = phone
                        if self.logger:
                            self.logger.log_booking_update("contact", phone)
                        break
    
    def _extract_booking_info(self, text: str):
        """
        Extract booking information from user text using pattern matching.
        This is a simple extraction - the LLM will handle the conversation flow.
        
        Args:
            text: User's spoken text
        """
        text_lower = text.lower()
        import re  # Import at the top of method
        
        # Extract customer name
        # Patterns: "my name is X", "I am X", "this is X", "name X"
        if not self.booking_data.customer_name:
            name_patterns = [
                r'(?:my name is|i am|this is|name is|i\'m)\s+([a-zA-Z\s]+?)(?:\s+and|\s+my|\.|,|$)',
                r'name\s+([a-zA-Z\s]+?)(?:\s+and|\s+my|\.|,|$)',
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    name = match.group(1).strip().title()
                    # Enhanced stop words list
                    stop_words = ['is', 'and', 'my', 'the', 'from', 'to', 'for', 'i', 'said', 'please', 'you', 'your']
                    name_parts = [word for word in name.split() if word.lower() not in stop_words]
                    # Only accept if we have at least one word and it's not too short
                    if name_parts and len(' '.join(name_parts)) >= 2:
                        clean_name = ' '.join(name_parts)
                        # Don't accept if it's a city name or common word
                        if clean_name.lower() not in ['chennai', 'bangalore', 'mumbai', 'delhi', 'pune', 'hyderabad']:
                            self.booking_data.customer_name = clean_name
                            if self.logger:
                                self.logger.log_booking_update("customer_name", self.booking_data.customer_name)
                            break
        
        # Extract phone number
        # Patterns: 10-digit Indian mobile numbers
        if not self.booking_data.contact:
            # Helper function to convert words to digits
            def words_to_digits(text_str):
                """Convert spoken numbers to digits (e.g., 'nine six' -> '96')"""
                word_to_num = {
                    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                    'double': '', 'triple': ''  # Handle 'double zero' etc
                }
                
                words = text_str.lower().split()
                digits = []
                i = 0
                while i < len(words):
                    word = words[i]
                    if word in word_to_num:
                        digit = word_to_num[word]
                        # Handle 'double X' or 'triple X'
                        if word == 'double' and i + 1 < len(words) and words[i + 1] in word_to_num:
                            digit = word_to_num[words[i + 1]] * 2
                            i += 1
                        elif word == 'triple' and i + 1 < len(words) and words[i + 1] in word_to_num:
                            digit = word_to_num[words[i + 1]] * 3
                            i += 1
                        digits.append(digit)
                    i += 1
                return ''.join(digits)
            
            # Try to extract from spoken words first
            spoken_number = words_to_digits(text)
            if spoken_number and len(spoken_number) == 10 and spoken_number[0] in '6789':
                self.booking_data.contact = spoken_number
                if self.logger:
                    self.logger.log_booking_update("contact", spoken_number)
            else:
                # Match 10-digit numbers (with or without spaces/dashes)
                phone_patterns = [
                    r'\b([6-9]\d{9})\b',  # 10 digits starting with 6-9
                    r'\b([6-9]\d[\s-]?\d{3}[\s-]?\d{5})\b',  # With spaces/dashes
                    r'(?:number|mobile|phone|contact)[\s:]+([6-9]\d{9})',  # After keywords
                ]
                
                for pattern in phone_patterns:
                    match = re.search(pattern, text)  # Use original text for numbers
                    if match:
                        phone = match.group(1).replace(' ', '').replace('-', '')
                        if len(phone) == 10:
                            self.booking_data.contact = phone
                            if self.logger:
                                self.logger.log_booking_update("contact", phone)
                            break
        
        # Extract pickup and drop locations
        # Pattern: "from X to Y" or "X to Y" or "trip from X to Y"
        import re
        
        # Try to find "from X to Y" pattern (more flexible)
        # Allow updates if user provides clearer information
        from_to_pattern = r'(?:from|pickup|trip from)\s+([a-zA-Z\s]+?)\s+(?:to|drop)\s+([a-zA-Z\s]+?)(?:\s|,|$|\.|\band\b)'
        match = re.search(from_to_pattern, text_lower)
        if match:
            pickup = match.group(1).strip().title()
            drop = match.group(2).strip().title()
            
            # Update if field is empty OR if new value is longer/clearer (likely a correction)
            if not self.booking_data.pickup_location or len(pickup) > len(self.booking_data.pickup_location or ""):
                self.booking_data.pickup_location = pickup
                if self.logger:
                    self.logger.log_booking_update("pickup_location", pickup)
            
            if not self.booking_data.drop_location or len(drop) > len(self.booking_data.drop_location or ""):
                self.booking_data.drop_location = drop
                if self.logger:
                    self.logger.log_booking_update("drop_location", drop)
        
        # Also try confirmation format: "Pickup X, drop Y" or "Pickup in X, drop in Y"
        # Updated regex to handle "in" optionally and capture the city name
        confirmation_pattern = r'pickup\s+(?:in\s+)?([a-zA-Z\s]+?),\s*drop\s+(?:in\s+)?([a-zA-Z\s]+?)(?:,|truck|\s+truck|\.|$)'
        conf_match = re.search(confirmation_pattern, text_lower)
        if conf_match:
            pickup = conf_match.group(1).strip().title()
            drop = conf_match.group(2).strip().title()
            
            # Clean up common prefixes/suffixes (case-insensitive)
            for prefix in ['in ', 'from ', 'at ']:
                if pickup.lower().startswith(prefix):
                    pickup = pickup[len(prefix):].title()
                if drop.lower().startswith(prefix):
                    drop = drop[len(prefix):].title()
            
            # Always update from confirmation (it's the AI's understanding)
            if pickup and len(pickup) > 2:
                self.booking_data.pickup_location = pickup
                if self.logger:
                    self.logger.log_booking_update("pickup_location", pickup)
            
            if drop and len(drop) > 2:
                self.booking_data.drop_location = drop
                if self.logger:
                    self.logger.log_booking_update("drop_location", drop)
        
        # Detect body type
        # Fetch valid body types from DB
        body_types = {"open": "Open", "container": "Container"} # Default fallback
        try:
            from core.db_client import DBClient
            db = DBClient()
            db_bodies = db.get_body_types()
            if db_bodies:
                for body in db_bodies:
                    name = body['name']
                    body_types[name.lower()] = name
        except Exception:
            pass

        if self.booking_data.body_type is None:
            for keyword, name in body_types.items():
                if keyword in text_lower:
                    self.booking_data.body_type = name
                    if self.logger:
                        self.logger.log_booking_update("body_type", name)
                    break
        
        # Detect vehicle type mentions with fuzzy matching
        # This handles mispronunciations and different accents
        



        if not self.booking_data.vehicle_type:
            from fuzzywuzzy import fuzz
            
            # Try fuzzy matching for each word/phrase in user text
            best_match = None
            best_score = 0
            best_vehicle = None
            
            # Split text into words and create n-grams (1-4 words)
            words = text_lower.split()
            for i in range(len(words)):
                for j in range(i + 1, min(i + 5, len(words) + 1)):  # Check up to 4-word phrases
                    phrase = " ".join(words[i:j])
                    
                    # Skip very short phrases unless they are specific known types (like "ac", "407")
                    if len(phrase) < 3 and phrase not in ["ac", "407"]:
                        continue
                    
                    # Check against all vehicle keywords
                    for keyword, vehicle_name in self.vehicle_keywords.items():
                        # Use partial ratio for better matching
                        score = fuzz.partial_ratio(phrase, keyword)
                        
                        # Debug logging removed for cleaner output
                        # if score > 50:
                        #     print(f"DEBUG: '{phrase}' vs '{keyword}' = {score} -> {vehicle_name}")
                        
                        # If score is high enough and better than previous matches
                        if score > best_score and score >= 85:  # Increased to 85% for better accuracy
                            best_score = score
                            best_match = keyword
            
            # If we found a good match, use it (allow updates if confidence is higher)
            if best_vehicle and best_score >= 85:
                self.booking_data.vehicle_type = best_vehicle
                if self.logger:
                    self.logger.log_booking_update("vehicle_type", best_vehicle)
                    self.logger.log_info(f"Fuzzy matched '{best_match}' with {best_score}% confidence")
        
        # Also try to extract from confirmation: "truck type X"
        # Updated regex to be more robust
        truck_type_pattern = r'truck\s+(?:type\s+)?([a-zA-Z0-9\s]+?)(?:,|body|\s+open|\s+container|\.|$)'
        truck_match = re.search(truck_type_pattern, text_lower)
        if truck_match:
            truck_mentioned = truck_match.group(1).strip()
            # Try fuzzy match on this
            from fuzzywuzzy import fuzz
            best_score = 0
            best_vehicle = None
            
            for keyword, vehicle_name in self.vehicle_keywords.items():
                # Use ratio for short strings, partial_ratio for longer
                score = fuzz.ratio(truck_mentioned.lower(), keyword)
                if score > best_score and score >= 65:  # Lower threshold for explicit confirmation mentions
                    best_score = score
                    best_vehicle = vehicle_name
            
            if best_vehicle:
                self.booking_data.vehicle_type = best_vehicle
                if self.logger:
                    self.logger.log_booking_update("vehicle_type", best_vehicle)
                    self.logger.log_info(f"Extracted from confirmation: '{truck_mentioned}' → {best_vehicle}")
        
        # Fallback: Check for feet sizes using regex
        if not self.booking_data.vehicle_type:
            feet_pattern = r'(\d+)\s*(?:feet|ft|foot)'
            feet_match = re.search(feet_pattern, text_lower)
            if feet_match:
                feet = feet_match.group(1)
                self.booking_data.vehicle_type = f"{feet} Feet"
                if self.logger:
                    self.logger.log_booking_update("vehicle_type", self.booking_data.vehicle_type)
            elif "truck" in text_lower:
                self.booking_data.vehicle_type = "Truck"
                if self.logger:
                    self.logger.log_booking_update("vehicle_type", "Truck")
        
        # Detect material/goods type using flexible pattern matching
        # This captures whatever the customer says instead of limiting to a predefined list
        if not self.booking_data.goods_type:
            # Pattern 1: "carrying X", "transporting X", "moving X"
            material_patterns = [
                r'carrying\s+([a-zA-Z0-9\s]+?)(?:\s+from|\s+to|,|\.|$)',
                r'transporting\s+([a-zA-Z0-9\s]+?)(?:\s+from|\s+to|,|\.|$)',
                r'moving\s+([a-zA-Z0-9\s]+?)(?:\s+from|\s+to|,|\.|$)',
                r'material\s+(?:is\s+)?([a-zA-Z0-9\s]+?)(?:\s+from|\s+to|,|\.|$)',
                r'goods\s+(?:is\s+)?([a-zA-Z0-9\s]+?)(?:\s+from|\s+to|,|\.|$)',
                r'load\s+(?:is\s+)?([a-zA-Z0-9\s]+?)(?:\s+from|\s+to|,|\.|$)',
            ]
            
            for pattern in material_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    material = match.group(1).strip().title()
                    # Clean up common words
                    material = material.replace(' And ', ' and ')
                    if len(material) > 2:  # Avoid single letters
                        self.booking_data.goods_type = material
                        if self.logger:
                            self.logger.log_booking_update("goods_type", material)
                        break
        
        # Also check confirmation format: "material X"
        if not self.booking_data.goods_type:
            conf_material_pattern = r'material\s+([a-zA-Z0-9\s]+?)(?:,|date|\.|$)'
            conf_match = re.search(conf_material_pattern, text_lower)
            if conf_match:
                material = conf_match.group(1).strip().title()
                if len(material) > 2:
                    self.booking_data.goods_type = material
                    if self.logger:
                        self.logger.log_booking_update("goods_type", material)
        
        # Detect trip date and convert to YYYY-MM-DD format
        from datetime import datetime, timedelta
        
        if not self.booking_data.trip_date:
            if "today" in text_lower or "now" in text_lower:
                date_obj = datetime.now()
                self.booking_data.trip_date = date_obj.strftime("%Y-%m-%d")
                if self.logger:
                    self.logger.log_booking_update("trip_date", self.booking_data.trip_date)
            elif "tomorrow" in text_lower:
                date_obj = datetime.now() + timedelta(days=1)
                self.booking_data.trip_date = date_obj.strftime("%Y-%m-%d")
                if self.logger:
                    self.logger.log_booking_update("trip_date", self.booking_data.trip_date)
            elif "day after tomorrow" in text_lower or "overmorrow" in text_lower:
                date_obj = datetime.now() + timedelta(days=2)
                self.booking_data.trip_date = date_obj.strftime("%Y-%m-%d")
                if self.logger:
                    self.logger.log_booking_update("trip_date", self.booking_data.trip_date)
    
    def _detect_confirmation(self, text: str):
        """
        Detect confirmation or rejection keywords in user text.
        
        Args:
            text: User's spoken text
        """
        text_lower = text.lower()
        
        # Expanded confirmation keywords
        confirmation_keywords = [
            "yes", "yeah", "yep", "ok", "okay", "correct", "right", "sure", 
            "fine", "perfect", "that's right", "confirmed", "done", 
            "absolutely", "exactly"
        ]
        
        # Detect confirmation
        if any(keyword in text_lower for keyword in confirmation_keywords):
            if self.booking_data.confirmation_status == "pending":
                self.booking_data.confirmation_status = "confirmed"
                if self.logger:
                    self.logger.log_confirmation_status("confirmed")
        
        # Detect rejection
        rejection_keywords = ["no", "not interested", "cancel", "don't want", "not now"]
        if any(keyword in text_lower for keyword in rejection_keywords):
            if self.booking_data.confirmation_status == "pending":
                self.booking_data.confirmation_status = "not_interested"
                if self.logger:
                    self.logger.log_confirmation_status("not_interested")
    
    def check_booking_confirmed_marker(self, response_text: str) -> bool:
        """
        Check if the LLM response contains the BOOKING_CONFIRMED marker.
        This indicates the customer has confirmed and we should send to API.
        
        Args:
            response_text: The LLM's response text
            
        Returns:
            True if BOOKING_CONFIRMED marker is present
        """
        return "BOOKING_CONFIRMED" in response_text
    
    def get_booking_data(self) -> BookingData:
        """
        Get the current booking data.
        
        Returns:
            BookingData object with collected information
        """
        return self.booking_data
    
    def is_booking_complete(self) -> bool:
        """
        Check if all required booking information has been collected.
        
        Returns:
            True if booking is complete, False otherwise
        """
        return self.booking_data.is_complete()
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation for debugging.
        
        Returns:
            String summary of conversation
        """
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.conversation_history if msg["role"] == "assistant"]
        
        return f"Conversation: {len(user_messages)} user messages, {len(assistant_messages)} assistant messages"
