
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Explicitly add user site-packages
sys.path.append('/Users/lokesh/Library/Python/3.9/lib/python/site-packages')

from app.llm import LLMAgent
from core.prompt import BookingData

def run_simulation():
    print("🚀 Starting Conversation Simulation Test...\n")
    
    # Initialize Agent
    # Note: Without OPENAI_API_KEY, it will use echo mode, but extraction logic still runs on inputs
    # We need to simulate the AI's responses for the extraction logic to work fully if it relies on AI confirmation
    # However, our recent fix extracts from AI response too.
    # In echo mode, the AI just echoes. So we might need to manually trigger extraction on "simulated" AI responses
    # or just test the user input extraction primarily.
    
    llm = LLMAgent()
    
    # Scenario 1: Happy Path
    print("--- Scenario 1: Happy Path ---")
    
    # 1. User provides Pickup and Drop
    user_input = "I need a trip from Chennai to Bangalore"
    print(f"User: {user_input}")
    llm._extract_booking_info(user_input)
    
    # Simulate AI response (since we don't have real LLM in test env usually)
    # But wait, the user wants to check the *workflow*.
    # The workflow logic is in `llm.generate_response` -> `_extract_booking_info`.
    
    print(f"  -> Extracted Pickup: {llm.booking_data.pickup_location}")
    print(f"  -> Extracted Drop: {llm.booking_data.drop_location}")
    
    if llm.booking_data.pickup_location == "Chennai" and llm.booking_data.drop_location == "Bangalore":
        print("  ✅ Location Extraction: PASS")
    else:
        print("  ❌ Location Extraction: FAIL")

    # 2. User provides Truck Type (with fuzzy match)
    user_input = "I need a Tata AC open truck"
    print(f"\nUser: {user_input}")
    llm._extract_booking_info(user_input)
    
    print(f"  -> Extracted Vehicle: {llm.booking_data.vehicle_type}")
    print(f"  -> Extracted Body: {llm.booking_data.body_type}")
    
    if llm.booking_data.vehicle_type == "Tata Ace" and llm.booking_data.body_type == "Open":
        print("  ✅ Vehicle/Body Extraction: PASS")
    else:
        print("  ❌ Vehicle/Body Extraction: FAIL")

    # 3. User provides Material
    user_input = "carrying steel pipes"
    print(f"\nUser: {user_input}")
    llm._extract_booking_info(user_input)
    
    print(f"  -> Extracted Material: {llm.booking_data.goods_type}")
    
    # Flexible extraction captures "Steel Pipes" from "carrying steel pipes"
    if llm.booking_data.goods_type in ["Steel", "Steel Pipes"]:
        print("  ✅ Material Extraction: PASS")
    else:
        print("  ❌ Material Extraction: FAIL")

    # 4. User provides Date
    user_input = "trip is for tomorrow"
    print(f"\nUser: {user_input}")
    llm._extract_booking_info(user_input)
    
    print(f"  -> Extracted Date: {llm.booking_data.trip_date}")
    
    if llm.booking_data.trip_date:
        print("  ✅ Date Extraction: PASS")
    else:
        print("  ❌ Date Extraction: FAIL")

    # 5. Check Completeness
    if llm.booking_data.is_complete():
        print("\n✅ Booking Data Complete: PASS")
    else:
        print("\n❌ Booking Data Complete: FAIL")
        print(f"Missing: {[k for k,v in llm.booking_data.__dict__.items() if v is None]}")

    # 6. Simulate Confirmation
    # AI says confirmation
    ai_confirmation = "Pickup Chennai, drop Bangalore, truck Tata Ace, body Open, material Steel, date 2025-11-28. Correct?"
    print(f"\nAI (Simulated): {ai_confirmation}")
    llm._extract_booking_info(ai_confirmation) # Test extraction from AI text
    
    # User confirms
    user_input = "Yes, that's correct"
    print(f"User: {user_input}")
    llm._detect_confirmation(user_input)
    
    if llm.booking_data.confirmation_status == "confirmed":
        print("  ✅ Confirmation Detection: PASS")
    else:
        print("  ❌ Confirmation Detection: FAIL")

    print("\n--- Scenario 2: Corrections & AI Confirmation Extraction ---")
    llm2 = LLMAgent()
    
    # User gives wrong info first
    user_input = "Pickup in Sanit, drop in Benbrook"
    print(f"User: {user_input}")
    llm2._extract_booking_info(user_input)
    print(f"  -> Initial Pickup: {llm2.booking_data.pickup_location}")
    
    # AI confirms (Simulating the fix we made where AI confirmation overrides/cleans)
    # Let's say AI understood it as "Chennai" and "Bangalore" contextually (or user corrected it)
    # Here we test if the AI confirmation string is parsed correctly
    ai_confirmation = "Pickup in Chennai, drop in Bangalore, truck type Tata IC"
    print(f"AI (Simulated): {ai_confirmation}")
    llm2._extract_booking_info(ai_confirmation)
    
    print(f"  -> Updated Pickup: {llm2.booking_data.pickup_location}")
    print(f"  -> Updated Drop: {llm2.booking_data.drop_location}")
    print(f"  -> Updated Vehicle: {llm2.booking_data.vehicle_type}")
    
    if llm2.booking_data.pickup_location == "Chennai" and llm2.booking_data.drop_location == "Bangalore" and llm2.booking_data.vehicle_type == "Tata Ace":
        print("  ✅ AI Confirmation Extraction: PASS")
    else:
        print("  ❌ AI Confirmation Extraction: FAIL")

if __name__ == "__main__":
    run_simulation()
