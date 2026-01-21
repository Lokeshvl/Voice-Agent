from fuzzywuzzy import fuzz

keywords = {
    "tata ace": "Tata Ace",
    "bolero": "Bolero",
    "bolero pickup": "Bolero",
}

phrases = ["tata", "ac", "tata ac", "open", "truck", "open truck"]

print("Debugging Fuzzy Scores:\n")

for phrase in phrases:
    print(f"Phrase: '{phrase}'")
    for keyword, name in keywords.items():
        score = fuzz.partial_ratio(phrase, keyword)
        if score > 50:
            print(f"  vs '{keyword}' ({name}): {score}")
