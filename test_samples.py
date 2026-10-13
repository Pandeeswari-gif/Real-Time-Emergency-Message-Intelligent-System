"""
test_samples.py
===============
Quick check of the whole system with sample messages.

Run:   python test_samples.py

Each test has the expected emergency type, priority, location text and people
count. The script prints PASS/FAIL per field and the overall pass rate.
Use None when a field should not be checked.

These messages were written by hand and are NOT in the training dataset, so the
result shows how the system behaves on new wording. Add your own messages to
TEST_CASES to keep testing as you improve the project.
"""

from nlp_utils import EmergencyAnalyzer

# (message, emergency type, priority, location must contain, people affected, service must contain)
TEST_CASES = [
    # --- the two examples from the project brief ---------------------------------
    ("There is a major accident near the railway station. Two people are seriously injured. "
     "Please send an ambulance immediately.",
     "Accident", "Critical", "Railway Station", "2", "Ambulance"),
    ("There is a fire accident at Madurai bus stand. Three people are injured. "
     "Please send an ambulance and fire rescue team immediately.",
     "Fire", "Critical", "Madurai Bus Stand", "3", "Ambulance + Fire Service"),
    # --- new wording ------------------------------------------------------------------
    ("A speeding bus knocked down a cyclist near Koyambedu market. He is bleeding badly and unconscious.",
     "Accident", "Critical", "Koyambedu Market", "1", "Ambulance"),
    ("Two autos collided at the Saidapet signal. A woman has a broken arm. Please send an ambulance.",
     "Accident", "High", "Saidapet Signal", "1", "Ambulance"),
    ("Flames are coming out of the third floor of Lakshmi Textiles in Tiruppur and five workers are trapped inside!",
     "Fire", "Critical", "Lakshmi Textiles", "5", "Fire Service"),
    ("The whole of Kotturpuram is under water. Around 20 people are stranded on rooftops. Send boats urgently.",
     "Flood", "High", "Kotturpuram", "20", "Disaster Management Team"),
    ("My grandfather suddenly fell down and is not responding, no pulse. We are at 12 Gandhi Street. Please come immediately.",
     "Medical Emergency", "Critical", "Gandhi Street", None, "Ambulance"),
    ("Severe tremors were felt in Bhuj. A three storey building collapsed and many people are buried under it.",
     "Earthquake", "Critical", "Bhuj", "Many", "Rescue Team"),
    ("Two men on a bike snatched my chain near Anna Nagar tower and pushed me. I am hurt. Please send the police.",
     "Crime", "High", "Anna Nagar Tower", None, "Police"),
    ("Armed men are inside the bank at Nungambakkam and firing shots. Send police immediately!",
     "Crime", "Critical", "Bank", None, "Police"),
    ("Two children are drowning in the pond behind the temple. Please help immediately!",
     "Rescue", "Critical", "Pond", "2", "Rescue Team"),
    ("A monkey is stuck in the electric fence near our school. Need help to release it.",
     "Rescue", "Medium", "School", None, "Rescue Team"),
    ("The drainage near my house is blocked and smells bad. Kindly send someone to clean it.",
     "Other", "Medium", "House", None, None),
    ("What is the phone number of the nearest fire station? Just asking.",
     "Fire", "Low", None, None, None),
    # --- edge cases: missing location, missing people, invalid text ---------------
    ("Someone is unconscious and bleeding. Please send help immediately.",
     None, "Critical", "Not specified", "1", "Ambulance"),
    ("Fire in our apartment building, smoke is spreading to all floors of Sunrise Apartments in Guindy.",
     "Fire", "Critical", "Sunrise Apartments", "Not mentioned", "Fire Service"),
    ("fire at madurai bus stand three injured pls send ambulance fast",
     "Fire", "Critical", "Madurai Bus Stand", "3", "Ambulance + Fire Service"),
]

INVALID_INPUTS = ["", "   ", "a", "!!!???", "12345", None]


def main() -> None:
    analyzer = EmergencyAnalyzer()
    passed = total = 0
    print("=" * 78)
    for message, category, priority, location, people, service in TEST_CASES:
        result = analyzer.analyze(message)
        checks = {
            "type": (category, result["emergency_type"], lambda e, a: e == a),
            "priority": (priority, result["priority"], lambda e, a: e == a),
            "location": (location, result["location"], lambda e, a: e.lower() in a.lower()),
            "people": (people, result["people_affected"], lambda e, a: e == a),
            "service": (service, result["required_service"], lambda e, a: e.lower() in a.lower()),
        }
        print(f"\n> {message}")
        for name, (expected, actual, compare) in checks.items():
            if expected is None:
                continue
            ok = compare(expected, actual)
            total += 1
            passed += ok
            print(f"   {'PASS' if ok else 'FAIL'}  {name:9s} expected={expected!r:28} got={actual!r}")

    print("\n" + "=" * 78)
    print("Invalid input handling (must not crash):")
    for bad in INVALID_INPUTS:
        result = analyzer.analyze(bad)
        ok = result.get("success") is False
        total += 1
        passed += ok
        print(f"   {'PASS' if ok else 'FAIL'}  {bad!r:10} -> {result.get('error')}")

    print("=" * 78)
    print(f"Checks passed: {passed}/{total}  ({100 * passed / total:.0f}%)")


if __name__ == "__main__":
    main()
