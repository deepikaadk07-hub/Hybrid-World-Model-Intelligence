from decision_engine import make_decision

# Test 1: Person outside the restricted zone
world_state = {
    1: {
        "class": "person",
        "position": (200, 250),
        "zone_state": "OUTSIDE"
    }
}
result = make_decision(world_state)
print("Test 1 - Outside:", result)

# Test 2: Person inside the restricted zone
world_state = {
    1: {
        "class": "person",
        "position": (500, 250),
        "zone_state": "STAYING_INSIDE"
    }
}
result = make_decision(world_state)
print("Test 2 - Inside:", result)

# Test 3: Person entering the restricted zone
world_state = {
    1: {
        "class": "person",
        "position": (450, 200),
        "zone_state": "ENTERING"
    }
}
result = make_decision(world_state)
print("Test 3 - Entering:", result)

# Test 4: VLM description included
result = make_decision(
    world_state,
    "Person approaching restricted area"
)
print("Test 4 - VLM:", result)

# Test 5: Non-person object inside the zone
world_state = {
    2: {
        "class": "car",
        "position": (500, 250),
        "zone_state": "STAYING_INSIDE"
    }
}
result = make_decision(world_state)
print("Test 5 - Non-person:", result)