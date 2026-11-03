
from world_model import WorldModel

model = WorldModel()
model.csv_file = "transition_test_log.csv"

# False = outside the zone; True = inside the zone
zone_sequence = [
    False, False, True, True, False, False
]

expected_states = [
    "OUTSIDE",
    "OUTSIDE",
    "ENTERING",
    "STAYING_INSIDE",
    "LEAVING",
    "OUTSIDE"
]

actual_states = []

for frame, inside in enumerate(zone_sequence, start=1):
    result = model.update(
        object_id=999,
        object_class="person",
        position=(100 + frame * 10, 200),
        inside_zone=inside
    )

    predicted = result["zone_state"]
    actual_states.append(predicted)

    print(
        f"Frame {frame}: "
        f"Inside={inside}, State={predicted}"
    )

print("\nExpected:", expected_states)
print("Actual:  ", actual_states)

if actual_states == expected_states:
    print("\nPASS: All transition tests passed.")
else:
    print("\nFAIL: Transition logic needs correction.")
