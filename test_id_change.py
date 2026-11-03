
from world_model import WorldModel

model = WorldModel()

print("\nTRACKING ID CHANGE TEST")
print("-" * 45)

# Same person, original tracking ID
first = model.update(
    object_id=1,
    object_class="person",
    position=(450, 250),
    inside_zone=True
)

print("ID 1 state:", first["zone_state"])

# Same person, but tracker assigns a new ID
second = model.update(
    object_id=2,
    object_class="person",
    position=(460, 250),
    inside_zone=True
)

print("ID 2 state:", second["zone_state"])
print("Stored object IDs:", list(model.get_current_state().keys()))

if 1 in model.get_current_state() and 2 in model.get_current_state():
    print("RESULT: Both IDs are stored as separate objects.")
else:
    print("RESULT: Unexpected object storage behavior.")
