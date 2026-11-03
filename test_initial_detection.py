from world_model import WorldModel

model = WorldModel()
model.csv_file = "initial_detection_test_log.csv"

result = model.update(
    object_id=1001,
    object_class="person",
    position=(450, 200),
    inside_zone=True
)

print("Initial detection state:", result["zone_state"])

if result["zone_state"] == "INSIDE":
    print("ISSUE FOUND: INSIDE is not one of the four required states.")
else:
    print("Initial detection state:", result["zone_state"])