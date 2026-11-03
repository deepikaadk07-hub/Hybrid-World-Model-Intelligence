from world_model import WorldModel

model = WorldModel()
model.csv_file = "tracking_id_test_log.csv"

# Person is detected outside
r1 = model.update(1, "person", (300, 200), False)

# Same person enters
r2 = model.update(1, "person", (450, 200), True)

# Tracking ID changes from 1 to 2
r3 = model.update(2, "person", (460, 200), True)

print("Original ID, outside:", r1["zone_state"])
print("Original ID, entering:", r2["zone_state"])
print("New ID, detected inside:", r3["zone_state"])