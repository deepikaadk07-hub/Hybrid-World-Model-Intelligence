from decision_engine import make_decision


world_state = {
    1: {
        "id": 1,
        "class": "person",
        "position": (300, 300),
        "time": 123456
    }
}


vlm_description = "a person standing near a restricted area"


result = make_decision(
    world_state,
    vlm_description
)


print("Risk:", result["risk"])
print("Reason:", result["reason"])