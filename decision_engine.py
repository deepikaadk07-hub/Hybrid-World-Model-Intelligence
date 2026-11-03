def make_decision(world_state, vlm_description=""):

    risk = "LOW"
    reason = "Normal situation"

    # Check whether a PERSON is inside the restricted zone
    for obj_id, obj in world_state.items():

        if obj["class"] == "person":

            x, y = obj["position"]

            # Same restricted zone as main.py
            if 400 <= x <= 600 and 100 <= y <= 400:

                risk = "HIGH"

                reason = "Person detected inside restricted zone"

                if vlm_description:
                    reason += " | VLM: " + vlm_description

                break

    return {
        "risk": risk,
        "reason": reason
    }