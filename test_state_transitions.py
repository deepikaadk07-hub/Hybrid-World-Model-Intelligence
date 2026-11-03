
from world_model import WorldModel


def test_state_transitions():
    model = WorldModel()
    person_id = 1

    test_cases = [
        ((300, 250), False, "OUTSIDE"),
        ((420, 250), True, "ENTERING"),
        ((450, 250), True, "STAYING_INSIDE"),
        ((620, 250), False, "LEAVING"),
        ((650, 250), False, "OUTSIDE"),
    ]

    passed = 0

    print("\nWORLD MODEL STATE-TRANSITION TEST")
    print("-" * 55)

    for step, (position, inside_zone, expected) in enumerate(
        test_cases, start=1
    ):
        result = model.update(
            object_id=person_id,
            object_class="person",
            position=position,
            inside_zone=inside_zone,
        )

        actual = result["zone_state"]
        success = actual == expected

        if success:
            passed += 1

        print(
            f"Step {step}: Expected={expected:<16} "
            f"Actual={actual:<16} "
            f"{'PASS' if success else 'FAIL'}"
        )

    print("-" * 55)
    print(f"Results: {passed}/{len(test_cases)} tests passed")
    print(
        "Transition test rate: "
        f"{passed / len(test_cases) * 100:.2f}%"
    )


if __name__ == "__main__":
    test_state_transitions()
