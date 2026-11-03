from datetime import datetime


def log_event(event_type, risk, reason):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = (
        f"[{timestamp}] "
        f"EVENT: {event_type} | "
        f"RISK: {risk} | "
        f"REASON: {reason}"
    )

    print(message)

    with open("events.log", "a", encoding="utf-8") as file:
        file.write(message + "\n")