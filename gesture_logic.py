def detect_gesture(landmarks):
    if not landmarks:
        return ""

    fingers = 0
    tip_ids = [4, 8, 12, 16, 20]

    if landmarks[tip_ids[0]][0] > landmarks[tip_ids[0] - 1][0]:
        fingers += 1

    for i in range(1, 5):
        if landmarks[tip_ids[i]][1] < landmarks[tip_ids[i] - 2][1]:
            fingers += 1

    # 🔥 Map to words (SIGN LANGUAGE MEANING)
    mapping = {
        1: "Hello",
        2: "Thank You",
        3: "Yes",
        4: "No",
        5: "Help"
    }

    return mapping.get(fingers, "")