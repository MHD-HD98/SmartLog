# app/core/tracking.py

from collections import defaultdict


class TrackingContext:
    def __init__(self):
        self.unknown_faces = {}  # {uid: [embeddings]}
        self.logged_unknowns = {}  # key = unknown_id, value = last logged action
        self.unknown_id_counter = 0
        self.unknown_action_log = {}  # {uid: {"I": timestamp, "O": timestamp}}
        self.last_positions = {}
        self.unknown_timestamps = {}
        self.unregistered_count = 0
        self.last_logged = {}
        self.positions = {}

    def match_unknown(self, embedding, threshold):
        from scipy.spatial.distance import cosine

        for uid, embeddings in self.unknown_faces.items():
            for stored_embedding in embeddings:
                similarity = 1 - cosine(embedding, stored_embedding)
                if similarity > threshold:
                    # Update the list with the new embedding to keep it current
                    self.unknown_faces[uid].append(embedding)
                    return uid

        # If no match found, assign a new ID
        self.unknown_id_counter += 1
        uid = f"Unknown #{self.unknown_id_counter}"
        self.unknown_faces[uid] = [embedding]
        return uid

    def update_position(self, name, cx, left_line_x, right_line_x):
        prev_x = self.positions.get(name, cx)
        self.positions[name] = cx
        if prev_x < left_line_x and cx >= left_line_x:
            return "I"
        elif prev_x > right_line_x and cx <= right_line_x:
            return "O"
        return None

    def cleanup_positions(self, active_ids):
        # Remove stale entries
        all_ids = list(self.last_positions.keys())
        for pid in all_ids:
            if pid not in active_ids:
                self.last_positions.pop(pid, None)
                self.logged_unknowns.pop(pid, None)

    def should_log(self, name, current_time, buffer_time):
        last = self.last_logged.get(name, 0)
        if current_time - last > buffer_time:
            return True
        return False

    def log_timestamp(self, name, current_time):
        self.last_logged[name] = current_time

