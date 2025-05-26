# app/services/frame_processor.py

import cv2
import time
import csv
import os
from datetime import datetime
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from models.models import Smartlog
from core.config import settings
from core.tracking import TrackingContext
from api.endpoints.face import face_data_log
from core.crop import save_unique_cropped_face
from core.face_recognition import (
    load_known_embeddings,
    detect_faces_yolov8,
    get_facenet_embedding,
    recognize_face_from_embedding,
    extract_region,
    assign_unknown_id
)
from core.ffmpeg import get_stream_instance, set_stream_instance, FFmpegStream

# ---- Configuration ----
known_faces = load_known_embeddings(settings.EMBEDDING_PATH)
ctx = TrackingContext()

BUFFER_TIME = settings.BUFFER_TIME
UNKNOWN_SIMILARITY_THRESHOLD = settings.UNKNOWN_SIMILARITY_THRESHOLD
UNKNOWN_BUFFER_TIME = 15  # seconds

width, height = 1280, 720
left_line_x = int(width * 0.45)
right_line_x = int(width * 0.55)
roi_coords = (200, 200, 970, 550)

# ---- CSV Setup ----
csv_filename = settings.CSV_PATH
unknown_csv_filename = settings.UNKNOWN_CSV_PATH

for path, headers in [
    (csv_filename, ["Name", "Action", "Timestamp"]),
    (unknown_csv_filename, ["Unknown ID", "Action", "Timestamp"]),
]:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

# ---- Frame Generator ----
async def generate_frames(request: Request, db: AsyncSession):
    stream = get_stream_instance()
    if not stream:
        stream = FFmpegStream(settings.VIDEO_SOURCE)
        stream.start()
        set_stream_instance(stream)

    failure_count = 0
    max_failures = 10
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 1

    try:
        while True:
            try:
                frame = stream.read_frame()
                if frame is None:
                    raise ValueError("Invalid frame from FFmpeg pipe")
                failure_count = 0
            except Exception as err:
                failure_count += 1
                print(f"[Decode Error] Frame skipped ({failure_count}/{max_failures}): {err}")
                time.sleep(0.05)
                if failure_count >= max_failures:
                    print("[Stream Error] Restarting FFmpeg stream...")
                    stream.stop()
                    time.sleep(1)
                    stream.start()
                    failure_count = 0
                continue

            frame_count += 1
            current_time = time.time()

            # --- Apply ROI ---
            x1_roi, y1_roi, x2_roi, y2_roi = roi_coords
            roi_frame = extract_region(frame, x1_roi, y1_roi, x2_roi, y2_roi)

            detected_faces = detect_faces_yolov8(
                roi_frame, x_offset=x1_roi, y_offset=y1_roi
            )
            detected_people = []

            for face in detected_faces:
                x1, y1, x2, y2 = map(int, face["bbox"])
                face_crop = face["face_crop"]
                cx = (x1 + x2) // 2

                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    embedding = get_facenet_embedding(face_crop)
                    if embedding is None:
                        continue

                    name = recognize_face_from_embedding(embedding, known_faces)

                    if name:
                        action = ctx.update_position(name, cx, left_line_x, right_line_x)
                        if action and ctx.should_log(name, current_time, BUFFER_TIME):
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            ctx.log_timestamp(name, current_time)
                            db_entry = Smartlog(
                                first_name=name, log_time=datetime.now(), log_mode=action
                            )
                            db.add(db_entry)
                            await db.commit()
                            face_data_log.append({"name": name, "timestamp": timestamp, "action": action})
                            with open(csv_filename, "a", newline="") as f:
                                csv.writer(f).writerow([name, action, timestamp])
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected_people.append(name)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        uid = assign_unknown_id(embedding)
                        action = None

                        if cx < left_line_x:
                            action = "I"
                        elif cx > right_line_x:
                            action = "O"

                        ctx.last_positions[uid] = cx

                        should_log = False
                        if action:
                            last_action = ctx.logged_unknowns.get(uid)
                            if last_action != action:
                                ctx.logged_unknowns[uid] = action
                                should_log = False

                        if current_time - ctx.unknown_timestamps.get(uid, 0) > UNKNOWN_BUFFER_TIME:
                            ctx.unknown_timestamps[uid] = current_time
                            should_log = True

                        if should_log:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if action in ("I", "O"):
                                last_log_time = ctx.unknown_action_log.get(uid, {}).get(action)
                                if last_log_time is None or current_time - last_log_time > UNKNOWN_BUFFER_TIME:
                                    db_entry = Smartlog(
                                        first_name=uid, log_time=datetime.now(), log_mode=action
                                    )
                                    db.add(db_entry)
                                    await db.commit()
                                    if uid not in ctx.unknown_action_log:
                                        ctx.unknown_action_log[uid] = {}
                                    ctx.unknown_action_log[uid][action] = current_time
                            with open(csv_filename, "a", newline="") as f:
                                csv.writer(f).writerow([uid, action or "U", timestamp])
                            with open(unknown_csv_filename, "a", newline="") as f:
                                csv.writer(f).writerow(["Total Unknowns", ctx.unregistered_count, timestamp])
                        detected_people.append(uid)
                        cv2.putText(frame, uid, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    # Draw boxes without recognition
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

            ctx.cleanup_positions(detected_people)

            # Draw static lines and counters
            cv2.line(frame, (left_line_x, 0), (left_line_x, height), (0, 255, 255), 2)
            cv2.putText(frame, "IN", (left_line_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.line(frame, (right_line_x, 0), (right_line_x, height), (0, 0, 255), 2)
            cv2.putText(frame, "OUT", (right_line_x - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 255), 2)
            cv2.putText(frame, f"Unregistered Count: {ctx.unregistered_count}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

            if await request.is_disconnected():
                break

    except Exception as e:
        print(f"[Streaming error] {e}")
    finally:
        stream.stop()