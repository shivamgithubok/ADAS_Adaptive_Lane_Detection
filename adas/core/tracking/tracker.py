from collections import defaultdict
from adas.config.settings import config

class TrackerHistory:
    def __init__(self, max_history=30):
        # self.tracks[track_id] = {
        #     "hits": int, (total frames seen)
        #     "lost_frames": int,
        #     "track_history": list of (x,y),
        #     "center_history": list of (cx,cy),
        #     "conf_history": list of float,
        #     "velocity": (vx, vy),
        #     "updated_this_frame": bool,
        #     "age": int (total frames since first seen, hits + lost)
        # }
        self.tracks = defaultdict(lambda: {
            "hits": 0,
            "lost_frames": 0,
            "age": 0,
            "track_history": [],
            "center_history": [],
            "conf_history": [],
            "velocity": (0.0, 0.0),
            "updated_this_frame": False
        })
        self.max_history = max_history

    def update(self, detected_objects):
        """
        detected_objects: list of dicts with 'track_id', 'bbox', 'conf'
        """
        for trk_id in self.tracks:
            self.tracks[trk_id]["updated_this_frame"] = False

        for obj in detected_objects:
            track_id = obj['track_id']
            if track_id is None:
                continue

            x1, y1, x2, y2 = obj['bbox']
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            conf = obj.get('conf', 0.0)

            track = self.tracks[track_id]
            track["hits"] += 1
            track["lost_frames"] = 0
            track["updated_this_frame"] = True

            if len(track["center_history"]) > 0:
                prev_cx, prev_cy = track["center_history"][-1]
                vx = cx - prev_cx
                vy = cy - prev_cy
                track["velocity"] = (vx, vy)

            track["track_history"].append((x1, y1, x2, y2))
            track["center_history"].append((cx, cy))
            track["conf_history"].append(conf)

            if len(track["track_history"]) > self.max_history:
                track["track_history"].pop(0)
            if len(track["center_history"]) > self.max_history:
                track["center_history"].pop(0)
            if len(track["conf_history"]) > self.max_history:
                track["conf_history"].pop(0)

        tracks_to_delete = []
        for trk_id, track in self.tracks.items():
            track["age"] += 1
            if not track["updated_this_frame"]:
                track["lost_frames"] += 1
                if track["lost_frames"] > config.TRACK_MAX_AGE:
                    tracks_to_delete.append(trk_id)

        for trk_id in tracks_to_delete:
            del self.tracks[trk_id]

    def get_track_info(self, track_id):
        if track_id not in self.tracks:
            return None
        t = self.tracks[track_id]
        avg_conf = sum(t["conf_history"]) / len(t["conf_history"]) if len(t["conf_history"]) > 0 else 0.0
        return {
            "hits": t["hits"],
            "lost_frames": t["lost_frames"],
            "age": t["age"],
            "avg_conf": avg_conf,
            "velocity": t["velocity"],
            "is_valid": t["hits"] >= config.TRACK_MIN_HITS
        }
