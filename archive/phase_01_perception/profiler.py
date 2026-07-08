import time

class Profiler:
    def __init__(self):
        self.times = {}
        self.starts = {}
        self.fps_history = []
        self.frame_counter = 0
        
    def start(self, name):
        self.starts[name] = time.perf_counter()
        
    def stop(self, name):
        if name in self.starts:
            elapsed_ms = (time.perf_counter() - self.starts[name]) * 1000
            self.times[name] = elapsed_ms
            
    def get_fps(self):
        total_ms = sum(self.times.values())
        if total_ms > 0:
            return 1000.0 / total_ms
        return 0.0
        
    def step(self):
        fps = self.get_fps()
        self.fps_history.append(fps)
        self.frame_counter += 1
        
    def should_print(self):
        return self.frame_counter % 30 == 0

    def print_profile(self):
        fps = self.fps_history[-1] if self.fps_history else 0.0
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0
        max_fps = max(self.fps_history) if self.fps_history else 0.0
        min_fps = min(self.fps_history) if self.fps_history else 0.0

        print("====================")
        print("FRAME PROFILER")
        print("====================")
        
        keys = ["Video Read", "YOLO", "Tracking", "Ground Contact", "Drawing"]
        # Include Total Frame
        total_frame_ms = sum(self.times.values())
        self.times["Total Frame"] = total_frame_ms
        keys.append("Total Frame")

        for k in keys:
            if k in self.times:
                print(f"{k}")
                if k in ("Ground Contact", "Total Frame"):
                    print(f"{self.times[k]:.1f} ms")
                else:
                    print(f"{int(round(self.times[k]))} ms")
                
        print("FPS")
        print(f"{int(round(fps))}")
        print("Average FPS")
        print(f"{int(round(avg_fps))}")
        print("Max FPS")
        print(f"{int(round(max_fps))}")
        print("Min FPS")
        print(f"{int(round(min_fps))}")
        print("====================\n")
        
        # Reset history after printing to calculate next 30 frames locally
        self.fps_history = []
