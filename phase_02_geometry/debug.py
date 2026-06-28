import cv2
import numpy as np

class GeometryDebug:
    def __init__(self, size=(600, 800)):
        self.width, self.height = size

    def render(self, vehicles):
        """
        Creates a text-based debug window tracing the mathematical pipeline.
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        y = 30
        cv2.putText(frame, "Geometry Debug Pipeline", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 40

        for vg in vehicles:
            if y > self.height - 100:
                break # Avoid drawing off screen
            
            lines = [
                f"--- Vehicle ID: {vg.id} ({vg.class_name}) ---",
                f"IMG: Left({int(vg.img_left[0])},{int(vg.img_left[1])}) | Right({int(vg.img_right[0])},{int(vg.img_right[1])})",
                f"BEV: Left({int(vg.bev_left[0])},{int(vg.bev_left[1])}) | Right({int(vg.bev_right[0])},{int(vg.bev_right[1])})",
                f"Width: {vg.width_px} px -> {vg.width_m:.2f} m",
                f"Center BEV: ({vg.bev_center[0]:.2f}, {vg.bev_center[1]:.2f})",
                f"Validation: {getattr(vg, 'validation_status', 'Unknown')}"
            ]

            for line in lines:
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y += 20
            
            y += 10 # spacing between vehicles

        return frame

def log_vehicle_geometry(vg, mpp=0.0255):
    print(f"Vehicle {vg.id}")
    print("Image")
    print(f"Left=({int(vg.img_left[0])},{int(vg.img_left[1])})")
    print(f"Right=({int(vg.img_right[0])},{int(vg.img_right[1])})")
    print("Projected")
    print(f"Left=({int(vg.bev_left[0])},{int(vg.bev_left[1])})")
    print(f"Right=({int(vg.bev_right[0])},{int(vg.bev_right[1])})")
    print("Projected Width")
    print(f"{vg.width_px} px")
    print("")
    
    print("Lane Width")
    print("145 px")
    print("Lane Width")
    print("3.70 m")
    print("Meters Per Pixel")
    print(f"{mpp:.4f}")
    print("Projected Width")
    print(f"{vg.width_px} px")
    print("Vehicle Width")
    print(f"{vg.width_m:.2f} m")
    print("")
