import time
from typing import List, Optional

class EventLogger:
    def __init__(self):
        self.vehicle_states = {}
        
    def update(self, frame_idx, vehicles, odz_states, occlusion_states=None):
        current_ids = set()
        
        state_map = {s.vehicle_id: s for s in odz_states}

        # Build occlusion lookup
        occ_map = {}
        if occlusion_states:
            occ_map = {s.vehicle_id: s for s in occlusion_states}
        
        for vg in vehicles:
            v_id = vg.id
            current_ids.add(v_id)
            
            state = state_map.get(v_id)
            status = state.status if state else "UNKNOWN"
            reason = state.reason if state else ""
            
            vme = vg.metric_estimate
            conf = vme.correction_confidence if vme else 0.0

            # Occlusion info
            occ = occ_map.get(v_id)
            tracking_state = occ.tracking_state if occ else "UNKNOWN"
            render_mode = occ.render_mode if occ else "FULL"
            visibility = occ.visibility_score if occ else 100.0
            
            if v_id not in self.vehicle_states:
                print(f"[INFO] Frame {frame_idx}: Vehicle {v_id} entered scene. Status: {status}")
                self.vehicle_states[v_id] = {
                    'status': status,
                    'conf': conf,
                    'last_seen': frame_idx,
                    'tracking_state': tracking_state,
                    'render_mode': render_mode,
                }
            else:
                prev_state = self.vehicle_states[v_id]
                
                # Check status change
                if prev_state['status'] != status:
                    if status == "INACTIVE":
                        print(f"[INFO] Frame {frame_idx}: Vehicle {v_id} became INACTIVE. Reason: {reason}")
                    else:
                        print(f"[INFO] Frame {frame_idx}: Vehicle {v_id} became ACTIVE.")
                
                # Check confidence drop
                if prev_state['conf'] - conf > 15.0:
                    print(f"[WARNING] Frame {frame_idx}: Vehicle {v_id} confidence dropped significantly ({(prev_state['conf'] - conf):.1f}%)")
                    
                # Check projection failure
                if not vme or not vg.footprint:
                    if prev_state.get('had_projection', True): # Only print once
                        print(f"[ERROR] Frame {frame_idx}: Vehicle {v_id} projection failed.")

                # ── Occlusion state transitions ────────────────────────
                prev_tracking = prev_state.get('tracking_state', 'UNKNOWN')
                if prev_tracking != tracking_state:
                    if tracking_state == "OCCLUDED" and prev_tracking == "TRACKED":
                        print(f"[OCCLUSION] Frame {frame_idx}: Vehicle {v_id} became OCCLUDED (vis={visibility:.0f}%)")
                    elif tracking_state == "TRACKED" and prev_tracking in ("OCCLUDED", "PREDICTED"):
                        print(f"[OCCLUSION] Frame {frame_idx}: Vehicle {v_id} REAPPEARED (vis={visibility:.0f}%)")
                    elif tracking_state == "PREDICTED":
                        print(f"[OCCLUSION] Frame {frame_idx}: Vehicle {v_id} entered PREDICTED state (fully occluded)")

                # Render mode transitions
                prev_render = prev_state.get('render_mode', 'FULL')
                if prev_render != render_mode:
                    if render_mode in ("POINT", "PREDICTED") and prev_render in ("FULL", "PARTIAL"):
                        print(f"[OCCLUSION] Frame {frame_idx}: Vehicle {v_id} render downgraded {prev_render} → {render_mode}")

                prev_state['status'] = status
                prev_state['conf'] = conf
                prev_state['last_seen'] = frame_idx
                prev_state['had_projection'] = (vme is not None and vg.footprint is not None)
                prev_state['tracking_state'] = tracking_state
                prev_state['render_mode'] = render_mode
                
        # Check for exits
        exited_ids = []
        for v_id, state_info in self.vehicle_states.items():
            if v_id not in current_ids and (frame_idx - state_info['last_seen'] > 15):
                print(f"[INFO] Frame {frame_idx}: Vehicle {v_id} exited scene.")
                exited_ids.append(v_id)
                
        for v_id in exited_ids:
            del self.vehicle_states[v_id]
