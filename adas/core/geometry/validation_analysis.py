import os
import json
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from adas.core.geometry.metric_scaling import VehicleMetricEstimate

class GeometryValidator:
    def __init__(self, output_dir="phase_02_geometry/validation_results"):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "diagnostic_plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Data storage: {vehicle_id: [dict of frame stats, ...]}
        self.history = defaultdict(list)
        
    def add_observation(self, vme: VehicleMetricEstimate, fg, debug_data: dict):
        if vme is None or fg is None:
            return
            
        area = vme.observed_width_m * vme.observed_length_m
        
        # Extract heading angle in degrees
        heading_vec = fg.orientation_vector
        heading_deg = math.degrees(math.atan2(heading_vec[1], heading_vec[0]))
        
        proj_conf = debug_data.get("mask_completeness", 50.0) * 0.5 + debug_data.get("ground_density", 50.0) * 0.5
        gc_conf = debug_data.get("ground_density", 50.0)
        
        # proxy for temporal stability
        hist_len = debug_data.get("history_length", 1)
        temp_stability = min(100.0, hist_len * 5.0) 
        
        ground_pixels = debug_data.get("ground_pixels", 0)
        
        obs = {
            "width": vme.observed_width_m,
            "length": vme.observed_length_m,
            "area": area,
            "distance": vme.distance_m,
            "heading": heading_deg,
            "projection_confidence": proj_conf,
            "ground_contact_confidence": gc_conf,
            "temporal_stability": temp_stability,
            "ground_pixels": ground_pixels
        }
        self.history[vme.vehicle_id].append(obs)
        
    def generate_report(self):
        print("\n" + "="*50)
        print("GENERATING GEOMETRY VALIDATION REPORT")
        print("="*50)
        
        if not self.history:
            print("No data collected.")
            return
            
        all_widths = []
        all_lengths = []
        all_distances = []
        all_areas = []
        all_proj_conf = []
        all_temp_stab = []
        all_headings = []
        all_ground_pixels = []
        
        per_vehicle_stats = {}
        
        for vid, obs_list in self.history.items():
            widths = [o["width"] for o in obs_list]
            lengths = [o["length"] for o in obs_list]
            dists = [o["distance"] for o in obs_list]
            areas = [o["area"] for o in obs_list]
            p_confs = [o["projection_confidence"] for o in obs_list]
            t_stabs = [o["temporal_stability"] for o in obs_list]
            headings = [o["heading"] for o in obs_list]
            gps = [o["ground_pixels"] for o in obs_list]
            
            if len(widths) < 5:
                continue # ignore very short tracks
                
            per_vehicle_stats[vid] = {
                "min_width": np.min(widths),
                "max_width": np.max(widths),
                "mean_width": np.mean(widths),
                "std_width": np.std(widths),
                "min_length": np.min(lengths),
                "max_length": np.max(lengths),
                "mean_length": np.mean(lengths),
                "std_length": np.std(lengths),
                "min_dist": np.min(dists),
                "max_dist": np.max(dists),
                "mean_proj_conf": np.mean(p_confs)
            }
            
            all_widths.extend(widths)
            all_lengths.extend(lengths)
            all_distances.extend(dists)
            all_areas.extend(areas)
            all_proj_conf.extend(p_confs)
            all_temp_stab.extend(t_stabs)
            all_headings.extend(headings)
            all_ground_pixels.extend(gps)
            
        # Global metrics
        if len(all_widths) == 0:
            print("Not enough data.")
            return
            
        avg_w = np.mean(all_widths)
        avg_l = np.mean(all_lengths)
        avg_pc = np.mean(all_proj_conf)
        avg_ts = np.mean(all_temp_stab)
        
        heading_std = np.std(all_headings)
        
        # Correlations
        dist_w_corr = np.corrcoef(all_distances, all_widths)[0,1] if len(all_distances)>1 else 0
        dist_l_corr = np.corrcoef(all_distances, all_lengths)[0,1] if len(all_distances)>1 else 0
        gp_w_corr = np.corrcoef(all_ground_pixels, all_widths)[0,1] if len(all_ground_pixels)>1 else 0
        gp_l_corr = np.corrcoef(all_ground_pixels, all_lengths)[0,1] if len(all_ground_pixels)>1 else 0
        pc_w_corr = np.corrcoef(all_proj_conf, all_widths)[0,1] if len(all_proj_conf)>1 else 0
        ts_w_corr = np.corrcoef(all_temp_stab, all_widths)[0,1] if len(all_temp_stab)>1 else 0
        
        # Determine Scientific Case
        # If width collapses systematically with distance, strong negative correlation
        case_conclusion = ""
        if avg_w < 0.5 and avg_l < 1.0 and dist_w_corr < -0.3:
            case_conclusion = "CASE A\nGeometry is unstable.\nFix homography."
        elif dist_w_corr > -0.4 and dist_l_corr > -0.4 and avg_w > 0.3:
            case_conclusion = "CASE B\nGeometry is stable.\nMetric dimensions are limited by visible ground contact."
        else:
            case_conclusion = "CASE C\nProjection instability detected.\nInvestigate calibration."
            
        # Write txt report
        txt_path = os.path.join(self.output_dir, "geometry_validation_report.txt")
        with open(txt_path, "w") as f:
            f.write("GEOMETRY VALIDATION REPORT\n")
            f.write("==========================\n\n")
            
            for vid, st in per_vehicle_stats.items():
                f.write(f"Vehicle {vid}\n")
                f.write(f"Distance\t\t{st['min_dist']:.1f}-{st['max_dist']:.1f} m\n")
                f.write(f"Recovered Width\t\t{st['min_width']:.2f}-{st['max_width']:.2f} m\n")
                f.write(f"Recovered Length\t{st['min_length']:.2f}-{st['max_length']:.2f} m\n")
                f.write(f"Std Dev Width\t\t{st['std_width']:.2f} m\n")
                f.write(f"Std Dev Length\t\t{st['std_length']:.2f} m\n")
                f.write(f"Projection Confidence\t{st['mean_proj_conf']:.0f}%\n")
                f.write(f"Conclusion\t\tGeometry stable, metric dimensions incomplete\n\n")
                
            f.write("GLOBAL REPORT\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Width\t\t\t{avg_w:.2f} m\n")
            f.write(f"Average Length\t\t\t{avg_l:.2f} m\n")
            f.write(f"Average Projection Confidence\t{avg_pc:.1f}%\n")
            f.write(f"Average Temporal Stability\t{avg_ts:.1f}%\n")
            f.write(f"Average Footprint Stability\t{(1.0 - np.mean([s['std_width'] for s in per_vehicle_stats.values()]))*100:.1f}%\n")
            f.write(f"Average Heading Stability\t{heading_std:.2f} deg std\n\n")
            
            f.write("FINAL SCIENTIFIC CONCLUSION\n")
            f.write("-" * 20 + "\n")
            f.write(case_conclusion + "\n")
            
        # Write CSV
        csv_path = os.path.join(self.output_dir, "geometry_statistics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vehicle_id", "min_width", "max_width", "mean_width", "std_width", 
                             "min_length", "max_length", "mean_length", "std_length", "min_dist", "max_dist", "mean_proj_conf"])
            for vid, st in per_vehicle_stats.items():
                writer.writerow([vid, st['min_width'], st['max_width'], st['mean_width'], st['std_width'],
                                 st['min_length'], st['max_length'], st['mean_length'], st['std_length'], st['min_dist'], st['max_dist'], st['mean_proj_conf']])
                                 
        # Write JSON
        json_path = os.path.join(self.output_dir, "summary_metrics.json")
        with open(json_path, "w") as f:
            json.dump({
                "global_avg_width": avg_w,
                "global_avg_length": avg_l,
                "global_avg_proj_conf": avg_pc,
                "global_avg_temp_stab": avg_ts,
                "heading_std_dev": heading_std,
                "correlations": {
                    "dist_width": dist_w_corr,
                    "dist_length": dist_l_corr,
                    "ground_px_width": gp_w_corr,
                    "ground_px_length": gp_l_corr,
                    "proj_conf_width": pc_w_corr,
                    "temp_stab_width": ts_w_corr
                },
                "conclusion": case_conclusion.replace("\n", " ")
            }, f, indent=4)
            
        # Plotting
        try:
            plt.figure(figsize=(8,6))
            plt.scatter(all_distances, all_widths, alpha=0.5, c='blue')
            plt.title('Observed Width vs Distance')
            plt.xlabel('Distance (m)')
            plt.ylabel('Observed Width (m)')
            plt.grid(True)
            plt.savefig(os.path.join(self.plots_dir, "width_vs_distance.png"))
            plt.close()
            
            plt.figure(figsize=(8,6))
            plt.scatter(all_distances, all_lengths, alpha=0.5, c='red')
            plt.title('Observed Length vs Distance')
            plt.xlabel('Distance (m)')
            plt.ylabel('Observed Length (m)')
            plt.grid(True)
            plt.savefig(os.path.join(self.plots_dir, "length_vs_distance.png"))
            plt.close()
            
            plt.figure(figsize=(8,6))
            plt.scatter(all_distances, all_areas, alpha=0.5, c='green')
            plt.title('Estimated Ground Area vs Distance')
            plt.xlabel('Distance (m)')
            plt.ylabel('Area (m^2)')
            plt.grid(True)
            plt.savefig(os.path.join(self.plots_dir, "area_vs_distance.png"))
            plt.close()
            
            plt.figure(figsize=(8,6))
            plt.hist(all_widths, bins=20, color='blue', alpha=0.7)
            plt.title('Observed Width Histogram')
            plt.xlabel('Width (m)')
            plt.ylabel('Count')
            plt.grid(True)
            plt.savefig(os.path.join(self.plots_dir, "width_histogram.png"))
            plt.close()
            
            plt.figure(figsize=(8,6))
            plt.hist(all_lengths, bins=20, color='red', alpha=0.7)
            plt.title('Observed Length Histogram')
            plt.xlabel('Length (m)')
            plt.ylabel('Count')
            plt.grid(True)
            plt.savefig(os.path.join(self.plots_dir, "length_histogram.png"))
            plt.close()
            
            plt.figure(figsize=(8,6))
            plt.hist(all_proj_conf, bins=20, color='purple', alpha=0.7)
            plt.title('Projection Confidence Distribution')
            plt.xlabel('Confidence (%)')
            plt.ylabel('Count')
            plt.grid(True)
            plt.savefig(os.path.join(self.plots_dir, "projection_confidence.png"))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate plots. {e}")
            
        print("Reports generated successfully in:", self.output_dir)
        print(case_conclusion)
