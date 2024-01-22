from benchmark import ObjectDetectionBenchMarker, BehaviorDetectionBenchmarker
import pathlib

# bm = ObjectDetectionBenchMarker(model_id='effdet0_fish')
# parent_dir = pathlib.Path.home() / 'Benchmarking'
# bm.generate_dense_detection_data(parent_dir)
annotated_clip_dir = r"C:\Users\tucke\Benchmarking"
behave_bm = BehaviorDetectionBenchmarker(model_id='effdet0_fish', annotated_clip_dir=annotated_clip_dir)
behave_bm.generate_data_summary(frame_interval=1)