from benchmark import BenchMarker
import pathlib

bm = BenchMarker(model_id='effdet0_fish')
parent_dir = pathlib.Path.home() / 'Benchmarking'
bm.generate_dense_detection_data(parent_dir)
