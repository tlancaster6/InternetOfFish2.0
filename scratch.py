from benchmark import BenchMarker
import pathlib

bm = BenchMarker(model_id='effdet0_fish')
test_dir = pathlib.Path.home().resolve() / 'valid'
bm.test_occupancy_accuracy(test_dir)
