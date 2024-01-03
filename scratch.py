# sendgrid_key = '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'

from benchmark import BenchMarker

bm = BenchMarker(model_id='effdet0_fish')
bm.run_all_benchmarks()
bm = BenchMarker(model_id='effdet1_fish')
bm.run_all_benchmarks()
