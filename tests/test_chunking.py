from src.features.chunking import sliding_window_chunks

def test_overlap_smaller_than_size():
	try:
		sliding_window_chunks("hello", size=5, overlap=5)
		assert False, "Expected ValueError"
	except ValueError:
		assert True