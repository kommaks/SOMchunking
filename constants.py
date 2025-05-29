# Parameters for chunking methods
threshold_standard = 0.3         # For standard chunking: threshold for cosine distance
min_chunk_size = 2               # Minimum number of sentences in a chunk (standard method)
initial_threshold = 0.7          # For double-pass method: initial threshold
appending_threshold = 0.8        # For double-pass method: appending threshold
merging_threshold = 0.7          # For double-pass method: merging threshold
max_chunk_length = 3             # Maximum number of sentences per chunk in double-pass
visualize = False                 # Print colored chunks during processing
