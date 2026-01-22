.PHONY: test evaluate compare visualize extract clean

# Run unit tests
test:
	python3 scripts/run_pipeline.py --test

# Run evaluation with default method
evaluate:
	python3 scripts/run_pipeline.py

# Compare all methods
compare:
	python3 scripts/run_pipeline.py --compare

# Generate visualizations
visualize:
	python3 scripts/run_pipeline.py --visualize

# Extract keypoints from RGB images
extract:
	python3 superpoint/extract_keypoints.py -i data/rgb -o data/keypoints

# Clean output files
clean:
	rm -rf data/output
