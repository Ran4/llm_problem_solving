.phony: run

run:
	@poetry run python3.13 src/main.py

cropper:
	@poetry run python3.13 src/cropper.py
