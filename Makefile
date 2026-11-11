PYTHON = env/bin/python3.12

prepare_data:
	$(PYTHON) code/prepare_data.py

train_transformer:
	$(PYTHON) code/train_transformer.py

check_quality:
	$(PYTHON) code/check_quality.py

analyze_dataset:
	$(PYTHON) code/analyze_dataset.py

install:
	env/bin/pip3.12 install -r code/requirements.txt