prepare_data:
	env/bin/python3.12 code/prepare_data.py

train_transformer:
	env/bin/python3.12 code/train_transformer.py

check_quality:
	env/bin/python3.12 code/check_quality.py

analyze_dataset:
	env/bin/python3.12 code/analyze_dataset.py

test_run:
	env/bin/python3.12 code/test_run.py

install:
	env/bin/pip3.12 install -r code/requirements.txt