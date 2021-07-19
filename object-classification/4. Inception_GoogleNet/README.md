During initial training, we set the base model as trainable=False and only train newly added layers for a few epochs.
In the second stage, we unfreeze some (or all) of the layers from the base model and train those layers, along with the newly added layers, for a few epochs with small learning rate.

Training dataset: ..\\datasets\\manually-created
Validation dataset: ..\\datasets\\imagenette2-320\\val (50%)
Testing dataset: ..\\datasets\\imagenette2-320\\val (remaining 50%)

	a) momentum=0.9; 512-0.4 (with Flatten layer)
		Best results for this configuration:
		
		Training LR: 0.003
		Tuning LR: 0.0005     (Last 22 layers)
		Training accuracy:  0.999333
		Validation accuracy: 0.982863
		Testing accuracy: 0.982998


	b) momentum=0.9; 512-0.6 (with Flatten layer)
		Best results for this configuration:
		
		Training LR: 0.003
		Tuning LR: 0.0005     (Last 22 layers)
		Training accuracy:  0.994333
		Validation accuracy: 0.987903
		Testing accuracy: 0.983514


	b) momentum=0.9; 256-0.3 (with global average pool)
		Best results for this configuration:
		
		Training LR: 0.005
		Tuning LR: 0.0005     (Last 22 layers)
		Training accuracy: 0.994333
		Validation accuracy: 0.983367
		Testing accuracy: 0.98712