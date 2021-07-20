During initial training, we set the base model as trainable=False and only train newly added layers for a few epochs.
In the second stage, we unfreeze some (or all) of the layers from the base model and train those layers, along with the newly added layers, for a few epochs with small learning rate.

Training dataset: ..\\datasets\\manually-created
Validation dataset: ..\\datasets\\imagenette2-320\\val (50%)
Testing dataset: ..\\datasets\\imagenette2-320\\val (remaining 50%)

	a) momentum=0.9; 1024-0.4 (with Flatten layer)
		Best results for this configuration:
		
		Training LR: 0.003
		Tuning LR: 0.0007     (Last 10 layers)
		Training accuracy:  0.995666683
		Validation accuracy: 0.980846763
		Testing accuracy: 0.985574424


	b) momentum=0.9; 512-0.3 (with Flatten layer)
		Best results for this configuration:
		
		Training LR: 0.003
		Tuning LR: 0.0006     (Last 10 layers)
		Training accuracy:  0.998333
		Validation accuracy: 0.980847
		Testing accuracy: 0.982483


	c) momentum=0.9; 512-0.3 (with global average pool)
		Best results for this configuration:
		
		Training LR: 0.003
		Tuning LR: 0.0006     (Last 10 layers)
		Training accuracy: 0.995000
		Validation accuracy: 0.985887
		Testing accuracy: 0.984544


	d) momentum=0.9; 512-0.4 (with global average pool)
		Best results for this configuration:
		
		Training LR: 0.003
		Tuning LR: 0.0006     (Last 10 layers)
		Training accuracy: 0.989667
		Validation accuracy: 0.989415
		Testing accuracy: 0.985059


	e) momentum=0.9; 512-0.4 (with global average pool)
		Best results for this configuration:
		
		Training LR: 0.003
		Tuning LR: 0.0003     (Last 10 layers)
		Training accuracy: 0.993667
		Validation accuracy: 0.981855
		Testing accuracy: 0.990211