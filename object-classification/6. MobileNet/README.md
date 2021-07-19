Training dataset: ..\\datasets\\manually-created
Validation dataset: ..\\datasets\\imagenette2-320\\val (50%)
Testing dataset: ..\\datasets\\imagenette2-320\\val (remaining 50%)

Tuning the last 12 layers is better than training the last 25 layers 

	a) momentum=0.9; 512-0.3 (with Flatten layer)
		Best results for this configuration:
		
		Training LR: 0.003
		Tuning LR: 0.0005     (Last 12 layers of the  base model)
		Training accuracy: 0.996999979
		Validation accuracy: 0.977822602
		Testing accuracy: 0.976300895


	b) momentum=0.9; 512-0.4 (with Flatten layer)
		Best results for this configuration:
		
		Training LR: 0.003
		Tuning LR: 0.0008     (Last 12 layers of the  base model)
		Training accuracy: 0.999666691
		Validation accuracy: 0.981854856
		Testing accuracy: 0.980422437
		

	c) momentum=0.9; 512-0.3 (with Global Average Pool Layer)
		Best results for this configuration:
		
		Training LR: 0.003
		Tuning LR: 0.0008     (Last 12 layers of the base model)
		Training accuracy: 0.988333344
		Validation accuracy: 0.979838729
		Testing accuracy: 0.977846444
		

	d) momentum=0.9; 1024-0.4 (with Global Average Pool Layer)
		Best results for this configuration:
		
		Training LR: 0.005
		Tuning LR: 0.0008     (Last 12 layers of the  base model)
		Training accuracy: 0.995333314
		Validation accuracy: 0.983870983
		Testing accuracy: 0.984028876