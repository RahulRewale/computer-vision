Using GlobalAveragePooling layer leads to better accuracy compared to Flatten layer

Tuning last 4 layers of the VGG16 base model leads to better accuracy compared to tuning last 8 layers		


Training dataset: ..\\datasets\\manually-created
Validation dataset: ..\\datasets\\imagenette2-320\\val (50%)
Testing dataset: ..\\datasets\\imagenette2-320\\val (remaining 50%)

	a) momentum=0.9; 512-0.5-512-0.5 (with global average pool)
		
		Training LR: 0.005
		Tuning LR: 0.00005     (Last 4 layers)
		Training accuracy: 0.97966665
		Validation accuracy: 0.967741907
		Testing accuracy: 0.968057692


	b) momentum=0.9; 512-0.5-512-0.5 (with global average pool)
		
		Training LR: 0.008
		Tuning LR: 0.001     (Last 4 layers)
		Training accuracy: 0.980000
		Validation accuracy: 0.961694
		Testing accuracy: 0.967027