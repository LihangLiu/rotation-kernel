Preparation:
	mkdir outputs
	mkdir outputs/losses
	mkdir outputs/params
	cd data
	mv path/to/ModelNet40_npy_30 data/ModelNet_list 3DShapeNets

Train:
	cd src-clasify32/shapenets32-000
	python3 train.py
