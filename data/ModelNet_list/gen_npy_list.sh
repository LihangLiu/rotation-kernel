# convert .mat to .npy
python batch_mat2points.py /scratch/cluster/leonliu/Model /scratch/cluster/leonliu/Model 

# generate npy_list.txt
find ShapeNetCore.v2/02958343 -name '*.npy' > npy_list.txt

find ShapeNetCore.v2/02958343 -type d  -exec chmod 755 {} +

# append syn id
python append_syn_id.py npy_list.txt npy_list_with_synid.txt

# split into train and test
python split_npy_list.py npy_list_with_synid.txt



