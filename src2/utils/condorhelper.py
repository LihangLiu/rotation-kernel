import os
import glob

def condor_config_content(project_desc, condor_log_path, exec_file_path):
    s = """+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="{0}"
+GPUJob=true
Requirements=(TARGET.GPUSlot && Eldar == True) 
Rank=memory
Environment=PATH=/scratch/cluster/leonliu/anaconda3/bin:/lusr/opt/condor/bin/:/lusr/opt/condor/bin/:/opt/cuda-8.0/bin:$PATH
Environment=PYTHONPATH=/u/leonliu/.local/lib/python2.7/site-packages:$PYTHONPATH
Environment=LD_LIBRARY_PATH=/u/leonliu/repos/cuDNN:/u/leonliu/repos/cuDNN/lib64:/opt/cuda-8.0/lib:/opt/cuda-8.0/lib64:$LD_LIBRARY_PATH

Universe=vanilla
Getenv=True

Log={1}log.$(Cluster).$(Process) 
Output={1}out.$(Cluster).$(Process) 
Error={1}err.$(Cluster).$(Process)
Executable={2}

Queue 1
    """.format(project_desc, condor_log_path, exec_file_path)
    
    return s

def exec_file_content(py_script, args=None, env=None):
    s = '#!/bin/bash\n'

    if env:
        s += 'source activate %s\n' % (env)

    s += 'python %s' % (py_script)
    if args:
    	for key in args:
    		s += ' --{0}=\"{1}\"'.format(key, args[key])
    return s






