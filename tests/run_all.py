import os
import utils

# add new test scripts here
files = ['basic.py', 'changing_kernels.py', 'load_checkpoint.py', 'maximize.py', 'slidingWindow.py', 'training_continuation.py']

# filter '.py' files endings
filtered_files = []
for file in files:
    exclude_files = ['__init__.py', 'run_all.py', 'test.py']
    exclude_file_extension = ['.pyc']
    
    if file in exclude_files:
        continue
    
    if any(list(map(lambda x: x in file, exclude_file_extension))):
        continue
        
    if file[-3:] != '.py':
        continue
        
    filtered_files.append(file)

print(filtered_files)
print(len(filtered_files))

# import
counter = 0
for file in filtered_files:
    mod = __import__(file[:-3], locals(), globals())
    utils.enablePrint()
    # out = mod.main()
    # print(out)