import os

#print(os.getcwd())
#for module in os.listdir(os.path.dirname(__file__)):

# selecting all files
files = []
for (_, _, module) in os.walk("."):
    files += module


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
for file in filtered_files:
    __import__(file[:-3], locals(), globals())