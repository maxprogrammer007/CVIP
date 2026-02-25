import os
import glob

files = glob.glob('**/*.py', recursive=True)
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Replace literal \ " " " with " " "
    # When I wrote `"""` in JSON, the file got `"""`
    content = content.replace('\\"\\"\\"', '"""')
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
print("Docstrings fixed.")
