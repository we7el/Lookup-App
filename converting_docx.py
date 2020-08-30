import docx2txt

# extract text
text = docx2txt.process("file.doc")
text = [t.strip() + '\n' for t in text.split('\n') if t.strip()]

with open('doc2txt.txt', 'x') as f:
    f.writelines(text)
f.close()

print(type(text))
print(len(text))

for t in text:
    print(t)
