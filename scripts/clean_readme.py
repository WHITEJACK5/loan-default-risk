import re
p=open('README.md','r',encoding='utf-8', errors='ignore').read()
p=p.replace('—','--').replace('?','->').replace('–','-').replace('’',"'").replace('“','\"').replace('”','\"')
# replace any remaining non-ascii
p=re.sub(r'[^\x00-\x7F]+', '-', p)
open('README.md','w',encoding='utf-8', newline='\n').write(p)
print('cleaned')
