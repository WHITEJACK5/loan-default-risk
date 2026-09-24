p=open('README.md','r',encoding='utf-8').read()
count=p.count('<!-- METRICS:END -->')
print('count before', count)
if count>1:
    first=p.find('<!-- METRICS:END -->')
    second=p.find('<!-- METRICS:END -->', first+1)
    p=p[:second] + p[second+len('<!-- METRICS:END -->\n'):]
    print('removed duplicate')
open('README.md','w',encoding='utf-8').write(p)
print('count after', p.count('<!-- METRICS:END -->'))
