__author__ = 'etnc6d'

filename = '/media/Storage/thesis/mcnp.gitignore/meshtav'

file = open(filename, 'r')
#with open(filename, 'r') as file:

indx = 1
#fin = open(filename, 'r')
fout = open(filename+str(indx), 'w')

lines = file.readlines()

for line in lines:
    l = line.lstrip()
    if l.startswith("Energy"):
        fout.close()
        indx += 1
        fout = open(filename+str(indx), 'w')

    if not l.startswith('Total'):
        try:
            float(l.split()[0])
            fout.write(line)
        except:
            pass
fout.close()
file.close()

print("Finished.")