import os

data = {}
for split in ['train', 'test']:
    with open('ModelNet_list/npy_list.30.points.{0}'.format(split)) as fp:
        data[split] = {}
        for line in fp.readlines():
            x, y = line.strip().split()
            y = int(y)
            if y not in data[split]:
                data[split][y] = []
            data[split][y].append(x)

data['valid'] = {}
for k in sorted(data['train'].keys()):
    files = sorted(data['train'][k])
    pivot = int(len(files) * .1)
    data['train'][k] = files[:-pivot]
    data['valid'][k] = files[-pivot:]

for split in ['train', 'valid', 'test']:
    counts = 0
    with open('{0}-32.txt'.format(split), 'w') as fp:
        for k in sorted(data[split].keys()):
            for datum in sorted(data[split][k]):
                datum = os.path.join('ModelNet_list', datum)
                print('{0} {1}'.format(datum, k), file = fp)
                counts += 1
    print('==> #{0} = {1}'.format(split, counts))
