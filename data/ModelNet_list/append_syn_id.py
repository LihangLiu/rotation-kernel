
import os
import sys

raw_str = """
airplane
bathtub
bed
bench
bookshelf
bottle
bowl
car
chair
cone
cup
curtain
desk
door
dresser
flower_pot
glass_box
guitar
keyboard
lamp
laptop
mantel
monitor
night_stand
person
piano
plant
radio
range_hood
sink
sofa
stairs
stool
table
tent
toilet
tv_stand
vase
wardrobe
xbox
"""

NUM_SYNS = 40

syn_name_id_dict = {
'airplane': 0,
'bathtub': 1,
'bed': 2,
'bench': 3,
'bookshelf': 4,
'bottle': 5,
'bowl': 6,
'car': 7,
'chair': 8,
'cone': 9,
'cup': 10,
'curtain': 11,
'desk': 12,
'door': 13,
'dresser': 14,
'flower_pot': 15,
'glass_box': 16,
'guitar': 17,
'keyboard': 18,
'lamp': 19,
'laptop': 20,
'mantel': 21,
'monitor': 22,
'night_stand': 23,
'person': 24,
'piano': 25,
'plant': 26,
'radio': 27,
'range_hood': 28,
'sink': 29,
'sofa': 30,
'stairs': 31,
'stool': 32,
'table': 33,
'tent': 34,
'toilet': 35,
'tv_stand': 36,
'vase': 37,
'wardrobe': 38,
'xbox': 39
}

syn_id_name_dict = {
0: 'airplane',
1: 'bathtub',
2: 'bed',
3: 'bench',
4: 'bookshelf',
5: 'bottle',
6: 'bowl',
7: 'car',
8: 'chair',
9: 'cone',
10: 'cup',
11: 'curtain',
12: 'desk',
13: 'door',
14: 'dresser',
15: 'flower_pot',
16: 'glass_box',
17: 'guitar',
18: 'keyboard',
19: 'lamp',
20: 'laptop',
21: 'mantel',
22: 'monitor',
23: 'night_stand',
24: 'person',
25: 'piano',
26: 'plant',
27: 'radio',
28: 'range_hood',
29: 'sink',
30: 'sofa',
31: 'stairs',
32: 'stool',
33: 'table',
34: 'tent',
35: 'toilet',
36: 'tv_stand',
37: 'vase',
38: 'wardrobe',
39: 'xbox'
}

def get_id(npy_path):
	for name in syn_name_id_dict:
		if name in npy_path:
			return syn_name_id_dict[name]
	print 'not found', npy_path
	exit(0)

if __name__ == '__main__':
	if not len(sys.argv) == 3:
		print "usage: python x.py npy_list npy_list_with_synid"
		exit(0)

	npylist_path = sys.argv[1]
	output_npylist_path = sys.argv[2]

	with open(output_npylist_path,'w') as out_f:
		for line in open(npylist_path, 'r'):
			npy_path = line.strip()
			cid = get_id(npy_path)
			out_f.write('%s %d\n' % (npy_path,cid))
	print 'created', output_npylist_path










