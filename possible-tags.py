__author__ = 'arenduchintala'
import sys
import codecs

if __name__ == "__main__":
    f = sys.argv[1]
    possible_tags = {}
    for line in codecs.open(f, 'r', 'utf-8').readlines():
        item = line.split()
        for i in item:
            [token, tag] = i.split('_')
            t_set = possible_tags.get(token, set([]))
            t_set.add(tag)
            possible_tags[token] = t_set

    for token, tags in possible_tags.items():
        pre = "PRE_" + token[:3]
        suf = "SUF_" + token[-3:]
        # print token + '\t' + pre + '\t' + suf  # '\t' + '\t'.join(tags)
        print token + '\t' + '\t'.join(tags)