import os
import json
import glob

# check wether the private dataset have split more than one folder like public
datatype = 'public'
lst = glob.glob(os.path.join('dataset', datatype, '*', '*.jpg'))
def changeslash(element, fslash='\\', tslash='/'):
    return element.replace(fslash, tslash)
targetlst = list(map(changeslash, lst))

def prepare_set_linux(targetlst, datatype="public"):
    target_list = []
    for i in range(len(targetlst)):
        target_list.append({"filepath": '/' + targetlst[i],
                           "image":  '/' + targetlst[i],
                           "label": int(-1), 
                           })
    diction = {f"{datatype}": target_list}
    return diction

diction = prepare_set_linux(targetlst, datatype=datatype)
json.dump(diction, open(f"{datatype}_.json", "w"), indent=4)