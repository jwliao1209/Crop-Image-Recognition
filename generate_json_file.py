import os
import json
import glob

def counter(datatype='public'): # list all data number in each folder
    ctrfile = 0
    for folder in sorted(os.listdir(os.path.join('dataset', datatype))):
        if ".csv" in folder :
            continue
        locpath = os.path.join('dataset', datatype, folder)        
        files = os.listdir(locpath)
        ctrfile+=len(files)
        # print(files)
        print(f"{datatype} | foldname:{folder:3s}, images:{len(files):6d}, accumulate:{ctrfile:6d}")
        
counter(datatype='public')
counter(datatype='private')


# check wether the private dataset have split more than one folder like public
datatype = 'public'
publst = glob.glob(os.path.join('dataset', datatype, '*', '*.jpg'))
prilst = glob.glob(os.path.join('dataset', 'private', '*', '*.jpg'))
def changeslash(element, fslash='\\', tslash='/'):
    return element.replace(fslash, tslash)
pub_targetlst = list(map(changeslash, publst))
pri_targetlst = list(map(changeslash, prilst))


def prepare_set_linux(targetlst, datatype="public", absloc=None):
    target_list = []
    if absloc:
        absloc='/'+absloc
    else:
        absloc=''
    
    for i in range(len(targetlst)):
        target_list.append({"filepath": absloc  + targetlst[i],
                           "image":  absloc  + targetlst[i],
                           "label": int(-1),
                           })
    return target_list

absloc = 'neodata/pathology_breast/aidea'
public_list = prepare_set_linux(pub_targetlst, datatype='public')
private_list = prepare_set_linux(pri_targetlst, datatype='private')
total = public_list + private_list
diction = {"public_and_private":total}
print(f"total length:{len(total)} in jw")
# json.dump(total, open(f"{datatype}_yj.json", "w"), indent=4)
json.dump(diction, open(f"{datatype}_jw.json", "w"), indent=4)

public_list = prepare_set_linux(pub_targetlst, datatype='public', absloc=absloc)
private_list = prepare_set_linux(pri_targetlst, datatype='private', absloc=absloc)
total = public_list + private_list
print(f"total length:{len(total)} in yj")
json.dump(total, open(f"{datatype}_yj.json", "w"), indent=4)