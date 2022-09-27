import os
import glob
import numpy as np
import pandas as pd

np.random.seed(42)

folders = os.listdir('dataset')
Train_ratio = 0.8
Valid_ratio = 0.1

def print_label_with_name():
    folders = os.listdir('dataset')
    Folder = []
    for folder in folders:
        if os.path.isdir(os.path.join("dataset", folder)):
           Folder.append(folder)
    print(len(Folder))
    L = len(Folder)
    Folder = sorted(Folder)
    Dict = dict(zip(np.arange(1, L+1), Folder))
    # maps = lambda x: (np.uint8(x.split('-')[1]), x.split('-')[0])
    # tp = dict(map(maps, Folder))
    # tp = dict(sorted(tp.items(), key=lambda x:x[0]))
    print(Dict)
    return Dict
    
def counter(): # list all data number in each folder
    ctrfile = 0
    for folder in os.listdir('dataset'):
        locpath = os.path.join('dataset', folder)
        files = os.listdir(locpath)
        ctrfile+=len(files)
        # print(files)
        print(f"foldname:{folder:20s}, images:{len(files):6d}, accumulate:{ctrfile:6d}")


def split_data(Train_ratio=0.8, Valid_ratio=0.1, csv_ID=0, folders=folders):
    if Train_ratio + Valid_ratio>1:
        raise AssertionError("R U Kidding Me?")

    TotalTypes = []
    Totalfolder = []
    Totalfile = []

    for idx, folder in enumerate(folders):
        files = os.listdir(os.path.join("dataset", folder))
        L = len(files)
        train_num = round(L*Train_ratio)
        valid_num = round(L*Valid_ratio)
        if Train_ratio+Valid_ratio!=1:
            Test_ratio = 1-(Train_ratio+Valid_ratio)
            test_num = L-train_num-valid_num
            Types = ['train' for _ in range(train_num)] + ['valid' for _ in range(valid_num)] + ['test' for _ in range(test_num)]
        else:
            Test_ratio = 0
            test_num = L-train_num-valid_num
            Types = ['train' for _ in range(train_num)] + ['valid' for _ in range(valid_num)]
        np.random.shuffle(Types)
        folder = [folder for _ in range(L)]
        
        TotalTypes  += Types
        Totalfolder += folder
        Totalfile   += files

    TotalL = len(Totalfile)
    tp = np.array([TotalTypes, Totalfolder, Totalfile]).T

    df = pd.DataFrame(tp, columns=["Type", "folder", "filename"])
    df.to_csv(os.path.join("index", f'fold_{csv_ID}.csv'), index=False)
    print("Finish splitting!")
    print(f"There are {TotalL} images, we split it into Training, Validation and Testing in \n ({Train_ratio*100:.0f}%){TotalL*Train_ratio:.0f}, ({Valid_ratio*100:.0f}%){TotalL*Valid_ratio:.0f} and ({(Test_ratio)*100:.0f}%){TotalL*Test_ratio:.0f}, respectively.")
    print(f"The csv file is saved in {os.path.join('index', f'fold_{csv_ID}.csv')}")

if __name__=="__main__":
    counter()
    print_label_with_name()
    split_data(Train_ratio=0.8, Valid_ratio=0.1, folders=folders)













