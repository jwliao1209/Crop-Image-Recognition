clc;clear;close all

loc = "D:\PrivateYC\AICUP\dataset";
Visualize = 0;
WriteDuplicate = 1;

FolderList = dir("dataset");FolderList  = FolderList(3:end);
TotalCtr = 0;
for  i = 1:numel(FolderList)
    folder = FolderList(i).name;
    if WriteDuplicate;diary(fullfile("Info", [folder '.txt']));end
    fprintf("===================================\n");
    fprintf("Folder:%10s\n",folder);
    fprintf("===================================\n");
    if WriteDuplicate;diary off;end

    FL = dir(fullfile(loc, folder, '*.jpg'));
    L = numel(FL);
    FS = zeros(L,1);
    ctr=0;
    for ii = 1: L
        filename = FL(ii).name;
        filesize = FL(ii).bytes;
        FS(ii) = filesize; % collect the bytes of image
    end
    [Uni, Uniidx] = unique(FS);
    [Bcount,BGroup] = groupcounts(FS);
    % check how many groups' number is bigger than 1
    ByteAndNumber = [BGroup(Bcount>1) Bcount(Bcount>1)];
%%

    if WriteDuplicate;diary on;end
    for bytesnum = ByteAndNumber(:,1).'
        % bytesnum = 7388276;
        fprintf("Bytes:%10d\n",bytesnum);
        for number = find(FS==bytesnum).'
            TotalCtr=TotalCtr+1;
            name = FL(number).name;
            if Visualize
                figure('Name', name);
                tmp2 = imread(fullfile(loc, folder, name));
                imshow(tmp2);
            end
            fprintf(".\\dataset\\%s\\%s\n",folder, name);
        end

        if Visualize
            pause()
            close all
        end
%     TotalCtr=TotalCtr-1;
    end
    if WriteDuplicate;diary off;end
end