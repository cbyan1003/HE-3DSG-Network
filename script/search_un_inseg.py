import os, sys

files_path = "/home/ycb/Scannet_dataset/data/dataset/scans/"
files_path_in = os.listdir(files_path)
flag = False
for files in files_path_in:
    senss = os.listdir(os.path.join(files_path,files))
    for sen in senss:
        inseg = "inseg.ply"
        if os.path.exists(os.path.join(files_path,files,inseg)):
            flag = True
    if(flag==False):
        print(files)
    else:
        flag = False
        
    #   if os.path.splitext(sen)[1] == ".sens":
    #     # load the data
    #     sys.stdout.write('loading %s.........' % os.path.join(opt.filename,files,sen))
    #     sd = SensorData(os.path.join(opt.filename,files,sen))
    #     sys.stdout.write('loaded!\n')
    #     print(os.path.join(opt.filename,files))
    #     if opt.export_intrinsics:
    #       sd.export_intrinsics(os.path.join(opt.filename,files))