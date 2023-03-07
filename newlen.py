import os
import shutil
import math

#=================================================================#
L=101 # Length of shaft in pixel 
#=================================================================#
#Add path where data is stored
data_dir="E:/vsv/Yolo_Final_Test/new ann/NewData/data"
#=================================================================#
#Add path where newdata to be stored
dst_dir="E:/vsv/Yolo_Final_Test/new ann/NewData/Images"
#=================================================================#


ff=[]
imff=[]
os.chdir(data_dir)

get_cwd=os.getcwd()

for (dirname, dirs, files) in os.walk(data_dir):
    for filename in files:
        if filename.endswith('.txt'):
            ff.append(filename)
        else:
            imff.append(filename)

for z in range(0,len(ff)):
    
    os.chdir(get_cwd)
    shutil.copy(imff[z], dst_dir)
    print(z+1)
    
    f = open ( ff[z] , 'r')
    
    Annotation_files = f.read()
    final_ann=[]      #overall matrix

    dataList = Annotation_files.split('\n')
    for i in range(0, len(dataList)-1, 1):
        temp_a=[]  #temp arr
        
        a = str.split(dataList[i])
        x0, y0, alpha, phi = (float(a[1])), (float(a[2])), (float(a[3])), (float(a[4]))

        #calculation for drawing line considering phi and alpha
        c= float((L/2)*(math.cos(alpha)*math.cos(phi)))  #L=shaft length in pixel
        d= float((L/2)*(math.sin(alpha)*math.cos(phi)))        
        
        #calculating new coordinates w.r.t. origin of shaft
        c=float(x0+(c))
        d=float(y0+(d))
        
        for j in range(0,5): #0 x0 y0 x1 y1 c
            if j==0:
                temp_a.append(a[j])        
            if j==1:
                temp_a.append(str(x0))
            if j==2:
                temp_a.append(str(y0))

            if j==3:
                temp_a.append(str(c))
            if j==4:
                temp_a.append(str(d))

                conf= float (a[5])* float(a[6]) #calculating total confidence
                temp_a.append(str(conf))
    
        if conf>0.6:
            final_ann.append(temp_a) # appending data only is total confidence is greater than 0.60
    
    ## Writing Txt
    os.chdir(dst_dir)
    with open(ff[z], "w") as txt_file:
        for line in final_ann:
            txt_file.write(" ".join(line) + "\n")
    


