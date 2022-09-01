# Python 3 code to rename multiple
# files in a directory or folder
 
# importing os module
import os
 
# Function to rename multiple files
def main():
   
    folder = os.getcwd()
    count=0
    for filename in (os.listdir(folder)):
         if(filename[-4:]=='.mp4'):
# rename the original file
           os.rename(filename,'NV_'+str(count)+'.mp4')
           count+=1
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()
