'''the code that integrate fitting results'''
import os,sys
import csv

def txt_gather():
    '''Merge txt files'''
    file_saves_name = os.listdir("./fitting_results/")
    #File name list under "path"
    for file_save_name in file_saves_name:
        file_save = "./fitting_results/" + file_save_name
        #Folder path under "path"
        f = open(file_save+'/gather_before_arrange.txt', 'w')
        for i in range(8):
            txt_path=file_save+"/print_"+str(i)+".txt"
            #Path of the specific txt file
            for line in open(txt_path, encoding='gbk'):
                f.writelines(line)
                #write gather.txt
        f.close()

def csv_gather():

    file_saves_name = os.listdir("./fitting_results/")
    #File name list under "path"
    for file_save_name in file_saves_name:
        file_save = "./fitting_results/" + file_save_name
        f = open(file_save + '/profile_gather_before_arrange.csv', 'w')
        for i in range(8):
            csv_path = file_save + "/Profile_u_all_" + str(i) + "_net.csv"
            for line in open(csv_path, encoding='gbk'):
                f.writelines(line)
                # write gather.csv
        f.close()

def arrange():
    '''Take out by z0, and then form txt in the original order'''
    file_saves_name = os.listdir("./fitting_results/")
    #Read the original height sort
    f=open("./symmetried_shapes.csv", "r")
    profile = list(csv.reader(f))
    heights=[]
    # read csv
    for i in range(int(len(profile)/2)):
        heights.append(list(map(float, list(filter(lambda x: x != '', profile[i*2+1]))))[0])
    f.close()

    for file_save_name in file_saves_name:
        file_save = "./fitting_results/" + file_save_name
        f = open(file_save + '/gather_before_arrange.txt', 'r')

        #Txt fitting data stored in list
        train_list=[]
        train_lists=[]
        z0=[]
        s = f.readlines()
        for x in s:
            if x.__contains__("z0:"):
                start = x.find("z0:") + len("z0:")
                end = x.find("00,")
                z0.append(float('%.3f' % float(x[start:end])))
                # Convert the string to float and divide it to two decimal places
            if x.__contains__("Trainning ended"):
                train_list.append(x)
                train_lists.append(train_list)
                train_list = []
            elif x == '\n': continue
            else:train_list.append(x)
        f.close()

        # Index arrangement of the original order
        indexs=[]
        for height in heights:
            for i in range(len(z0)):
                if(height==z0[i]):
                    indexs.append(i)
                    break

        # The txt training data originally not sorted by height
        original_train_list=[]
        for index in indexs:
            original_train_list.append(train_lists[index])


        # Write the final gather.txt
        f = open(file_save + '/gather.txt', 'w')
        for L in original_train_list:
            for line in L:
                f.writelines(line)
        f.close()

        #Save csv training data to list
        f = open(file_save + '/profile_gather_before_arrange.csv', 'r')
        profile = list(csv.reader(f))
        for i in range(len(profile)):
            profile[i] = list(map(float, list(filter(lambda x: x != '', profile[i]))))
        f.close()

        # Csv training data originally not sorted by height
        original_train_list = []
        for index in indexs:
            for i in range(6):
                original_train_list.append(profile[index*6+i])

        # Write the final gather.csv
        f = open(file_save + '/profile_gather.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerows(original_train_list)
        f.close()

if __name__ == '__main__':

    '''integrate fitting results'''
    txt_gather()
    csv_gather()
    arrange()





