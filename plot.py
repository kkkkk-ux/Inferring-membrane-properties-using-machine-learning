
import numpy as np
from matplotlib import pyplot as plt
import csv,os

c_red = [237 / 255, 28 / 255, 36 / 255]
c_orange = [255 / 255, 169 / 255, 5 / 255]
c_green = [116 / 255, 212 / 255, 7 / 255]
c_cyan = [8 / 255, 228 / 255, 255 / 255]
c_purple = [86 / 255, 4 / 255, 212 / 255]
c_magenta = [1 / 255, 0 / 255, 1 / 255]
c_blue = [0 / 255, 191 / 255, 255 / 255]

def Fig_3(index,figsize):

    #Fig3a&b
    loss = []
    sigma = []
    p = []
    epoch = []
    loss_BVP = []
    Rbs = []
    flag=0;
    with open("./fitting_results/save_16group_"+str(index)+"/gather.txt", "r") as f:
        s = f.readlines()
        for x in s:
            if x.__contains__("Start trainning,z0:71.176"):
                flag=1;
                continue
            if flag:
                if x.__contains__("0: "):
                    start = x.find("0: ") + len("0: ")
                    end = x.find(" , sigma=")
                    loss.append(x[start:end])
                if x.__contains__("epoch "):
                    start = x.find("epoch ") + len("epoch ")
                    end = x.find(":")
                    epoch.append(x[start:end])
                if x.__contains__("sigma="):
                    start = x.find("sigma=") + len("sigma=")
                    end = x.find(",p=")
                    sigma.append(x[start:end])
                if x.__contains__("p="):
                    start = x.find("p=") + len("p=")
                    end = x.find(",rb=")
                    p.append(x[start:end])
                if x.__contains__("loss_BVP="):
                    start = x.find("loss_BVP=") + len("loss_BVP=")
                    end = x.find(",loss_BVP2")
                    loss_BVP.append(x[start:end])
                if x.__contains__("rb="):
                    start = x.find("rb=") + len("rb=")
                    end = x.find(",loss_BVP=")
                    Rbs.append(x[start:end])

            if x.__contains__("Trainning ended"):
                flag=0
        f.close()

    fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=figsize)
    ax1.plot(np.array(list(map(float, epoch))) / 1e4, np.log10(list(map(float, loss))), label='log10(Ltotal)',
             color=c_blue, marker='o')
    ax1.plot(np.array(list(map(float, epoch))) / 1e4, np.log10(list(map(float, loss_BVP))), label='log10(Ldata)',
             color=c_green, marker='o')
    ax1.set_xlabel("epoch (×1e4)", fontdict={'size': 17})
    ax1.tick_params(labelsize=17)  # 刻度字体大小13
    ax1.legend()

    ax2.plot(np.array(list(map(float, epoch))) / 1e4, np.array(list(map(float, sigma))) * 8e-1 / 10,
             label='σ/10 (pN/nm)', color=c_blue, marker='o')
    ax2.plot(np.array(list(map(float, epoch))) / 1e4, np.array(list(map(float, p))) * 8e4 / 1e6, label='p (MPa)',
             color=c_red, marker='o')
    ax2.plot(np.array(list(map(float, epoch))) / 1e4, np.array(list(map(float, Rbs))) / 50, label='Rb/50 (nm)',
             color=c_green,
             marker='o')
    ax2.set_xlabel("epoch (×1e4)", fontdict={'size': 17})
    ax2.tick_params(labelsize=17)  # 刻度字体大小13
    ax2.legend()

    #Fig3c
    origin = list(
        csv.reader(open("original_shapes.csv", "r")))  # 读取csv
    for i in range(len(origin)):
        origin[i] = list(map(float, list(filter(lambda x: x != '', origin[i]))))
    # 读取对称数据
    symmetry = list(csv.reader(open("symmetried_shapes.csv", "r")))  # 读取csv
    for i in range(len(symmetry)):
        symmetry[i] = list(map(float, list(filter(lambda x: x != '', symmetry[i]))))
    # 读取网络拟合数据,含sigma等参数

    net = list(csv.reader(open("./fitting_results/save_16group_" + str(index) +"/profile_gather.csv", "r")))  # 读取csv
    for i in range(len(net)):
        net[i] = list(map(float, list(filter(lambda x: x != '', net[i]))))
    # 读取BVP数据
    BVP = list(csv.reader(
        open("./fitting_results/save_16group_" + str(index) + "/profile_BVP.csv",
             "r")))
    for i in range(len(BVP)):
        BVP[i] = list(map(float, list(filter(lambda x: x != '', BVP[i]))))
    # 读取Force数据
    F = list(csv.reader(
        open("./fitting_results/save_16group_" + str(index) + "/profile_Force.csv",
             "r")))
    for i in range(len(F)):
        F[i] = list(map(float, list(filter(lambda x: x != '', F[i]))))


    i = 48 - 1
    ax3.plot(origin[2 * i], origin[2 * i + 1], label='Original shape', color=c_blue, linestyle='solid', linewidth=2)

    ax3.plot(symmetry[2 * i], symmetry[2 * i + 1], label='Symmetrized shape', color=c_green, linestyle='dashdot',
            linewidth=2)
    ax3.plot(-np.array(symmetry[2 * i]), symmetry[2 * i + 1], color=c_green, linestyle='dashdot', linewidth=2)

    ax3.plot(net[6 * i + 4], net[6 * i + 1 + 4], label='ML shape', color=c_orange, linestyle='dashed', linewidth=2)
    ax3.plot(-np.array(net[6 * i + 4]), net[6 * i + 1 + 4], color=c_orange, linestyle='dashed', linewidth=2)

    ax3.plot(BVP[2 * i], BVP[2 * i + 1], label='FD shape', color=c_red, linestyle='dotted', linewidth=2)
    ax3.plot(-np.array(BVP[2 * i]), BVP[2 * i + 1], color=c_red, linestyle='dotted', linewidth=2)

    ax3.set_title("p=%.2fMPa,sigma=%.2fpN/nm,f=%.2fpN" % (
    np.array(net[6 * i + 0]) * 8e4 / 1e6, np.array(net[6 * i + 1]) * 8e-1, np.array(F[i][0]) * 8e3 * 2 * np.pi),
                 fontdict={'size': 17})
    ax3.set_ylabel("z (nm)", fontdict={'size': 17})
    ax3.set_xlabel("r (nm)", fontdict={'size': 17})
    ax3.legend()
    ax3.axis('equal')
    ax3.tick_params(labelsize=17)  # 刻度字体大小13

    # plt.savefig("./fit" + str(index) + "/figure/Profile_Force.pdf")
    # plt.show()

def Fig_5(index,figsize):
    # 读取原始数据
    origin = list(
        csv.reader(open("original_shapes.csv", "r")))  # 读取csv
    for i in range(len(origin)):
        origin[i] = list(map(float, list(filter(lambda x: x != '', origin[i]))))
    # 读取对称数据
    symmetry = list(csv.reader(open("symmetried_shapes.csv", "r")))  # 读取csv
    for i in range(len(symmetry)):
        symmetry[i] = list(map(float, list(filter(lambda x: x != '', symmetry[i]))))
    # 读取网络拟合数据,含sigma等参数
    net = list(csv.reader(open("./fitting_results/save_16group_"+str(index)+"/profile_gather.csv", "r")))  # 读取csv
    for i in range(len(net)):
        net[i] = list(map(float, list(filter(lambda x: x != '', net[i]))))
    # 读取BVP数据
    BVP = list(csv.reader(
        open("./fitting_results/save_16group_"+str(index)+"/profile_BVP.csv",
             "r")))
    for i in range(len(BVP)):
        BVP[i] = list(map(float, list(filter(lambda x: x != '', BVP[i]))))

    # 读取Force数据
    F = list(csv.reader(
        open("./fitting_results/save_16group_"+str(index)+"/profile_Force.csv",
             "r")))
    for i in range(len(F)):
        F[i] = list(map(float, list(filter(lambda x: x != '', F[i]))))

    fig, axes = plt.subplots(3,3,sharey=True,figsize=figsize)
    j=0
    for k in [76,77,53,60,73,36,48,42,9] :  # i=0表示profile 53
        i=k-1
        ax = axes[int(j/3),j%3]

        ax.plot(origin[2*i],origin[2*i+1],label='Original shape',color=c_blue,linestyle='solid',linewidth=2)

        ax.plot(symmetry[2*i], symmetry[2*i+1], label='Symmetrized shape',color=c_green,linestyle='dashdot',linewidth=2)
        ax.plot(-np.array(symmetry[2*i]), symmetry[2*i+1],color=c_green,linestyle='dashdot',linewidth=2)

        ax.plot(net[6*i+4], net[6*i+1+4], label='ML shape', color=c_orange,linestyle='dashed',linewidth=2)
        ax.plot(-np.array(net[6*i+4]), net[6*i+1+4], color=c_orange,linestyle='dashed',linewidth=2)

        ax.plot(BVP[2*i], BVP[2*i+1], label='FD shape', color=c_red,linestyle='dotted',linewidth=2)
        ax.plot(-np.array(BVP[2*i]), BVP[2*i+1], color=c_red,linestyle='dotted',linewidth=2)

        ax.set_title("p=%.2fMPa,sigma=%.2fpN/nm,f=%.2fpN"%(np.array(net[6*i+0])*8e4/1e6,np.array(net[6*i+1])*8e-1,np.array(F[i][0])*8e3*2*np.pi),
                     fontdict={'size':17})
        if k==76 or k==60 or k==48:
            ax.set_ylabel("z (nm)", fontdict={'size': 17})
        if k==48 or k==42 or k==9:
            ax.set_xlabel("r (nm)", fontdict={'size': 17})
        ax.legend()
        ax.axis('equal')
        ax.tick_params(labelsize=17)  # 刻度字体大小13
        j=j+1
    # plt.show()

def Fig_4():
    file_saves_name = os.listdir("./fitting_results/")
    figsize = (25, 13)
    z0 = []
    sigmas = np.zeros([len(file_saves_name), 79])
    ps = np.zeros([len(file_saves_name), 79])
    losses = np.zeros([len(file_saves_name), 79])

    for index in range(len(file_saves_name)):
        final_loss = []
        sigma = []
        p = []
        loss = []
        with open("./fitting_results/save_16group_" + str(index) + "/gather.txt", "r") as f:
            s = f.readlines()
            for x in s:
                if x.__contains__("Start trainning,z0:") and index == 4:
                    start = x.find("Start trainning,z0:") + len("Start trainning,z0:")
                    end = x.find(",domainsize")
                    z0.append(x[start:end])

                if x.__contains__("400000: "):
                    final_loss.append(x)

            for x in final_loss:
                if x.__contains__("sigma="):
                    start = x.find("sigma=") + len("sigma=")
                    end = x.find(",p=")
                    sigma.append(x[start:end])
                if x.__contains__("p="):
                    start = x.find("p=") + len("p=")
                    end = x.find(",rb=")
                    p.append(x[start:end])
                if x.__contains__("400000: "):
                    start = x.find("400000: ") + len("400000: ")
                    end = x.find(" , sigma=")
                    loss.append(x[start:end])
            f.close()
        sigmas[index] = np.array(list(map(float, sigma)))
        ps[index] = np.array(list(map(float, p)))
        losses[index] = np.array(list(map(float, loss)))

    fig, (ax1) = plt.subplots(1, 1, figsize=figsize)
    '''loss histogram'''
    ax1.hist(losses.flatten(), bins=20, label='loss', color=[0, 191 / 255, 1])
    ax1.set_xlabel("Ltot", fontdict={'size': 17})
    ax1.set_ylabel("Frequency", fontdict={'size': 17})
    ax1.legend()
    ax1.tick_params(labelsize=17)


def fig6():
    file_saves_name = os.listdir("./fitting_results/")
    figsize = (25, 13)
    sigmas = np.zeros([len(file_saves_name), 79])
    ps = np.zeros([len(file_saves_name), 79])
    Fs = np.zeros([len(file_saves_name), 79])

    '''将数据读取到数组'''
    for index in range(len(file_saves_name)):
        final_loss = []
        sigma = []
        p = []
        with open("./fitting_results/save_16group_" + str(index) + "/gather.txt", "r") as f:
            s = f.readlines()
            for x in s:
                if x.__contains__("400000: "):
                    final_loss.append(x)

            for x in final_loss:
                if x.__contains__("sigma="):
                    start = x.find("sigma=") + len("sigma=")
                    end = x.find(",p=")
                    sigma.append(x[start:end])
                if x.__contains__("p="):
                    start = x.find("p=") + len("p=")
                    end = x.find(",rb=")
                    p.append(x[start:end])
            f.close()
        sigmas[index] = np.array(list(map(float, sigma)))
        ps[index] = np.array(list(map(float, p)))

        F = list(csv.reader(
            open("./fitting_results/save_16group_" + str(index) + "/profile_Force.csv",
                 "r")))
        for i in range(len(F)):
            F[i] = float(F[i][0])
        Fs[index] = np.array(F)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.hist(ps.flatten() * 8e-2, bins=20, label='p', color=c_blue)
    ax1.set_xlabel("p (MPa)",fontdict={'size': 17})
    ax1.set_ylabel("Frequency",fontdict={'size': 17})
    ax1.set_title("<p>=%.2f MPa,std=%.2f MPa" % (np.average(ps*8e-2),
                                       np.std(ps*8e-2, ddof=1)),fontdict={'size': 17})

    ax2.hist(sigmas.flatten()* 8e-1, bins=20, label='sigma', color=c_blue)
    ax2.set_xlabel("σ (pN/nm)",fontdict={'size': 17})
    ax2.set_ylabel("Frequency",fontdict={'size': 17})
    ax2.set_title("<σ>=%.2f pN/nm,std=%.2f pN/nm" % (np.average(sigmas*8e-1),
                                    np.std(sigmas*8e-1, ddof=1)),fontdict={'size': 17})

    ax3.hist(Fs.flatten() * 8e3 * 2 * np.pi, bins=20, label='sigma', color=c_blue)
    ax3.set_xlabel("F (pN)",fontdict={'size': 17})
    ax3.set_ylabel("Frequency",fontdict={'size': 17})
    ax3.set_title("<F>=%.2f pN,std=%.2f pN" % (np.average(Fs * 8e3 * 2 * np.pi),
                                np.std(Fs * 8e3 * 2 * np.pi, ddof=1)),fontdict={'size': 17})
    ax1.tick_params(labelsize=17)
    ax2.tick_params(labelsize=17)
    ax3.tick_params(labelsize=17)

def fig7():
    file_saves_name = os.listdir("./fitting_results/")
    figsize = (25, 13)
    z0 = []
    sigmas = np.zeros([len(file_saves_name), 79])
    ps = np.zeros([len(file_saves_name), 79])
    Fs = np.zeros([len(file_saves_name), 79])
    for index in range(len(file_saves_name)):
        final_loss = []
        sigma = []
        p = []
        with open("./fitting_results/save_16group_" + str(index) + "/gather.txt", "r") as f:
            s = f.readlines()
            for x in s:
                if x.__contains__("Start trainning,z0:") and index == 4:
                    start = x.find("Start trainning,z0:") + len("Start trainning,z0:")
                    end = x.find(",domainsize")
                    z0.append(x[start:end])

                if x.__contains__("400000: "):
                    final_loss.append(x)

            for x in final_loss:
                if x.__contains__("sigma="):
                    start = x.find("sigma=") + len("sigma=")
                    end = x.find(",p=")
                    sigma.append(x[start:end])
                if x.__contains__("p="):
                    start = x.find("p=") + len("p=")
                    end = x.find(",rb=")
                    p.append(x[start:end])
            f.close()
        sigmas[index] = np.array(list(map(float, sigma)))
        ps[index] = np.array(list(map(float, p)))

        # Fig7c scatter
        F = list(csv.reader(
            open("./fitting_results/save_16group_" + str(index) + "/profile_Force.csv",
                 "r")))
        for i in range(len(F)):
            F[i] = float(F[i][0])
        Fs[index] = np.array(F)

    index_p = np.std(ps * 8e-2, axis=0, ddof=1).argsort()[-5:][::-1]
    index_sigma = np.std(sigmas * 8e-1, axis=0, ddof=1).argsort()[-5:][::-1]
    index_F = np.std(Fs * 8e3*2*np.pi, axis=0, ddof=1).argsort()[-5:][::-1]

    z0 = np.array(list(map(float, z0)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    ax1.errorbar(z0, np.average(sigmas * 8e-1, axis=0),
                 yerr=np.std(sigmas * 8e-1, axis=0, ddof=1),
                 linestyle="none", marker='o', color='k', mfc=c_blue)
    # ax1.errorbar(z0[index_sigma], np.average(sigmas * 8e-1, axis=0)[index_sigma],
    #              yerr=np.std(sigmas * 8e-1, axis=0, ddof=1)[index_sigma],
    #              linestyle="none", marker='o', color=c_red, mfc=c_green)
    ax1.set_xlabel("z0 (nm)", fontdict={'size': 17})
    ax1.set_ylabel("σ (pN/nm)", fontdict={'size': 17})
    ax1.tick_params(labelsize=17)  # 刻度字体大小13


    ax2.errorbar(z0, np.average(ps * 8e-2, axis=0),
                 yerr=np.std(ps * 8e-2, axis=0, ddof=1),
                 linestyle="none", marker='o', color='k', mfc=c_blue)
    # ax2.errorbar(z0[index_p], np.average(ps * 8e-2, axis=0)[index_p],
    #              yerr=np.std(ps * 8e-2, axis=0, ddof=1)[index_p],
    #              linestyle="none", marker='o', color=c_red, mfc=c_green)
    ax2.set_xlabel("z0 (nm)", fontdict={'size': 17})
    ax2.set_ylabel("p (MPa)", fontdict={'size': 17})
    ax2.tick_params(labelsize=17)  # 刻度字体大小13

    ax3.errorbar(z0, np.average(Fs * 8e3 * 2 * np.pi, axis=0),
                 yerr=np.std(Fs * 8e3 * 2 * np.pi, axis=0, ddof=1),
                 linestyle="none", marker='o', color='k', mfc=c_blue)
    # ax3.errorbar(z0[index_F], np.average(Fs * 8e3 * 2 * np.pi, axis=0)[index_F],
    #              yerr=np.std(Fs * 8e3 * 2 * np.pi, axis=0, ddof=1)[index_F],
    #              linestyle="none", marker='o', color=c_red, mfc=c_green)
    ax3.set_xlabel("z0 (nm)", fontdict={'size': 17})
    ax3.set_ylabel("f (pN)", fontdict={'size': 17})
    ax3.tick_params(labelsize=17)  # 刻度字体大小13

def fig8():
    file_saves_name = os.listdir("./fitting_results/")
    figsize = (25, 13)
    sigmas = np.zeros([len(file_saves_name), 79])
    ps = np.zeros([len(file_saves_name), 79])
    Fs = np.zeros([len(file_saves_name), 79])

    '''将数据读取到数组'''
    for index in range(len(file_saves_name)):
        final_loss = []
        sigma = []
        p = []
        with open("./fitting_results/save_16group_" + str(index) + "/gather.txt", "r") as f:
            s = f.readlines()
            for x in s:
                if x.__contains__("400000: "):
                    final_loss.append(x)

            for x in final_loss:
                if x.__contains__("sigma="):
                    start = x.find("sigma=") + len("sigma=")
                    end = x.find(",p=")
                    sigma.append(x[start:end])
                if x.__contains__("p="):
                    start = x.find("p=") + len("p=")
                    end = x.find(",rb=")
                    p.append(x[start:end])
            f.close()
        sigmas[index] = np.array(list(map(float, sigma)))
        ps[index] = np.array(list(map(float, p)))

        F = list(csv.reader(
            open("./fitting_results/save_16group_" + str(index) + "/profile_Force.csv",
                 "r")))
        for i in range(len(F)):
            F[i] = float(F[i][0])
        Fs[index] = np.array(F)


    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=figsize)

    ax1.errorbar(np.average(sigmas * 8e-1, axis=0),
                np.average(Fs * 8e3 * 2 * np.pi, axis=0),
                xerr=np.std(sigmas * 8e-1, axis=0, ddof=1),
                yerr=np.std(Fs * 8e3 * 2 * np.pi, axis=0, ddof=1),
                linestyle="none", marker='o', color='k', mfc=c_blue)

    #person相关系数是对平均值做
    ax1.set_title("R=%.2f"%(np.corrcoef(np.average(sigmas * 8e-1, axis=0),
                np.average(Fs * 8e3 * 2 * np.pi, axis=0))[0,1]), fontdict={'size': 17})
    ax1.set_ylabel("f (pN)", fontdict={'size': 17})
    ax1.set_xlabel("σ (pN/nm)", fontdict={'size': 17})
    ax1.tick_params(labelsize=17)  # 刻度字体大小13

    ax2.errorbar(np.average(ps * 8e-2, axis=0),
                 np.average(Fs * 8e3 * 2 * np.pi, axis=0),
                 xerr=np.std(ps * 8e-2, axis=0, ddof=1),
                 yerr=np.std(Fs * 8e3 * 2 * np.pi, axis=0, ddof=1),
                 linestyle="none", marker='o', color='k', mfc=c_blue)
    ax2.set_title("R=%.2f"%(np.corrcoef(np.average(ps * 8e-2, axis=0),
                np.average(Fs * 8e3 * 2 * np.pi, axis=0))[0,1]), fontdict={'size': 17})
    ax2.set_ylabel("f (pN)", fontdict={'size': 17})
    ax2.set_xlabel("p (MPa)", fontdict={'size': 17})
    ax2.tick_params(labelsize=17)  # 刻度字体大小13

def Fig_all_fit(index,figsize):
    # plot all of the fitting shape

    # 读取原始数据
    origin = list(
        csv.reader(open("original_shapes.csv", "r")))  # 读取csv
    for i in range(len(origin)):
        origin[i] = list(map(float, list(filter(lambda x: x != '', origin[i]))))
    # 读取对称数据
    symmetry = list(csv.reader(open("symmetried_shapes.csv", "r")))  # 读取csv
    for i in range(len(symmetry)):
        symmetry[i] = list(map(float, list(filter(lambda x: x != '', symmetry[i]))))
    # 读取网络拟合数据,含sigma等参数
    net = list(csv.reader(open("./fitting_results/save_16group_" + str(index) + "/profile_gather.csv", "r")))  # 读取csv
    for i in range(len(net)):
        net[i] = list(map(float, list(filter(lambda x: x != '', net[i]))))
    # 读取BVP数据
    BVP = list(csv.reader(
        open("./fitting_results/save_16group_" + str(index) + "/profile_BVP.csv",
             "r")))
    for i in range(len(BVP)):
        BVP[i] = list(map(float, list(filter(lambda x: x != '', BVP[i]))))

    # 读取Force数据
    F = list(csv.reader(
        open("./fitting_results/save_16group_" + str(index) + "/profile_Force.csv",
             "r")))
    for i in range(len(F)):
        F[i] = list(map(float, list(filter(lambda x: x != '', F[i]))))

    for j in range(0,8):
        start=9*j
        fig, axes = plt.subplots(3,3,sharey=True,figsize=figsize)
        if j<=7:
            interval=9
        else:
            interval=7
        for i in range(start,start+interval):  # i=0表示profile 53

            ax = axes[int((i-start)/3),(i-start)%3]

            ax.plot(origin[2*i],origin[2*i+1],label='Original shape',color=c_blue,linestyle='solid',linewidth=2)

            ax.plot(symmetry[2*i], symmetry[2*i+1], label='Symmetrized shape',color=c_green,linestyle='dashdot',linewidth=2)
            ax.plot(-np.array(symmetry[2*i]), symmetry[2*i+1],color=c_green,linestyle='dashdot',linewidth=2)

            ax.plot(net[6*i+4], net[6*i+1+4], label='ML shape', color=c_orange,linestyle='dashed',linewidth=2)
            ax.plot(-np.array(net[6*i+4]), net[6*i+1+4], color=c_orange,linestyle='dashed',linewidth=2)

            ax.plot(BVP[2*i], BVP[2*i+1], label='FD shape', color=c_red,linestyle='dotted',linewidth=2)
            ax.plot(-np.array(BVP[2*i]), BVP[2*i+1], color=c_red,linestyle='dotted',linewidth=2)

            ax.set_title("p=%.2fMPa,sigma=%.2fpN/nm,f=%.2fpN"%(np.array(net[6*i+0])*8e4/1e6,np.array(net[6*i+1])*8e-1,np.array(F[i][0])*8e3*2*np.pi),
                         fontdict={'size':15})
            ax.legend()
            ax.axis('equal')
            ax.tick_params(labelsize=15)  # 刻度字体大小13

        plt.savefig("./figure/save_16group_" + str(index) + "/all_fit_data" + str(j) + ".pdf")

if __name__ == "__main__":
    file_saves_name = os.listdir("./fitting_results/")
    figsize=(25,13)
    for index in range(len(file_saves_name)):

        # Delete if the file exists, and create if it does not exist
        if os.path.exists("./figure/save_16group_" + str(index)):
            del_list = os.listdir("./figure/save_16group_" + str(index))
            for f in del_list:
                file_path = os.path.join("./figure/save_16group_" + str(index), f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print("delete file:" + file_path)
        else:
            os.mkdir("./figure/save_16group_" + str(index))
            print("create file:" + "./figure/save_16group_" + str(index))

        Fig_3(index,figsize)
        plt.savefig("./figure/save_16group_" + str(index) + "/Fig3.pdf")
        Fig_5(index,figsize)
        plt.savefig("./figure/save_16group_" + str(index) + "/Fig5.pdf")
        Fig_all_fit(index,figsize)
    Fig_4()
    plt.savefig("./figure/Fig4.pdf")
    fig6()
    plt.savefig("./figure/Fig6.pdf")
    fig7()
    plt.savefig("./figure/Fig7.pdf")
    fig8()
    plt.savefig("./figure/Fig8.pdf")
