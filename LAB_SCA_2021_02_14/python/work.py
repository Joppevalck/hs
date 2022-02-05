import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat


ctfile1 = "ciphertext-00112233445566778899aabbccddeeff.txt"
ctfile2 = "ciphertext-unknown_key.txt"
ptfile1 = "plaintext-00112233445566778899aabbccddeeff.txt"
ptfile2 = "plaintext-unknown_key.txt"
# trfile1 = "traces-00112233445566778899aabbccddeeff.bin"
trfile1 = "traces-00112233445566778899aabbccddeeff.npy"
# trfile2 = "traces-unknown_key.bin"
trfile2 = "traces-unknown_key.npy"


Sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

#........................................................................................................................................................................................................
#..........................................................................................................................,%,...................................................../(....................
#........................................................................................................................./#........%@#.......,%@&*............../@&........*@@(....,&,..................
#........................................................................................................................,&........../@@,...*@@%..................*@@*....,@@#.......*%..................
#..................................................................................,@@@@@@@(...&@@@@@@&...&@,....*@&.....%(............&@/(@@/.....................,@@(.,&@%..........&*.................
#.................................................................................(@&...../..(@%.....,@&../@(...#@(......&/.............%@@*.........................%@@@&............%(.................
#.................................................................................@@........,@@,...../@%..,@@.,@@,.......&/..........*@@%,@@*.........................&@*.............%(.................
#.................................................................................&@*...(@@..&@/...,%@&....#@#@%.........##........(@@(....%@#.......................(@&..............&*.................
#..................................................................................(@@@@%,....*&@@@@(......,@@/...........&,.....%@@*.......*@@,......./@&...........&@,............./#..................
#.........................................................................................................................,%...........................(&...........................,%...................
#........,&/-#-&&,..................................&@@@@@@@@@@@&...........................................................**.....................................................(.....................
#.......,@%.....%@,..................................................@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@....
#.......*@%.....#@,...#/....*%*.........#*....*%*...%@@@@@@@@@@@&........................................................................................................................................
#.......*@@%...,@/.....#&.(@*............@#.,&#.................................,##################################################################################################################......
#.......*@#.............&@*...............#@&...................................(........................................................................................................................
#.......*@#..........,@#..%%..............#%...................................*/...........................................%@%......*@@#............................................&@(.......&@%.......
#.................................%,...........................................(...../*......//,..,(%&%(,....,/,,#%(.........*@@,..(@@/........*/,.....,/*..,/#%%#*...../*.(%#........#@&....%@&,........
#............................................................................./*.....&@.....&@,./@@/.../@&...%@@(..............&@@@&,........../@(....(@#.,@@(,..*@@...(@@#,.,........./@&,%@&,..........
#.....................................................................,**@....(......(@(../@%..../%&&&@@@#..*@&...............*@@@@............,@@...&@,...*%&&&@@@&..,@@,..............,@@@,............
#........................................................................*@,.(........@&.%@*...%@%.....@@,..&@,.............(@@/..@@/...........#@/*@%.../@&.....%@/..(@/................@@..............
#.........................................................................,@//........#@@&.....#@@/*(@@@&..*@#............%@@,.....#@%..........,@@@/..../@@(*/&@@@..,@@................%@#..............
#..........................................................................*%....................,/*.......................................................,**,..........................................
#........................................................................................................................................................................................................

#################################################
#                                               #
#                                               #
#        C O D E   S T A R T S   H E R E        #
#                                               #
#                                               #
#################################################

tohex = np.vectorize(lambda x: int(x, 16))

def myin(filename):
    with open(filename, 'r') as f:
        data = f.readlines()[:-1]
    return tohex(np.array([line[:-2].split(" ") for line in data]))    

def myload(filename):
    return np.load(filename)

def myload2(fname,trlen=370000,start=0,len=370000,n=200):
    myfile = open(fname, 'rb')
    traces = np.zeros((n, len))
    myfile.seek(start)
    for i in range(n):      
        if len+start > trlen:
            t = [int.from_bytes(myfile.read(1), byteorder='big') for i in range(len-start)]
        else:
            t = [int.from_bytes(myfile.read(1), byteorder='big') for i in range(len)]
        traces[i] = t
    myfile.close()
    return traces

def mycorr(x,y):
    xr,xc = x.shape
    yr,yc = y.shape
    assert xr==yr, "Matrix row count mismatch"     

    x = x - x.mean(0)
    y = y - y.mean(0)
    C = x.T.dot(y)
    xsq = np.atleast_2d(np.sqrt(np.sum(x**2, 0)))
    ysq = np.atleast_2d(np.sqrt(np.sum(y**2, 0)))
    C = np.divide(C, repmat(xsq.T, 1, yc))
    C = np.divide(C, repmat(ysq, xc, 1))
    return C

# XOR
def getxor(pt):
    rows = np.shape(pt)[0]
    xor = np.empty((16, rows, 256))                # empty array
    for i in range(16):                         # pt_col, one per round key
        for j in range(rows):                    # pt_row, 200 per round key
            for k in range(256):                # round key guess, 256 per [row, col]
                xor[i, j, k] = (pt[j, i])^k     # bitwise xor operation
    return xor

# Sbox lookups
def getsbox(xor):
    rows = np.shape(xor)[1]
    sbox_vals = np.empty((16, rows, 256))                                   # empty array
    for i in range(16):                                                     # pt_col, one per round key
        for j in range(rows):                                               # pt_row, 200 per round key
            for k in range(256):                                            # intermediate values (xor output)
                sbox_vals[i, j, k] = Sbox[xor[i, j, k].astype("uint32")]    # mapping from intermediate value to sbox output
    return sbox_vals
    
# Hemming weight power model (count 1's)
def gethw(sbox_vals):
    rows = np.shape(sbox_vals)[1]
    hemming = np.empty((16, rows, 256))                                                     # empty array
    for i in range(16):                                                                     # pt_col, one per round key
        for j in range(rows):                                                               # pt_row, 200 per round key
            for k in range(256):                                                            # round key guess, 256 per [row, col]
                hemming[i, j, k] = bin(sbox_vals[i, j, k].astype("uint32")).count("1")      # Number of ones for each sbox output
    return hemming

# Prints out secret key
def printrk(hemming, traces):
    for RK in range(16): 
        CC = mycorr(hemming[RK], traces)            # calculate CC
        max_corr = np.max(CC)                       # find max value of correlation
        result = np.where(CC == max_corr)[0][0]     # find indices representing max CC value
                                                    # result[0, 0] will give row of CC array, ie RK guess)
        print("RK[",RK,"]:\t", format(int(result), '#04x'))
        # plt.plot(CC)
        # plt.show()

#TODO:
#Select which files to open. Filenames defined at the top.
#Functions for loading are named the same as in the Matlab code.
tr1 = myload(trfile1)   # known key
pt1 = myin(ptfile1)     # known key

tr2 = myload(trfile2)   # unknown key
pt2 = myin(ptfile2)     # unknown key

#Use this to check that you loaded the files correctly
print("tr1", tr1.shape)
print("tr2", tr2.shape)
print("pt1", pt1.shape)
print("pt2", pt2.shape)

#TODO:
#After doing the next part you can come back here and change the
#start and stop values to remove the parts of the trace we don't need.
#E.g. let's make up some numbers. If the leakage window starts at x = 123456
#and the window ends at x = 246912 you set start and stop to those values
#respectively
tr1_start = 45140
tr1_stop = tr1_start + 60000
tr1_traces = tr1[:, tr1_start:tr1_stop]

tr2_start = 0
tr2_stop = tr2_start + 30000
tr2_traces = tr2[:, tr2_start:tr2_stop]


#TODO:
#plot one of the power traces.
#Try to determine if you can see the 10 rounds of AES
#in the traces. Also, try to determine if keybytes are
#calculated in series (8-bit operations), 4 at a time
#(32-bit operations), or all parallely.
# plt.plot(tr1_traces[0])
plt.plot(tr2_traces[0])

#TODO:
#Plot only the first round (or two) of AES.
#Also plot vertical lines around where you think we will
#find the information leakage. To plot vertical lines
#you can use the command axvline(...)
# plt.axvline(x=12000, color="r", ymin=0, ymax=250)
# plt.axvline(x=29000, color="r", ymin=0, ymax=250)
plt.axvline(x=6200, color="r", ymin=0, ymax=250)
plt.axvline(x=14000, color="r", ymin=0, ymax=250)
plt.show()

#You may wish to delimit your traces around where
#the leakage point is to speed up computations later

#Tip: to get resizeable popup plots in jupyter notebook
#you can tell the backend to use a library such as 
#qt to handle the plotting in a new window. 
#Test this following code to see how:
#
#%matplotlib qt 
#plt.plot([1,2,3])

#TODO:
#for each possible value of the keybytes, formulate
#a power hypothesis. Make sure you understand what
#the power hypothesis represents and why we need it
#
#The Sbox for AES is provided in case you wish to 
#use it for your power hypothesis


# Known key
# known_xor = getxor(pt1)
# known_sbox = getsbox(known_xor)
# known_hemming = gethw(known_sbox)

# Unknown key 
uk_xor = getxor(pt2)
uk_sbox = getsbox(uk_xor)
uk_hemming = gethw(uk_sbox)

#TODO:
#for each of the 16 subkeys, use your power hypothesis
#to calculate CC using the mycorr function.
#This function is an exact copy of the matlab function.
#
#The instructions say to make sure you know what CC is.
#Obviously one of the Cs is correlation. Which of these
#describes what CC is:
#1) Cross Correlation 
#2) Correlation Coefficients <----------
#3) Cumulative Correlation
#4) Central Correlation
#
#What shape would you expect the CC variable to have?
#Try plotting its shape and see if your guess was correct.


# print("known key")
# printrk(known_hemming, tr1_traces)

print("\nunknown key")
printrk(uk_hemming, tr2_traces)

    #TODO:
    #Write code to find the correct keybyte.
    #You will want to print it. Depending on your code
    #you may have to print it inside the loop.

    
#NOTE: The amount of code you need to write in this section
#is very small. It can reasonably be done in 4 lines of
#code or less...