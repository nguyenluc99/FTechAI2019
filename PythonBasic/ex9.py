from datetime import datetime

def getName(iniStr) :
    now = datetime.timestamp(datetime.now()) * 10**6
    return iniStr + str(int(now))

print("new name is : ",getName("abc"))

    