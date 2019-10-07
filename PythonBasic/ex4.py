def convertNum(num) : 
    outNum = 0
    while num > 0 :
        outNum = outNum * 10 + num % 10
        num = num / 10
    return outNum

def main() :
    try :
        num = int(input("Insert a number\t"))
        print("conversion of {}, is {}".format(num, int(str(num)[::-1])))
    except : print("Error happened")

main()