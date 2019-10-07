def calFactorial(num) :
    if num == 0 :  
        return 1
    else : return num * calFactorial(num - 1)

def main():
    try :
        num = int(input('insert a number\t'))
        print("its factorial is : ", calFactorial(num))
    except : print("wrong input")

main()
