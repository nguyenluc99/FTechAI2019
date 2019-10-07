def checkNum() :
    try:
        num = int(input('insert a number\t'))
        if num % 2 == 0 :
            print("This is an even number\n")
        else : print("This is an odd number\n")
    except : print("This is not an integer\n")
        
checkNum()
